"""
Combines polygon mask predictions from multiple sources into single
files.

Reads in prediction files from different directories, and combines the
predictions by shared filenames. The predictions are then saved out to a
single file, sharing the same filename as the source files.
"""
# Imports
from typing import Final
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from os import environ
import progressbar

progressbar.streams.flush()
progressbar.streams.wrap_stderr()

# Constants and presets
PROJECT_DIR: Final[Path]
PROJECT_DIR = Path(find_dotenv(".env", 1, 1)).absolute().parent
environ["PROJECT_DIR"] = str(PROJECT_DIR)
load_dotenv(PROJECT_DIR.joinpath(".env"))

FILENAME: Final[str] = Path(__file__).stem
_CONFIG_DIR: Final[Path]
_CONFIG_DIR = PROJECT_DIR.joinpath(f"config/{FILENAME}.json")
_PRESET: Final = dict(relative_path = "PROJECT_DIR")

_WIDGETS: Final[list] = [
    ' [', progressbar.widgets.Counter(format='%(value)d of %(max_value)d'),
    ' (', progressbar.widgets.Percentage(), ')] ',
    progressbar.widgets.GranularBar(),
    ' ', progressbar.Timer(), ' | ',
     progressbar.ETA(), '|'
]

if __name__ == "__main__":
    # Imports
    from json import load as load_json
    from pandas import concat, merge as merge_dataframes
    from geopandas import read_file, GeoDataFrame
    from outputs import normalize_geometries
    from project_utils import parse_path, build_argument_parser, build_logger

    parser = build_argument_parser(
        filename = FILENAME,
        description =\
            "Combines polygon mask predictions from multiple sources into "\
            "single files."
    )
    parser.add_argument(
        "save_dir",
        action = "store",
        type = str,
        metavar = "to/save/dir",
        help =\
            "Required. Specify directory to save GeoDataFrames out to. The "\
            "GeoDataFrames are saved under filenames corresponding to the "\
            "TIFF filenames the text instances belong within. Can provide a "\
            "relative or absolute path; relative paths will be set "\
            "relative to the path variable specified in config."
    )
    parser.add_argument(
        "mask_dirs",
        action = "store",
        nargs = "+",
        type = str,
        metavar = "to/prediction/gpkg/dirs",
        help =\
            "Required. Paths to directories containing ambiguous prediction"\
            "mask directories. Can provide relative or absolute "\
            "paths; relative paths will be set against path variable "\
            "specified in config."
    )
    cla_args = parser.parse_args()

    logger = build_logger(
        stream_level = cla_args.stream_level,
        write_to = PROJECT_DIR.joinpath("logs") if cla_args.file else None,
        filename = FILENAME
    )
    
    try:
        # Try reading config
        f = (
            _CONFIG_DIR
            if cla_args.config is None
            else parse_path(cla_args.config, "PROJECT_DIR")
        )
        with open(f, "r") as f:
            config = {**_PRESET, **load_json(f)}
    
        logger.debug("Parsing read in and save out file paths")
        save_dir = parse_path(cla_args.save_dir, config["relative_path"])
        if not save_dir.exists():
            raise ValueError(
                f"Command line argument for the combined polygon masks to be "\
                f"saved out to does not lead to an existing path. Argument "\
                f"passed: {cla_args.save_dir}"
            )
        if save_dir.is_file():
            raise ValueError(
                f"Command line argument for the combined polygon masks to be "\
                f"saved out to leads to a file, rather than a directory. "\
                f"Value passed in command line: {cla_args.save_dir}"
            )
        mask_dirs = [
            parse_path(el, config["relative_path"])
            for el in cla_args.mask_dirs
        ]
        if not all(p.exists() for p in mask_dirs):
            raise ValueError(
                f"At least on command line argument for the polygon mask "\
                f"directories does not lead to an existing directory. "\
                f"Instance passed: "\
                f"{next((p for p in mask_dirs if not p.exists()))}"
            )
        if any(p.is_file() for p in mask_dirs):
            raise ValueError(
                f"At least on command line argument for the polygon mask "\
                f"directories leads to a file, rather than a directory. "\
                f"Instance passed: "\
                f"{next((p for p in mask_dirs if p.is_file()))}"
            )
        
        logger.debug("Getting polygon mask filepaths")
        mask_filepaths = list(
            set(i for j in mask_dirs for i in j.glob("*.gpkg"))
        )
        logger.debug("Group prediction polygon files by file name")
        temp = sorted(set(fp.name for fp in mask_filepaths))
        mask_filepaths = [
            sorted(
                [fp for fp in mask_filepaths if fp.name == fn],
                key = lambda d: mask_dirs.index(d.parent)
            )
            for fn in temp
        ]

        logger.debug("Iterate through prediction masks for each file group")
        mask_filepaths_iter = progressbar.progressbar(
            mask_filepaths,
            widgets = _WIDGETS,
            prefix = "Combining groups of mask predictions:"
        )
        mask_groups_fp: list[Path]
        for mask_groups_fp in mask_filepaths_iter:
            # load predictions
            mask_preds = [read_file(mask_fp) for mask_fp in mask_groups_fp]
            # concatenate predictions
            mask_preds: GeoDataFrame = concat(
                mask_preds, axis = 0, ignore_index = True
            )

            # Normalize geometries
            mask_preds["geometry"] = normalize_geometries(mask_preds.geometry)

            # Normalize groupid
            mask_preds["key"] =\
                mask_preds.png_filename + mask_preds.groupid.astype("string")

            key_group = mask_preds[["key"]]\
                .drop_duplicates(ignore_index = True)
            key_group["groupid"] = [*range(len(key_group))]
            
            mask_preds = merge_dataframes(
                mask_preds[[c for c in mask_preds.columns if c != "groupid"]],
                key_group,
                on = "key"
            )

            # Select columns
            mask_preds = mask_preds[config["base_columns"]]
            
            # save processed predictions out
            mask_preds.to_file(save_dir.joinpath(mask_groups_fp[0].name))

    except Exception as e:
        logger.error(e, exc_info = True)
        raise
