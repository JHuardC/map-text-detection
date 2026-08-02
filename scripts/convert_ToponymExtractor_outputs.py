"""
Script to parse the outputs from the ToponymExtractor model.

Converts the outputs to geodata polygons. Polygon masks are grouped by
the TIFF files the texts are contained within and these groups are
saved out to seperate geopackage (.gpkg) files, with filenames
corresponding to the TIFF file names.

ToponymExtractor sourced from:
https://github.com/SesamePaste233/ToponymExtractor/tree/main
"""
# Imports
from typing import Final
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from os import environ
import progressbar
from pickle import load as load_pickle

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

# Funtions

if __name__ == "__main__":
    # Imports
    from json import load as load_json
    from geopandas import read_file
    from outputs import\
        convert_ToponymExtractor_outputs_to_gdf, read_pickle_queue
    from project_utils import parse_path
    from project_utils import parse_path, build_argument_parser, build_logger

    parser = build_argument_parser(
        filename = FILENAME,
        description =\
            "Script to parse the outputs from the ToponymExtractor model."
    )
    parser.add_argument(
        "predictions",
        action = "store",
        type = str,
        metavar = "to/preds/pickle",
        help =\
            "Required. Path to pickle file containing ToponymExtractor "\
            "predictions. Can provide relative or absolute paths; relative "\
            "paths will be set against path variable specified in config."
    )
    parser.add_argument(
        "gcps",
        action = "store",
        type = str,
        metavar = "to/gcp/ext",
        help =\
            "Required. Path to geodata file containing the control points "\
            "used for georeferencing the images passed to the "\
            "ToponymExtractor model. Dataset must include fields: "\
            "\"tiff_filename\", \"png_filename\", \"pixel_x\", \"pixel_y\", "\
            "and \"geometry\". Can provide relative or absolute paths; "\
            "relative paths will be set against path variable specified in "\
            "config."
    )
    parser.add_argument(
        "save_geo_to",
        action = "store",
        type = str,
        metavar = "to/save/dir",
        help =\
            "Required. Specify directory to save converted GeoDataFrames out "\
            "to. The GeoDataFrames are saved as geopackages (.gpkg) under "\
            "file names corresponding to the TIFF filenames the text " \
            "instances belong within. Can provide a relative or absolute "\
            "path; relative paths will be set relative to the path variable "\
            "specified in config."
    )
    parser.add_argument(
        "save_err_to",
        action = "store",
        type = str,
        metavar = "to/save/errors/csv",
        help =\
            "Required. Specify CSV file path to save error information out "\
            "to. Can provide a relative or absolute path; relative paths "\
            "will be set relative to the path variable specified in config."
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
        pred_fp = parse_path(cla_args.predictions, config["relative_path"])
        if pred_fp.suffix != ".pkl":
            raise ValueError(
                f"Command line argument for predictions filepath does not " \
                f"lead to a pickle (.pkl) type file. Argument passed: "\
                f"{cla_args.predictions}"
            )
        gcp_fp = parse_path(cla_args.gcps, config["relative_path"])
        save_geo_fp = parse_path(cla_args.save_geo_to, config["relative_path"])
        if save_geo_fp.is_file():
            raise ValueError(
                f"Path to save geodata to is a file, rather than a "\
                f"directory. Value passed in command line: "\
                f"{cla_args.save_geo_to}"
            )
        save_err_fp = parse_path(cla_args.save_err_to, config["relative_path"])
        if save_err_fp.suffix != ".csv":
            raise ValueError(
                f"Command line argument for error details filepath does not "\
                f"lead to a csv (.csv) type file. Argument passed: "\
                f"{cla_args.save_err_to}"
            )

        
        logger.debug("Loading files")
        predictions = read_pickle_queue(pred_fp)
        ctrl_points = read_file(gcp_fp)

        logger.debug("Convert model predictions.")
        geodata, errors = convert_ToponymExtractor_outputs_to_gdf(
            out = predictions,
            control_points = ctrl_points,
            png_h = config["png_h"],
            png_w = config["png_w"]
        )
        logger.debug("Save error data out.")
        errors.to_csv(save_err_fp)

        logger.debug("Save polygon masks for each tiff image.")
        for tiff_fn in ctrl_points["tiff_filename"].unique():
            selection = set(ctrl_points.loc[
                ctrl_points["tiff_filename"] == tiff_fn, "png_filename"
            ])
            gdf = geodata[geodata["png_filename"].isin(selection)]
            gdf.to_file(save_geo_fp.joinpath(tiff_fn.replace(".tif", ".gpkg")))

    except Exception as e:
        logger.error(e, exc_info = True)
        raise
