"""
Script to process the outputs from the ToponymExtractor model. Cleans
overlapping predictions that come from PNG overlaps.

ToponymExtractor sourced from:
https://github.com/SesamePaste233/ToponymExtractor/tree/main
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

# Funtions
def parse_path(path: str, relative_to_envar: str | None = None) -> Path:
    """
    Utility function used to derive a path variable in conjunction with
    paths stored as environment variables.

    Parameters
    ----------
    path: str.
        Required. Path string.

    ralative_to_envar: str or None. Default: None.
        Optional. Environment variable to use as the root that the
        `path` argument is considered relative to. If no argument is
        passed, then no environment variable will be called.
    
    Return
    ------
    Path.
    """
    # Convert path argument to a Path instance
    path: Path = Path(path)
    if path.is_absolute() or (relative_to_envar is None):
        # Directly return absolute path
        return path
    # Get relative to path component
    try:
        root = environ[relative_to_envar]
    except KeyError as _:
        msg =\
            "relative_to_envar argument not recognised as an environment "\
            f"variable. Argument passed: {relative_to_envar}."
        raise ValueError(msg) from None
    return Path(root).joinpath(path)


if __name__ == "__main__":
    # Imports
    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    from logging import getLogger, StreamHandler, FileHandler, Formatter
    from datetime import datetime
    from json import load as load_json
    from pandas import concat
    from geopandas import read_file, GeoDataFrame
    from outputs import ProcessToponymExtractorPredictions

    parser = ArgumentParser(
        description = __doc__, formatter_class = RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "tiff_dir",
        action = "store",
        type = str,
        metavar = "to/tiff/dir",
        help =\
            "Required. Path to directory containing Edina downloaded tiff "\
            "files and their associated metadata files (.tfw). Can provide "\
            "relative or absolute paths; relative paths will be set against "\
            "path variable specified in config."
    )
    parser.add_argument(
        "geopreds",
        action = "store",
        type = str,
        metavar = "to/predictions/dir",
        help =\
            "Required. Path to directory containing ToponymExtractor "\
            "predictions with polygon masks. Can provide relative or "\
            "absolute paths; relative paths will be set against path "\
            "variable specified in config."
    )
    parser.add_argument(
        "gcps",
        action = "store",
        type = str,
        metavar = "to/gcp/gpkg",
        help =\
            "Required. Path to geopackage (.gpkg) file containing the "\
            "control points used for georeferencing the images passed to the "\
            "ToponymExtractor model. Dataset must include fields: "\
            "\"tiff_filename\", \"png_filename\", \"pixel_x\", \"pixel_y\", "\
            "and \"geometry\". Can provide relative or absolute paths; "\
            "relative paths will be set against path variable specified in "\
            "config."
    )
    parser.add_argument(
        "save_preds_to",
        action = "store",
        type = str,
        metavar = "to/save/dir",
        help =\
            "Required. Specify directory to save non-suppressed prediction "\
            "GeoDataFrames out to. The GeoDataFrames are saved under "\
            "filenames corresponding to the TIFF filenames the text "\
            "instances belong within; these files will be saved as "\
            "geopackages (.gpkg). Can provide a relative or absolute path; "\
            "relative paths will be set relative to the path variable "\
            "specified in config."
    )
    parser.add_argument(
        "save_suppressed_to",
        action = "store",
        type = str,
        metavar = "to/suppressed/dir",
        help =\
            "Required. Specify directory to save suppressed predictions "\
            "GeoDataFrames out to. The GeoDataFrames are saved under "\
            "filenames corresponding to the TIFF filenames the text "\
            "instances belong within; these files will be saved as "\
            "geopackages (.gpkg). Can provide a relative or absolute path; "\
            "relative paths will be set relative to the path variable "\
            "specified in config."
    )
    parser.add_argument(
        "save_ambiguous_img_to",
        action = "store",
        type = str,
        metavar = "to/ambiguous/img/dir",
        help =\
            "Required. Specify directory to save ambiguous predictions' "\
            "images out to. The images are saved under the filename "\
            "format \"{tiff filename}-clique-{clique index}.png\"; where "\
            "{tiff filename} represents the corresponding TIFF filename the "\
            "image segment is sourced from, and {clique index} represents "\
            "the unique index for the image segment. Can provide a relative "\
            "or absolute path; relative paths will be set relative to the "\
            "path variable specified in config."
    )
    parser.add_argument(
        "-g", "--gcp",
        action = "store",
        type = str,
        metavar = "to/save/gcp.ext",
        default = None,
        dest = "ctrl_out",
        help =\
            "Optional. Specify path to save the GeoDataFrame of the control "\
            "points out to. The the control points are used for coordinate "\
            "transforms for the ambiguous predictions image snippets. "\
            "The format of saved output will be the extension of the "\
            "filename passed. Can provide a relative or absolute path; "\
            "relative paths will be set against the path variable specified "\
            "in the config. If no argument is provided the metadata will be "\
            "will be saved as a geopckage -- control-points.gpkg -- in the "\
            "same directory the ambiguous image snippets were saved out to."
    )
    parser.add_argument(
        "-m", "--clique-meta",
        action = "store",
        type = str,
        metavar = "to/save/cliques/dir",
        default = None,
        dest = "clique_dir",
        help =\
            "Optional. Specify the directory to save the ambiguous "\
            "predictions' geodata out to. The GeoDataFrames are saved under "\
            "the filename format \"{tiff filename}-cliques.gpkg\"; where "\
            "{tiff filename} represents the corresponding TIFF filename the "\
            "ambiguous predictions are sourced from. Can provide a relative "\
            "or absolute path; relative paths will be set against the path "\
            "variable specified in the config. If no argument is provided "\
            "the metadata will be will be saved to the same directory that "\
            "the ambiguous images were saved out to."
    )
    parser.add_argument(
        "-c", "--config",
        action = "store",
        type = str,
        dest = "config",
        metavar = "path/to/config/json",
        default = None,
        help =\
            "Optional. Specify path to config json, containing presets used "\
            "to split tiffs into pngs. Can provide either a relative or "\
            "absolute path; relative paths will be set relative to the "\
            "project root directory. If no argument is provided, will "\
            f"attempt to load config from 'config/{FILENAME}.json', " \
            "relative to project root folder."
    )
    parser.add_argument(
        "-s", "--stream-level",
        action = "store",
        choices = [10, 20, 30, 40, 50],
        default = 20,
        dest = "stream_level",
        help = \
            "Optional. Level for logging messages to be streamed out. "\
            "Default is 20 - info level and above."
    )
    parser.add_argument(
        "-f", "--file-logs",
        action = "store_true",
        dest = "file",
        help = \
            "Optional. Save logging messages to .log file. If flagged, logs "\
            f"will be saved out to 'logs/{FILENAME}_YYYYmmDDHHMMSS.log' "\
            "relative to project root folder. All logging messages will be "\
            "saved (from debug up)."
    )
    cla_args = parser.parse_args()

    logger = getLogger()
    logger.setLevel(10)
    # Format
    fmt = Formatter(
        "[%(asctime)s] - %(levelname)s - %(filename)s - Line %(lineno)d - "\
        "%(funcName)s: %(message)s"
    )
    # Stream to terminal
    f = StreamHandler()
    f.setLevel(cla_args.stream_level)
    f.setFormatter(fmt)
    logger.addHandler(f)
    # Optionally log to file
    if cla_args.file:
        f = PROJECT_DIR.joinpath(
            f"logs/{FILENAME}_{datetime.now().strftime("%Y%m%d%H%M%S")}.log"
        )
        f = FileHandler(f, mode = "w")
        f.setLevel(10)
        f.setFormatter(fmt)
        logger.addHandler(f)
    
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
        tiff_dir = parse_path(cla_args.tiff_dir, config["relative_path"])
        if not tiff_dir.exists():
            raise ValueError(
                f"Command line argument for the tiff directory does not "\
                f"lead to an existing path. Argument passed: "\
                f"{cla_args.tiff_dir}"
            )
        if tiff_dir.is_file():
            raise ValueError(
                f"Command line argument for the tiff directory leads to a "\
                f"file, rather than a directory. Value passed in command "\
                f"line: {cla_args.tiff_dir}"
            )
        preds_dir = parse_path(cla_args.geopreds, config["relative_path"])
        if not preds_dir.exists():
            raise ValueError(
                f"Command line argument for the predictions directory does "\
                f"not lead to an existing path. Argument passed: "\
                f"{cla_args.geopreds}"
            )
        if preds_dir.is_file():
            raise ValueError(
                f"Command line argument for the predictions directory leads "\
                f"to a file, rather than a directory. Value passed in "\
                f"command line: {cla_args.geopreds}"
            )
        gcp_fp = parse_path(cla_args.gcps, config["relative_path"])
        if not gcp_fp.exists():
            raise ValueError(
                f"Command line argument for the path to georeferencing "\
                f"control points file does not lead to an existing file "\
                f"path. Argument passed: {cla_args.gcps}"
            )
        if not gcp_fp.is_file():
            raise ValueError(
                f"Command line argument for the path to georeferencing "\
                f"control points file leads to a directory, rather than a "\
                f"path. Argument passed: {cla_args.gcps}"
            )
        preds_out = parse_path(cla_args.save_preds_to, config["relative_path"])
        if not preds_out.exists():
            raise ValueError(
                f"Command line argument for the directory to save the "\
                f"retained predictions out to does not lead to an existing "\
                f"path. Argument passed: {cla_args.save_preds_to}"
            )
        if preds_out.is_file():
            raise ValueError(
                f"Command line argument for the path to save the retained "\
                f"predictions out to leads to a file, rather than a "\
                f"directory. Value passed in command line: "\
                f"{cla_args.save_preds_to}"
            )
        suppressed_out =\
            parse_path(cla_args.save_suppressed_to, config["relative_path"])
        if not suppressed_out.exists():
            raise ValueError(
                f"Command line argument for the directory to save the "\
                f"suppressed predictions out to does not lead to an existing "\
                f"path. Argument passed: {cla_args.save_suppressed_to}"
            )
        if suppressed_out.is_file():
            raise ValueError(
                f"Command line argument for the path to save the suppressed "\
                f"predictions out to leads to a file, rather than a "\
                f"directory. Value passed in command line: "\
                f"{cla_args.save_suppressed_to}"
            )
        ambiguous_img_out =\
            parse_path(cla_args.save_ambiguous_img_to, config["relative_path"])
        if not ambiguous_img_out.exists():
            raise ValueError(
                f"Command line argument for the directory to save the image "\
                f"snippets for ambiguous predictions out to does not lead to "\
                f"an existing path. Argument passed: "\
                f"{cla_args.save_ambiguous_img_to}"
            )
        if ambiguous_img_out.is_file():
            raise ValueError(
                f"Command line argument for the path to save the image "\
                f"snippets for ambiguous predictions out to leads to a file, "\
                f"rather than a directory. Value passed in command line: "\
                f"{cla_args.save_ambiguous_img_to}"
            )
        gcp_out_fp = (
            parse_path(cla_args.ctrl_out, config["relative_path"])
            if cla_args.ctrl_out is not None
            else ambiguous_img_out.joinpath("control-points.gpkg")
        )
        cliques_out_dir = (
            parse_path(cla_args.clique_dir, config["relative_path"])
            if cla_args.clique_dir is not None
            else ambiguous_img_out
        )
        if not cliques_out_dir.exists():
            raise ValueError(
                f"Command line argument for the directory to save the "\
                f"ambiguous predictions out to does not lead to an existing "\
                f"path. Argument passed: {cla_args.clique_dir}"
            )
        if cliques_out_dir.is_file():
            raise ValueError(
                f"Command line argument for the path to save the ambiguous "\
                f"predictions out to leads to a file, rather than a "\
                f"directory. Value passed in command line: "\
                f"{cla_args.clique_dir}"
            )
        
        logger.debug("Loading georefencing control points file")
        control_points = read_file(gcp_fp)

        logger.debug("Initializing predictions post-processor")
        post_processor = ProcessToponymExtractorPredictions(
            ctrl_points = control_points,
            tiff_dir = tiff_dir,
            img_h = config["img_h"],
            img_w = config["img_w"]
        )

        logger.debug("Initialise ambiguous image control points GeoDataFrame")
        ambiguous_img_control_points = GeoDataFrame()

        tiff_files = progressbar.progressbar(
            [*tiff_dir.glob("*.tif")],
            widgets = _WIDGETS,
            prefix = "Post-processing predictions:"
        )
        for tiff_fp in tiff_files:

            logger.debug(f"Loading predictions for TIFF: {tiff_fp.stem}")
            predictions = read_file(preds_dir.joinpath(f"{tiff_fp.stem}.gpkg"))
            predictions["geometry"] = predictions.geometry.buffer(0)

            logger.debug("Post-processing the predictions.")
            outputs = post_processor.process_predictions(
                tiff_fn = tiff_fp.name, predictions = predictions
            )
            if len(outputs[0]):
                logger.debug("Save retained predictions out.")
                outputs[0].to_file(preds_out.joinpath(f"{tiff_fp.stem}.gpkg"))
            if len(outputs[1]):
                logger.debug("Save suppressed predictions out.")
                outputs[1].to_file(
                    suppressed_out.joinpath(f"{tiff_fp.stem}.gpkg")
                )
            if len(outputs[2]["image"]):
                logger.debug("Save ambiguous prediction snippets")
                for clique_idx, img in enumerate(outputs[2]["image"]):
                    img.save(ambiguous_img_out.joinpath(
                        f"{tiff_fp.stem}-clique-{clique_idx}.png"
                    ))
                logger.debug("Save ambiguous predictions geodata out")
                outputs[2]["word_groups"].to_file(cliques_out_dir.joinpath(
                    f"{tiff_fp.stem}-cliques.gpkg"
                ))
                logger.debug("Add control points")
                temp = outputs[2]["control_points"]
                temp["tiff_name"] = tiff_fp.name
                ambiguous_img_control_points =\
                    concat([ambiguous_img_control_points, temp], axis = 0)
        
        if len(ambiguous_img_control_points):
            logger.debug("Save ambiguous image control points out")
            ambiguous_img_control_points.to_file(gcp_out_fp)

    except Exception as e:
        logger.error(e, exc_info = True)
        raise
