"""
Retrieves points from Gb1900 Gazetteer within the bounds of the PNGs
provided and converts their geospatial coordinates to pixel coordinates.
"""
# Imports
from typing import Final
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from os import getenv, environ
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

# Funtions
def parse_path(path: str, relative_to_envar: str) -> Path:
    # Convert path argument to a Path instance
    path: Path = Path(path)
    if path.is_absolute():
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
    from pyogrio.errors import DataLayerError
    from geopandas import\
        read_file as geo_read_file, GeoDataFrame, points_from_xy, sjoin
    from pandas import read_csv as pandas_read_csv

    parser = ArgumentParser(
        description = __doc__, formatter_class = RawDescriptionHelpFormatter
    )
    # parser.add_argument(
    #     "pngs_dir",
    #     action = "store",
    #     type = str,
    #     metavar = "path/to/png/dir",
    #     help =\
    #         "Required. Path to directory containing clipped png files.  Can "\
    #         "provide relative or absolute paths; relative paths will be set "\
    #         "against path variable specified in the config."
    # )
    parser.add_argument(
        "meta_dir",
        action = "store",
        type = str,
        metavar = "path/to/meta.ext",
        help =\
            "Required. Path to metadata file containing the control points "\
            "used for georeferencing the PNGs found in the PNG directory "\
            "provided. Can provide a relative or absolute path; relative "\
            "paths will be set relative to the path variable specified in "\
            "the config."
    )
    parser.add_argument(
        "gb1900",
        action = "store",
        type = str,
        metavar = "path/to/gb1900.csv",
        help =\
            "Required. Path to Gb1900 gazetteer csv. Can provide a relative "
            "or absolute path; relative paths will be set relative to the "\
            "path variable specified in the config."
    )
    parser.add_argument(
        "-o", "--output",
        action = "store",
        type = str,
        metavar = "path/to/save/output.ext",
        default = None,
        dest = "save_out",
        help =\
            "Optional. Specify directory to save the relevant Gb1900 points "\
            "out to. The output is a GeoDataFrame of points. The format of "\
            "the saved output will be determined the extension of the "\
            "filename passed. Can provide a relative or absolute path; "\
            "relative paths will be set against the path variable specified "\
            "in the config. If no argument is provided the data will be "\
            "saved as a geopckage -- text-locations.gpkg -- in the same "\
            "directory the PNGs were fread from."
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
        "[%(asctime)s] - %(levelname)s - %(filename)s - Line %(lineno)d "\
        "- %(funcName)s: %(message)s"
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
    
        logger.debug("Parsing read in and save out directory paths")
        meta_fp = parse_path(cla_args.meta_dir, config["relative_path"])
        gb1900_fp = parse_path(cla_args.gb1900, config["relative_path"])
        out_fp = (
            parse_path(cla_args.save_out, config["relative_path"])
            if cla_args.save_out is not None
            else meta_fp.parent.joinpath("text-locations.gpkg")
        )

    except Exception as e:
        logger.error(e, exc_info = True)
        raise

    try:
        logger.info("Reading georeferencing data.")
        georef_df = geo_read_file(
            meta_fp,
            layer = (
                config.get("meta_geopackage_layer", meta_fp.stem)
                if meta_fp.suffix == ".gpkg"
                else None
            )
        )
    except DataLayerError as e:
        err = DataLayerError(
            f"Error loading metadata file {meta_fp.name}. {e.args[0]}.  "\
            "Please check file type is correct (geopackage) and check layer "\
            "name."
        )
        logger.error(err, exc_info = True)
        raise err from e
    except Exception as e:
        logger.error(e, exc_info = True)
        raise

    try:
        logger.info("Reading Gb1900 gazetteer data.")
        gb1900 = pandas_read_csv(gb1900_fp, **config["gb1900_read_csv_kwargs"])
        # convert to GeoDataFrame
        gb1900 = GeoDataFrame(
            gb1900,
            geometry = points_from_xy(
                gb1900[config["gb1900_x"]],
                gb1900[config["gb1900_y"]],
                crs = config["gb1900_crs"]
            )
        )

        logger.debug(
            "Building bounding boxes from PNG control points to search for "\
            "gb1900 toponymns within."
        )
        georef_df = georef_df[["tiff_filename", "png_filename", "geometry"]]
        georef_df = georef_df\
            .dissolve(by = ["tiff_filename", "png_filename"], as_index = False)
        georef_df["geometry"] = georef_df.geometry.convex_hull

        logger.info("Merging GB1900 toponyms with png bounds")
        gb1900 = sjoin(gb1900, georef_df, "inner")\
            .drop(columns = ["index_right"])\
            .reset_index(drop = True)
        
        logger.debug("Saving selected gb1900 map text points out")
        gb1900.to_file(out_fp)

    except Exception as e:
        logger.error(e, exc_info = True)
        raise
