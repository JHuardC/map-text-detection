"""
Reads downloaded EDINA tiff files, clips and saves them to pngs.
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

if __name__ == "__main__":
    # Imports
    from logging import getLogger, StreamHandler, FileHandler, Formatter
    from datetime import datetime
    from json import load as load_json
    from edina import EDINATiffPNGConverter
    from project_utils import parse_path, build_argument_parser

    parser = build_argument_parser(filename = FILENAME, docstr = __doc__)
    parser.add_argument(
        "tiff_dir",
        action = "store",
        type = str,
        metavar = "path/to/tiff/dir",
        help =\
            "Required. Path to directory containing Edina downloaded tiff "\
            "files and their associated metadata files (.tfw). Can provide "\
            "relative or absolute paths; relative paths will be set against "\
            "path variable specified in config."
    )
    parser.add_argument(
        "pngs_dir",
        action = "store",
        type = str,
        metavar = "path/to/png/dir",
        help =\
            "Required. Specify directory to save clipped png files out to. "\
            "Can provide a relative or absolute path; relative paths will be "\
            "set relative to the path variable specified in config."
    )
    parser.add_argument(
        "-m", "--meta-dir",
        action = "store",
        type = str,
        metavar = "path/to/save/meta.ext",
        default = None,
        dest = "meta",
        help =\
            "Optional. Specify directory to save metadata file out to. The "\
            "metadata is a GeoDataFrame of control points used for "\
            "coordinate transforms. Format of saved output will be "\
            "the extension of the filename passed. Can provide a relative or "\
            "absolute path; relative paths will be set against the path "\
            "variable specified in the config. If no argument is provided "\
            "the metadata will be will be saved as a geopckage -- "\
            "control-points.gpkg -- in the same directory the PNGs were "\
            "saved out to."
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
    
        logger.debug("Parsing read in and save out directory paths")
        tiff_dir = parse_path(cla_args.tiff_dir, config["relative_path"])
        if tiff_dir.is_file():
            raise ValueError(
                f"Path to tiff dir is a file, rather than a directory. "\
                f"Value passed in command line: {cla_args.tiff_dir}"
            )
        png_dir = parse_path(cla_args.pngs_dir, config["relative_path"])
        if png_dir.is_file():
            raise ValueError(
                f"Path to png dir is a file, rather than a directory. "\
                f"Value passed in command line: {cla_args.pngs_dir}"
            )
        meta_fp = (
            parse_path(cla_args.meta, config["relative_path"])
            if cla_args.meta is not None
            else png_dir.joinpath("control-points.gpkg")
        )

        # instance EDINATiffPNGConverter
        converter = EDINATiffPNGConverter()

    except Exception as e:
        logger.error(e, exc_info = True)
        raise

    try:
        converter_kwargs = {
            "tiff_paths": tiff_dir.glob("*.tif"),
            "png_dest": png_dir,
            "png_h": config["png_h"],
            "png_w": config["png_w"],
            "overlap": config["overlap"],
            "to_crs": config.get("crs"),
            "start_h": config["start_h"],
            "start_w": config["start_w"]
        }
    except KeyError as f:
        e = KeyError(f"Config missing kwarg: '{f.args[0]}'")
        logger.error(e, exc_info = True)
        raise e from f
    except Exception as e:
        logger.error(e, exc_info = True)
        raise

    try:
        logger.debug("Running tiff to png conversion")
        gdf = converter.convert_batch_tiff_to_pngs(**converter_kwargs)

        logger.debug("Saving GeoDataFrame out.")
        gdf.to_file(meta_fp)

        logger.info("Script complete.")

    except Exception as e:
        logger.error(e, exc_info = True)
        raise
