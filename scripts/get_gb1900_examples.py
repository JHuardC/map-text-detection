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
from pandas import DataFrame
from geopandas import GeoDataFrame
from edina import get_transformer_from_geodataframe

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
def get_pixel_xy(
    coords_df: GeoDataFrame, control_points_df: GeoDataFrame
) -> DataFrame:
    """
    Get pixel values for each coordinate in `coords_df`, using
    `control_points_df`.

    Parameters
    ----------
    coords_df: GeoDataFrame.
        Required. GeoDataFrame containing Point coordinates we seek to
        obtain pixel row and column indexes for. The GeoDataFrame
        requires the fields:
            - png_filename: str. Which PNG the geospatial coordinate
            belongs to, used to reference the correct control points.
            - geometry: Point. Geospatial coordinate to convert to
            pixel row and column values.

    control_points_df: GeoDataFrame.
        Required. GeoDataFrame containing Ground Control Point
        coordinates used for georefencing PNG files. The GeoDataFrame
        requires the fields:
            - png_filename: str. Which PNG the Ground Control Points
            belong to.
            - pixel_x: int. Column index associated with the records
            control point.
            - pixel_y: int. Row index associated with the records
            control point.
            - geometry: Point. Ground Control Geospatial coordinate.
    """
    check = set(coords_df["png_filename"])\
        .difference(control_points_df["png_filename"])
    if len(check):
        raise ValueError(
            f"Missing control points for {len(check)} records. Compare PNG "\
            "filenames to check."
        )
    # Construct progressbar
    progress = progressbar\
        .ProgressBar(0, len(coords_df), _WIDGETS, prefix = "Get pixel idxs:")
    
    xy = []
    # cycle through coords
    progress.start()
    try:
        for row in coords_df.itertuples(index = False):
            # Constrain control_points_df to specific png and get
            # Geo-Transformer
            gcptrans = get_transformer_from_geodataframe(control_points_df.loc[
                control_points_df["png_filename"] == row.png_filename
            ])
            xy.append(gcptrans.rowcol(row.geometry.x, row.geometry.y))
            progress.increment()
    except Exception as e:
        progress.finish(dirty = True)
        raise
    progress.finish()
    # return row-column values as a DataFrame
    return DataFrame(
        data = xy, index = coords_df.index, columns = ["pixel_x", "pixel_y"]
    )

if __name__ == "__main__":
    # Imports
    from logging import getLogger, StreamHandler, FileHandler, Formatter
    from datetime import datetime
    from json import load as load_json
    from pyogrio.errors import DataLayerError
    from geopandas import read_file as geo_read_file, points_from_xy, sjoin
    from pandas import read_csv as pandas_read_csv
    from project_utils import parse_path
    from project_utils import parse_path, build_argument_parser

    parser = build_argument_parser(filename = FILENAME, docstr = __doc__)
    parser.add_argument(
        "meta_dir",
        action = "store",
        type = str,
        metavar = "path/to/gcp.ext",
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
            "the saved output will be determined by the extension passed in "\
            "the filename. Can provide a relative or absolute path; "\
            "relative paths will be set against the path variable specified "\
            "in the config. If no argument is provided the data will be "\
            "saved as a geopckage -- text-locations.gpkg -- in the same "\
            "directory the control points metadata file was read from."
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
        png_bounds = georef_df[["tiff_filename", "png_filename", "geometry"]]\
            .dissolve(by = ["tiff_filename","png_filename"], as_index = False)
        png_bounds["geometry"] = png_bounds.geometry.convex_hull

        logger.info("Merging GB1900 toponyms with png bounds")
        gb1900 = sjoin(gb1900, png_bounds, "inner")\
            .drop(columns = ["index_right"])\
            .reset_index(drop = True)
        del png_bounds
        
        logger.info("Get pixel values for each for each gazetteer point.")
        gb1900[["pixel_x", "pixel_y"]] = get_pixel_xy(gb1900, georef_df)
        
        logger.debug("Saving selected gb1900 map text points out.")
        gb1900.to_file(out_fp)

    except Exception as e:
        logger.error(e, exc_info = True)
        raise
