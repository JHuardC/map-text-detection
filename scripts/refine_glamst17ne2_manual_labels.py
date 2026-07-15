"""
Manually labelled file is cleansed of records not verified. Cropped
copy of the TIFF image the labels were derived from is also saved out.

Code is locked to specific input: glamst17ne2-manually-labelled.gpkg
as there is only one manually labelled file in the project.

Script outputs:

- glam-st17ne-2.tiff: Cropped tiff file from which predictions were
derived, cropped to exclude the area where predictions were not verified

- glamst17ne2-manually-labelled.gpkg: Refined manually labelled
predictions. This file excludes unverified predictions, suppresses
overlapping predictions, and removes false instances of word detections.
"""
# Imports
from typing import Final
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from os import environ

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
    from project_utils import build_argument_parser, build_logger, parse_path
    from json import load as load_json
    from geopandas import read_file
    from rasterio import open as open_raster
    from rasterio.coords import BoundingBox
    from rasterio.mask import mask as mask_raster
    from shapely import box, difference

    parser = build_argument_parser(filename = FILENAME, docstr = str)
    cla_args = parser.parse_args()

    logger = build_logger(
        stream_level = cla_args.stream_level,
        write_to = PROJECT_DIR.joinpath("logs") if cla_args.file else None,
        filename = FILENAME
    )
    
    try:
        logger.debug("Loading config.")
        f = (
            _CONFIG_DIR
            if cla_args.config is None
            else parse_path(cla_args.config, "PROJECT_DIR")
        )
        with open(f, "r") as f:
            config = {**_PRESET, **load_json(f)}

        logger.debug("Loading glamst17ne2-manually-labelled.gpkg")
        gdf = read_file(parse_path(
            config["relative_path_to_labels"], config["relative_path"]
        ))

        logger.debug("Loading glam-st17ne-2.tif")
        tiff_fp = parse_path(
            config["relative_path_to_tiff"], config["relative_path"]
        )
        with open_raster(tiff_fp, "r") as src:
            tiff_crs = src.read_crs() # coordinate reference system
            tiff_bounds: BoundingBox = src.bounds
        
        logger.debug("Remove unverified records")
        unverified_gdf = gdf[~gdf.verified].copy()
        gdf = gdf[gdf.verified]

        logger.debug("Normalise unverified geometry types to Polygon type")
        unverified_gdf["geometry"] =\
            unverified_gdf.geometry.buffer(0).convex_hull

        logger.debug("Convert unverified records CRS to tif raster CRS")
        unverified_gdf = unverified_gdf.to_crs(tiff_crs)

        logger.debug("Derive bounding box for all unverified predictions")
        unverified_bounds = unverified_gdf.total_bounds

        logger.debug(
            "Construct bounding box of tiff region containing unverified "\
            "predictions"
        )
        unverified_tiff_bounds = box(
            xmin = tiff_bounds.left,
            ymin = tiff_bounds.bottom,
            xmax = tiff_bounds.right,
            ymax = unverified_bounds[3]
        )

        logger.debug(
            "Get negative box, covering tiff region not containing "\
            "unverified predictions."
        )
        negative_unverified_tiff_bounds =\
            difference(a = box(*tiff_bounds), b = unverified_tiff_bounds)
        
        logger.debug("Crop tiff image by negative box.")
        with open_raster(tiff_fp, "r") as src:
            out_tiff, out_transform = mask_raster(
                dataset = src,
                shapes = [negative_unverified_tiff_bounds, ],
                crop = True
            )
            out_meta = src.meta

        logger.debug("Save cropped tiff out.")
        out_fp = parse_path(
            config["relative_path_save_to"], config["relative_path"]
        )
        out_tiff_fp = out_fp.joinpath(tiff_fp.name)
        out_meta.update({
            "driver": "GTiff",
            "height": out_tiff.shape[1],
            "width": out_tiff.shape[2],
            "transform": out_transform,
            "compress": "lzw"
        })
        with open_raster(out_tiff_fp, "w", **out_meta) as dest:
            dest.write(out_tiff[0], 1)

        logger.debug("Remove overlapping predictions")
        gdf = gdf[
            (~gdf.status.str.contains("O")) | gdf.status.str.contains("N")
        ]

        logger.debug("Remove predictions that mistake map features for text")
        gdf = gdf[gdf.status != "DNE"]
        
        logger.debug("Save verified predictions out")
        gdf.to_file(out_fp.joinpath("glamst17ne2-manually-labelled.gpkg"))

    except Exception as e:
        logger.error(e, exc_info = True)
        raise

