"""
Script to process the ambiguous predictions from the ToponymExtractor
model.

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
    from logging import getLogger, StreamHandler, FileHandler, Formatter
    from datetime import datetime
    from json import load as load_json
    from functools import partial
    from pandas import concat
    from shapely import box
    from geopandas import read_file, GeoDataFrame, sjoin as spatial_join
    from outputs import\
        read_pickle_queue,\
        convert_ToponymExtractor_outputs_to_gdf,\
        georeference_geometries
    from edina import get_transformer_from_geodataframe
    from project_utils import parse_path
    from project_utils import parse_path, build_argument_parser

    parser = build_argument_parser(filename = FILENAME, docstr = __doc__)
    parser.add_argument(
        "meta_img",
        action = "store",
        type = str,
        metavar = "to/ambiguous/img/meta",
        help =\
            "Required. Path to directory containing ambiguous image metadata "\
            "files. The metadata files are saved under the filename format "\
            "\"{tiff filename}-ambiguous-meta.json\" and "\
            "\"{tiff filename}-cliques.gpkg\"; where {tiff filename} "\
            "represents the corresponding TIFF filename the associated image "\
            "segments are sourced from. Can provide relative or absolute "\
            "paths; relative paths will be set against path variable "\
            "specified in config."
    )
    parser.add_argument(
        "preds",
        action = "store",
        type = str,
        metavar = "to/preds/pickle",
        help =\
            "Required. Path to pickle file containing predictions from "\
            "ToponymExtractor for the ambiguous images. Can provide relative "\
            "or absolute paths; relative paths will be set against path "\
            "variable specified in config."
    )
    parser.add_argument(
        "gcps",
        action = "store",
        type = str,
        metavar = "to/gcp/gpkg",
        help =\
            "Required. Path to geopackage (.gpkg) file containing the "\
            "control points used for georeferencing the ambiguous images "\
            "snippets passed to the ToponymExtractor model. Dataset must "\
            "include fields: \"tiff_name\", \"clique_idx\", \"pixel_x\", "\
            "\"pixel_y\", and \"geometry\". Can provide relative or absolute "\
            "paths; relative paths will be set against path variable "\
            "specified in config."
    )
    parser.add_argument(
        "save_geopreds_to",
        action = "store",
        type = str,
        metavar = "to/save/preds/dir",
        help =\
            "Required. Specify directory to save converted prediction "\
            "GeoDataFrames out to. The GeoDataFrames are saved under the "\
            "filename format \"{tiff filename}-ambiguous.gpkg\" ; where "\
            "{tiff filename} represents the corresponding to the TIFF "\
            "filenames the text instances belong within. Can provide a "\
            "relative or absolute path; relative paths will be set relative "\
            "to the path variable specified in config."
    )
    parser.add_argument(
        "save_err_to",
        action = "store",
        type = str,
        metavar = "to/save/error/csv",
        help =\
            "Required. Specify CSV file path to save errored prediction "\
            "information out to. Can provide a relative or absolute path; "\
            "relative paths will be set relative to the path variable "\
            "specified in config."
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
        meta_dir = parse_path(cla_args.meta_img, config["relative_path"])
        if not meta_dir.exists():
            raise ValueError(
                f"Command line argument for the ambiguous metadata directory "\
                f"does not lead to an existing path. Argument passed: "\
                f"{cla_args.meta_dir}"
            )
        if meta_dir.is_file():
            raise ValueError(
                f"Command line argument for the ambiguous metadata directory "\
                f"leads to a file, rather than a directory. Value passed in "\
                f"command line: {cla_args.meta_dir}"
            )
        pred_fp = parse_path(cla_args.preds, config["relative_path"])
        if not pred_fp.exists():
            raise ValueError(
                f"Command line argument for the ambiguous predictions "\
                f"file does not lead to an existing path. Argument "\
                f"passed: {cla_args.preds}"
            )
        if not pred_fp.is_file():
            raise ValueError(
                f"Command line argument for the ambiguous predictions "\
                f"file leads to a directory, rather than a file. Value "\
                f"passed in command line: {cla_args.preds}"
            )
        if pred_fp.suffix != ".pkl":
            raise ValueError(
                f"Command line argument for predictions filepath does not " \
                f"lead to a pickle (.pkl) type file. Argument passed: "\
                f"{cla_args.predictions}"
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
        save_preds_dir =\
            parse_path(cla_args.save_geopreds_to, config["relative_path"])
        if not save_preds_dir.exists():
            raise ValueError(
                f"Path to save ambiguous predictions to does not lead to an "\
                f"existing path. Argument passed: {cla_args.preds}"
            )
        if save_preds_dir.is_file():
            raise ValueError(
                f"Path to save ambiguous predictions to is a file, rather "\
                f"than a directory. Value passed in command line: "\
                f"{cla_args.save_geo_to}"
            )
        save_err_fp = parse_path(cla_args.save_err_to, config["relative_path"])
        if save_err_fp.suffix != ".csv":
            raise ValueError(
                f"Command line argument for error details filepath does not "\
                f"lead to a csv (.csv) type file. Argument passed: "\
                f"{cla_args.save_err_to}"
            )
        
        logger.debug("Loading georefencing control points file")
        gcp = read_file(gcp_fp)
        logger.debug("Loading ambiguous predictions files")
        predictions = read_pickle_queue(pred_fp)
        
        logger.debug("Convert predictions to geodata format")
        predictions, errors = convert_ToponymExtractor_outputs_to_gdf(
            out = predictions, png_h = config["img_h"], png_w = config["img_w"]
        )
        logger.debug("Save error data out.")
        errors.to_csv(save_err_fp)

        logger.debug("Iterate through predictions for each ambiguous image")
        ambiguous_meta_filename_iter = [
            f for f in meta_dir.glob("*-ambiguous-meta.json")
            if f.name not in getattr(errors, "png_filename", [])
        ]
        ambiguous_meta_filename_iter = progressbar.progressbar(
            ambiguous_meta_filename_iter,
            widgets = _WIDGETS,
            prefix = "Post-processing ambiguous predictions:"
        )
        for ambiguous_meta_fn in ambiguous_meta_filename_iter:
            logger.debug(f"Load metadata json: {ambiguous_meta_fn.name}")
            with open(ambiguous_meta_fn, "r") as f:
                metadata: dict = load_json(f)
            
            logger.debug("Load original clique predictions")
            optional_preds = ambiguous_meta_fn.name
            original_preds = read_file(meta_dir.joinpath(
                optional_preds.replace("ambiguous-meta.json", "cliques.gpkg")
            ))
            original_preds = original_preds\
                .rename(columns = {"clique_idx": "clique_id"})
            
            processed_preds = []
            logger.debug("Iterating through image metadata items")
            for img_fn, img_meta in metadata.items():
                # get image predictions
                img_preds =\
                    predictions[(predictions.png_filename == img_fn)].copy()

                if len(img_preds):
                    # create clique bounding boxes for eath image snippet
                    boxes = []
                    box_iter = zip(
                        img_meta["cliques"],
                        img_meta["row_pos"],
                        img_meta["col_pos"],
                        img_meta["shapes"],
                        strict = True
                    )
                    for clique, row, col, (h, w) in box_iter:
                        boxes.append({
                            "clique_id": clique,
                            "minx": col,
                            "miny": row,
                            "maxx": col + w - 1,
                            "maxy": row + h - 1,
                            "geometry": box(col, row, col + w - 1, row + h - 1)
                        })
                    boxes = GeoDataFrame(boxes)

                    # Join bounding box cliques to predictions
                    img_preds = spatial_join(
                        img_preds,
                        boxes[["clique_id", "geometry"]],
                        how = "inner",
                        predicate = "intersects"
                    )
                    
                    img_preds["intersect_pc"] = boxes\
                        .loc[img_preds.index_right.to_list(), "geometry"]\
                        .intersection(img_preds.geometry, align = False)\
                        .area\
                        .to_list()
                    img_preds["intersect_pc"] /= img_preds.geometry.area
                    
                    # clip image predictions geometries to the bounds of
                    # the PNG snippet
                    selection = boxes\
                        .loc[img_preds.index_right.to_list(), "geometry"]
                    img_preds["geometry"] = img_preds\
                        .geometry.intersection(selection, align = False)
                    
                    # adjust prediction polygons by their associated clique
                    # box
                    for tup in boxes.itertuples(index = False):
                        selection = (img_preds.clique_id == tup.clique_id)
                        img_preds.loc[selection, "geometry"] = img_preds\
                            .loc[selection, "geometry"]\
                            .transform(lambda x: x - [tup.minx, tup.miny])
                        
                    # Create georeferenced polygons
                    for clique in img_preds.clique_id.unique():
                        gcp_trans = gcp.loc[(
                            (gcp.tiff_name == (img_meta["tiff_stem"] + ".tif"))
                            & (gcp.clique_idx == clique)
                        )]
                        gcp_trans =\
                            get_transformer_from_geodataframe(gcp_trans)
                        trans = partial(georeference_geometries, gcp_trans)
                        
                        # Convert pixel location coordinates to latitude/
                        # longitude
                        selection = (img_preds.clique_id == clique)
                        img_preds.loc[selection, "geometry"] = img_preds\
                            .loc[selection, "geometry"]\
                            .transform(trans, include_z = 0)
                        # Documentation recommends calling close on
                        # GCPTransformer after calling transforms
                        gcp_trans.close()
                    
                    img_preds = img_preds.set_crs(gcp.crs)
                    img_preds.drop(columns = ["index_right"], inplace = True)

                    # keep only predictions from image snippets that intersect
                    # with original predictions they are to replace
                    pairs = spatial_join(
                        img_preds,
                        original_preds,
                        how = "inner",
                        predicate = "intersects",
                        on_attribute = "clique_id"
                    )
                    img_preds = img_preds\
                        .loc[pairs.index.unique().sort_values()]
                    
                    # add img_preds to processed list
                    processed_preds.append(img_preds)
            
            # combine processed predictions
            processed_preds = concat(processed_preds, ignore_index = True)
            # save processed predictions out
            processed_preds.to_file(save_preds_dir.joinpath(
                img_meta["tiff_stem"] + ".gpkg"
            ))

    except Exception as e:
        logger.error(e, exc_info = True)
        raise
