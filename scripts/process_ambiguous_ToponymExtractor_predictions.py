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
from collections.abc import Generator
from typing import Final
from pathlib import Path
from logging import getLogger
from dotenv import find_dotenv, load_dotenv
from os import environ
from geopandas import GeoDataFrame
from pandas import concat
import progressbar

progressbar.streams.flush()
progressbar.streams.wrap_stderr()

_logger = getLogger(__name__)

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

def extract_multibox_entities(
    gdf: GeoDataFrame, entity_col: str, box_id_col: str
) -> tuple[GeoDataFrame, GeoDataFrame]:
    """
    Extracts records of entities spread across multiple image snippet
    boxes.

    Parameters
    ----------
    gdf: GeoDataFrame.
        Required. GeoDataFrame containing records to check.
    
    entity_col: str.
        Required. Column name in `gdf` that contains the entity values
        to check against.
    
    box_id_col: str.
        Required. Column name in `gdf` that references the image snippet
        box identities associated with each record. If more than one
        box ID value is associated with an entity, then the records will
        be extracted from `gdf`.
    
    Returns
    -------
    2-element tuple.
        - GeoDataFrame: `gdf` GeoDataFrame with the records associated
        with more than one image snippet box removed.
        - GeoDataFrame: Containing the records associated with more than
        one image snippet box.
    """
    # get boolean Series representing records whose entities are associated
    # with more than one image snippet box
    multibox_entities = gdf\
        .groupby(by = entity_col, as_index = False)\
        .agg(box_count = (box_id_col, "nunique"))
    multibox_entities = multibox_entities[multibox_entities.box_count > 1]
    multibox_entities = set(multibox_entities[entity_col])
    multibox_entities = gdf[entity_col].isin(multibox_entities)

    # extract multibox records
    multibox = gdf[multibox_entities].copy()

    # remove multibox records from gdf
    gdf = gdf[~multibox_entities]

    return gdf, multibox


def update_supressed_gdf(
    suppressed: GeoDataFrame, new_records: GeoDataFrame, reason: str
) -> GeoDataFrame:
    """
    Updates GeoDataFrame logging suppressed records with `new_records`
    and annotates with reason for suppression.

    Parameters
    ----------
    suppressed: GeoDataFrame.
        Required. Log of records suppressed.
    
    new_records: GeoDataFrame.
        Required. Records to add to `suppressed`.
    
    reason: str.
        Required. Reason `new_records` are being sppressed.
    
    Returns
    -------
    GeoDataFrame. Suppressed records with `new_records` added.
    """
    # add reason to new_records data
    if len(new_records):
        new_records["suppressed_reason"] = reason
        suppressed = concat(
            [new_records, suppressed], axis = 0, ignore_index = True
        )
    return suppressed


def _get_minword_clique(group: GeoDataFrame) -> GeoDataFrame:
    """
    Gets words belonging to the same clique as the first word in group.

    First word is determined by wordid and returned cliques include the
    first word.

    Parameters
    ----------
    group: GeoDataFrame.
        Required. GeoDataFrame containing word predictions grouped into
        a toponym by ToponymExtractor. Requires fields "wordid" and 
        "clique_id".
    
    Returns
    -------
    GeoDataFrame. Contains the first word, by smallest "wordid", and all
    other records that have the same "clique_id" as this record.
    Contains same fields as `group`.
    """
    clique_id = group.at[group.wordid.idxmin(), "clique_id"]
    group = group[(group.clique_id == clique_id)]
    return group


def _yield_group_minword_clique(
    gdf: GeoDataFrame
) -> Generator[GeoDataFrame, None, None]:
    """
    Yields words belonging to the same clique as the first word for each
    group in the GeoDataFrame passed.

    First word is determined by wordid and returned cliques include the
    first word.

    Parameters
    ----------
    gdf: GeoDataFrame.
        Required. GeoDataFrame containing word predictions from one or
        more toponym groups, output by ToponymExtractor. Requires fields
        "wordid", "clique_id", and "pred_id".
    
    Yields
    ------
    GeoDataFrame. Contains the first word, by smallest "wordid", and all
    other records that have the same "clique_id" as this record, for
    each group in `gdf`. Contains same fields as `gdf`.
    """
    for groupid in gdf.groupid.unique():
        yield _get_minword_clique(gdf[(gdf.groupid == groupid)].copy())


def _get_minword_clique_groups(gdf: GeoDataFrame) -> GeoDataFrame:
    """
    Gets words belonging to the same clique as the first word for each
    group in the GeoDataFrame passed.

    First word is determined by wordid and returned cliques include the
    first word.

    Parameters
    ----------
    gdf: GeoDataFrame.
        Required. GeoDataFrame containing word predictions from one or
        more toponym groups, output by ToponymExtractor. Requires fields
        "wordid", "clique_id", and "pred_id".
    
    Returns
    -------
    GeoDataFrame. Each group contains the same first word as the groups
    in `gdf`, with the following words in the same group and within the
    same clique.
    """
    return concat(
        [*_yield_group_minword_clique(gdf)],
        axis = 0,
        ignore_index = True
    )


def _update_groupids(gdf: GeoDataFrame, start: int):
    """
    Reassigns groupid values in `gdf` given, from start value onwards.

    New groupid values are provided sequenctially from `start` value
    onwards.
    """
    replace_mapping =\
        {v: start + i for i, v in enumerate(gdf.groupid.unique())}
    gdf["groupid"] = gdf.groupid.replace(replace_mapping)
    return gdf


if __name__ == "__main__":
    # Imports
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
    from project_utils import parse_path, build_argument_parser, build_logger

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
    parser.add_argument(
        "--supto",
        action = "store",
        type = str,
        metavar = "to/save/suppressed/gpkg",
        default = None,
        dest = "suppressed",
        help =\
            "Optional. Specify directory to save suppressed predictions file "\
            "out to. The suppressed predictions are a GeoDataFrame of "\
            "polygon masks. The format of saved output will be the extension "\
            "of the filename passed. Can provide a relative or "\
            "absolute path; relative paths will be set against the path "\
            "variable specified in the config. If no argument is provided "\
            "the suppressed predictions data will be will be saved as a "\
            "geopckage -- suppressed-predictions.gpkg -- in the same "\
            "directory the geopreds were saved out to."
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
        suppressed_fp = (
            parse_path(cla_args.suppressed, config["relative_path"])
            if cla_args.suppressed is not None
            else save_preds_dir.joinpath("suppressed-predictions.gpkg")
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

        logger.debug("Initialize GeoDataFrame to log suppressed records.")
        suppressed = GeoDataFrame()

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
            original_preds = ambiguous_meta_fn.name
            original_preds = read_file(meta_dir.joinpath(
                original_preds.replace("ambiguous-meta.json", "cliques.gpkg")
            ))
            original_preds = original_preds\
                .rename(columns = {"clique_idx": "clique_id"})
            original_preds["geometry"] = original_preds\
                .geometry.buffer(0).convex_hull.centroid
            
            processed_preds = []
            logger.debug("Iterating through image metadata items")
            for img_fn, img_meta in metadata.items():
                # get image predictions
                img_preds: GeoDataFrame =\
                    predictions[(predictions.png_filename == img_fn)].copy()
                # Get root of png file name
                img_preds["tiff_stem"] = img_meta["tiff_stem"]
                # Create unique identifier field for each prediction
                img_preds["pred_id"] = [*range(len(img_preds))]

                if len(img_preds):
                    # create clique bounding boxes for each image snippet
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

                    # Join bounding box of clique image snippets to predictions
                    img_preds = spatial_join(
                        img_preds,
                        boxes[["clique_id", "geometry"]],
                        how = "inner",
                        predicate = "intersects"
                    )
                    
                    # Drop any word predictions belonging to more than
                    # one bounding box.
                    img_preds, multibox_flag = extract_multibox_entities(
                        gdf = img_preds,
                        entity_col = "pred_id",
                        box_id_col = "clique_id"
                    )

                    if len(multibox_flag):
                        logger.info(
                            f"{len(multibox_flag)} predictions spread across "\
                            f"more than one image snippet."
                        )
                    
                    # Add multibox suppressed records to suppressed logs
                    suppressed = update_supressed_gdf(
                        suppressed,
                        multibox_flag,
                        "Prediction spread across more than one image snippet"
                    )
                    
                    # Get percent of polygons belonging within their respective
                    # image snippet boxes
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
                    
                    # Break up groups spread across multiple image snippet
                    # boxes
                    img_preds, multibox_flag = extract_multibox_entities(
                        gdf = img_preds,
                        entity_col = "groupid",
                        box_id_col = "clique_id"
                    )
                    
                    logger.info(
                        f"{len(multibox_flag)} records contained in groups "\
                        f"spread across more than one image snippet."
                    )
                    if len(multibox_flag):
                        # Pass back records to img_preds - group-clique by
                        # group-clique
                        temp = _get_minword_clique_groups(multibox_flag)
                        # remove records from multibox_flag
                        multibox_flag = multibox_flag[
                            ~multibox_flag.pred_id.isin(set(temp.pred_id))
                        ]
                        # Add records back to img_preds
                        img_preds =\
                            concat([img_preds, multibox_flag], axis = 0)

                    while len(multibox_flag):
                        temp = _get_minword_clique_groups(multibox_flag)
                        # remove records from multibox_flag
                        multibox_flag = multibox_flag[
                            ~multibox_flag.pred_id.isin(set(temp.pred_id))
                        ]
                        # Update groupids
                        temp =\
                            _update_groupids(temp, img_preds.groupid.max() + 1)
                        # Add records back to img_preds
                        img_preds = concat([img_preds,multibox_flag], axis = 0)
                    
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
                            .transform(trans, include_z = False)
                        # Documentation recommends calling close on
                        # GCPTransformer after calling transforms
                        gcp_trans.close()
                    
                    img_preds = img_preds.set_crs(gcp.crs)
                    img_preds.drop(columns = ["index_right"], inplace = True)
                    
                    # Reset index
                    img_preds.reset_index(drop = True, inplace = True)

                    # keep only predictions from image snippets that intersect
                    # with the centroids of the original predictions they are
                    # to replace
                    pairs = spatial_join(
                        img_preds,
                        original_preds,
                        how = "inner",
                        predicate = "contains",
                        on_attribute = "clique_id"
                    )
                    relevant_preds_idx = pairs.index.unique().sort_values()

                    if (temp := len(img_preds) - len(relevant_preds_idx)):
                        logger.info(
                            f"{temp} predictions do not overlap with the "\
                            f"centroids from the polygons originally marked "\
                            f"as ambiguous."
                        )
                    
                    # Add suppressed records to suppressed logs
                    temp = img_preds.index.difference(relevant_preds_idx)
                    suppressed = update_supressed_gdf(
                        suppressed,
                        img_preds.loc[temp].copy(),
                        "Prediction does not overlap with the centroids from "\
                        "the polygons originally marked as ambiguous"
                    )
                    img_preds = img_preds.loc[relevant_preds_idx]

                    # Recalculate wordid values
                    img_preds["wordid"] = img_preds\
                        .groupby("groupid", as_index = True)\
                        .wordid\
                        .rank("first", ascending = True)
                    
                    # add img_preds to processed list
                    processed_preds.append(img_preds)
            
            # combine processed predictions
            processed_preds = concat(processed_preds, ignore_index = True)
            # save processed predictions out
            processed_preds.to_file(save_preds_dir.joinpath(
                img_meta["tiff_stem"] + ".gpkg"
            ))
        # save suppressed predictions
        suppressed.to_file(suppressed_fp)

    except Exception as e:
        logger.error(e, exc_info = True)
        raise
