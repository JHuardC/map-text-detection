"""
Builds custom metrics using the predictions and ground truths provided.

Can optionally provide a .tiff image file, which if provided only the
prediction and ground truth records that intersect with the image area
will be selected.

Produces the following metrics:

- Loose Toponym Detection Recall: Poportion of ground-truth toponyms
detected. Detection is counted by IoU between the convex hulls of the
ground-truth and prediction toponym polygons.

- Loose Toponym Detection Precision: Poportion of predicted toponyms
determinded to be true-positive by the IoUs between the convex hulls of
the ground-truth and prediction toponym polygons.

- Strict Toponym Detection Recall: Proportion of ground-truth toponym
masks detected. Detection is counted by reaching IoU thresholds for each
polygon in the ground-truth label.

- Strict Toponym Detection Precision: Poportion of predicted toponyms
determinded to be true-positive by reaching IoU thresholds for each of
the ground-truth and prediction polygons that make up each word toponym.

- Loose Toponym Recognition Recall: Proportion of loose-detected
ground-truth toponyms with matching text between the combined
ground-truth polygon and the associated combined prediction polygon.
Word labels for the toponym are concatenated with spaces as the
delimeter.

- Loose Toponym Recognition Precision: Proportion of predicted toponyms
with matching text between the loose detected ground-truth polygon and
the associated prediction polygon. Word labels for the toponym are
concatenated with spaces as the delimeter.

- Strict Toponym Recognition Recall: Proportion of strict-detected
ground-truth toponyms with matching text between each of the
ground-truth polygons and their associated prediction polygons within a
toponym.

- Strict Toponym Recognition Precsion: Proportion of predicted toponyms
with matching text between each the strict detected ground-truth
polygons and their associated prediction polygons within a toponym.

These metrics are provided for all predictions, and broken down by each
image.

Metrics are output in a JSON form:

```
{
    "results": {
        'loose-detection-precision': float,
        'loose-detection-recall': float,
        'loose-recognition-precision': float,
        'loose-recognition-recall': float,
        'strict-detection-precision': float,
        'strict-detection-recall': float,
        'strict-recognition-precision': float,
        'strict-recognition-recall': float
    },
    "images": {
        "image-filename-1": {
            'loose-detection-precision': float,
            'loose-detection-recall': float,
            'loose-recognition-precision': float,
            'loose-recognition-recall': float,
            'strict-detection-precision': float,
            'strict-detection-recall': float,
            'strict-recognition-precision': float,
            'strict-recognition-recall': float
        },
        ...,
        "image-filename-N": {
            'loose-detection-precision': float,
            'loose-detection-recall': float,
            'loose-recognition-precision': float,
            'loose-recognition-recall': float,
            'strict-detection-precision': float,
            'strict-detection-recall': float,
            'strict-recognition-precision': float,
            'strict-recognition-recall': float
        }
    }
}
```
"""
# Imports
from typing import Final, Any
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from os import environ
from shapely import Geometry
from geopandas import GeoDataFrame, sjoin
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

def clip_geometries(
    gdf: GeoDataFrame, shape: Geometry, shape_crs: Any
) -> GeoDataFrame:
    """
    Filters `gdf` GeoDataFrame records to those intersecting with
    `shape` and takes the intersections between geometries and `shape`.
    """
    gdf = gdf.to_crs(shape_crs)
    gdf = gdf[gdf.geometry.intersects(shape)]
    gdf["geometry"] = gdf.geometry.intersection(shape)
    return gdf


def get_pred_ground_truth_overlaps(
    preds: GeoDataFrame,
    gts: GeoDataFrame,
    iou_thresh: float,
    preds_label_col: str,
    gt_label_col: str
) -> GeoDataFrame:
    """
    Performs a spatial join between predictions (`preds`) and
    ground-truth labels (`gts`) and suppresses certain overlaps.

    Calulates Intersection over Union (iou) between joined records and
    provides the "text_match" field; a boolean field that indicates
    word labels between `preds` and `gts` are the same.

    Spatial join uses "inner" join kind and "intersects" predicate.
    Ground-truth labels flagged as truncated or illegible have matches
    suppressed.
    Max iou is selected for each ground-truth label.
    If iou does not reach the required threshold value, then the record
    will be suppressed.

    Parameters
    ----------
    preds: GeoDataFrame.
        Required. Prediction labels.

    gts: GeoDataFrame.
        Required. Ground-truth labels. Must include boolean fields:
        "truncated" and "illegible".
    
    iou_thresh: float.
        Required. Threshold value for "iou" field; any record with an
        iou below this threshold will be suppressed.
    
    preds_label_col: str.
        Required. Name of the field containing the predicted label's
        text. The values in this field will be checked against the
        values in the `gt_label_col` field.
    
    gt_label_col: str.
        Required. Name of the field containing the ground-truth label's
        text. The values in this field will be checked against the
        values in the `preds_label_col` field.
    
    Returns
    -------
    GeoDataFrame. The "geometry" field will contain the polygons for
    the prediction labels. The "index_right" field will refer to the
    index values of the ground-truth GeoDataFrame. Includes "iou" and
    "text_match" fields.
    """
    # Spatial join prediction labels and ground-truth labels
    pred_gts = sjoin(
        left_df = preds,
        right_df = gts,
        how = "inner",
        predicate = "intersects"
    )
    # Exclude overlaps where ground-truth labels are truncated or illegible
    pred_gts = pred_gts[((~pred_gts.truncated) & (~pred_gts.illegible))]
    # Calculate iou
    ious = gts.loc[pred_gts.index_right, "geometry"]
    ious = (
        pred_gts.geometry.intersection(ious, align = False).area
        / pred_gts.geometry.union(ious, align = False).area
    )
    pred_gts["iou"] = ious.array
    # Select max iou for each ground truth label
    pred_gts.sort_values(
        by = "iou", ascending = False, ignore_index = False, inplace = True
    )
    pred_gts.drop_duplicates(
        subset = "index_right",
        keep = "first",
        ignore_index = False,
        inplace = True
    )
    # Suppress records that do not meet iou threshold
    pred_gts = pred_gts[pred_gts.iou >= iou_thresh]
    # Flag matching words
    gt_label_col = (
        f"{gt_label_col}_right"
        if gt_label_col in preds.columns
        else gt_label_col
    )
    preds_label_col = (
        f"{preds_label_col}_left"
        if preds_label_col in gts.columns
        else preds_label_col
    )
    pred_gts["text_match"] = (
        pred_gts[preds_label_col] == pred_gts[gt_label_col]
    )
    return pred_gts


if __name__ == "__main__":
    # Imports
    from functools import partial
    from json import load as load_json, dump as dump_json
    from pprint import pformat
    from shapely import box
    from rasterio import open as open_raster
    from geopandas import read_file
    from project_utils import parse_path, build_argument_parser, build_logger
    from outputs import build_toponym_gdf

    parser = build_argument_parser(filename = FILENAME, docstr = __doc__)
    parser.add_argument(
        "preds",
        action = "store",
        type = str,
        metavar = "read/geo/predictions/from",
        help =\
            "Required. Specify directory to read geo-encoded predictions "\
            "from. The filenames in the directory are expected to correspond "\
            "to the TIFF filenames the text instances belong within. Can "\
            "provide a relative or absolute path; relative paths will be set "\
            "relative to the path variable specified in config."
    )
    parser.add_argument(
        "gts",
        action = "store",
        type = str,
        metavar = "read/geo/ground/truths/from",
        help =\
            "Required. Specify directory to read geo-encoded predictions "\
            "from. The filenames in the directory are expected to correspond "\
            "to the TIFF filenames the text instances belong within. Can "\
            "provide a relative or absolute path; relative paths will be set "\
            "relative to the path variable specified in config."
    )
    parser.add_argument(
        "dest",
        action = "store",
        type = str,
        metavar = "save/metrics/to/json",
        help =\
            "Required. Specify JSON filepath to save custom metrics to. "\
            "Can provide a relative or absolute path; relative paths will be "\
            "set relative to the path variable specified in config."
    )
    parser.add_argument(
        "--tiff-dir",
        action = "store",
        type = str,
        metavar = "path/to/tiff/dir",
        dest = "tiff_dir",
        default = None,
        help =\
            "Optional. Path to directory containing TIFF files. Use this "\
            "argument to constrain metrics within the bounds of the tiff "\
            "file Can provide relative or absolute paths; relative paths "\
            "will be set against path variable specified in config."
    )
    parser.add_argument(
        "--ext",
        action = "store",
        type = str,
        metavar = ".EXT",
        default = ".gpkg",
        dest = "ext",
        help =\
            "Optional. Specify file extension for the files storing "\
            "geo-encoded predictions and ground-truth labels. If no argument "\
            "is passed, then the default extension is \".gpkg\". The "\
            "argument provided will be combined with the predictions "\
            "directory argument to perform glob search."
    )
    parser.add_argument(
        "--iou",
        action = "store",
        type = float,
        metavar = "THRESHOLD",
        default = 0.5,
        dest = "iou",
        help =\
            "Optional. Default 0.5. Threshold for intersection over union "\
            "(iou) scores between prediction labels and ground truth labels; "\
            "any iou value above this threshold will be considered a match "\
            "between the prediction and ground truth polygon."
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
        preds_dir = parse_path(cla_args.preds, config["relative_path"])
        if not preds_dir.exists():
            raise ValueError(
                f"Command line argument for the directory to the "\
                f"georeferenced predictions does not lead to an existing "\
                f"destination. Argument passed: {cla_args.preds}"
            )
        if preds_dir.is_file():
            raise ValueError(
                f"Command line argument for the directory to the "\
                f"georeferenced predictions lead to a file, rather than a "\
                f"directory. Argument passed: {cla_args.preds}"
            )
        gt_dir = parse_path(cla_args.gts, config["relative_path"])
        if not gt_dir.exists():
            raise ValueError(
                f"Command line argument for the directory to the "\
                f"georeferenced ground truth labels does not lead to an "\
                f"existing destination. Argument passed: {cla_args.gts}"
            )
        if gt_dir.is_file():
            raise ValueError(
                f"Command line argument for the directory to the "\
                f"georeferenced ground truth labels leads to a file, rather "\
                f"than a directory. Argument passed: {cla_args.gts}"
            )
        tiff_dir = parse_path(cla_args.tiff_dir, config["relative_path"])
        if not tiff_dir.exists():
            raise ValueError(
                f"Command line argument for the tiff directory does not lead "\
                f"to an existing destination. Argument passed: "\
                f"{cla_args.geodata}"
            )
        if tiff_dir.is_file():
            raise ValueError(
                f"Command line argument for the tiff directory leads to a "\
                f"file, rather than a directory. Value passed in command "\
                f"line: {cla_args.tiff_dir}"
            )
        save_to = parse_path(cla_args.dest, config["relative_path"])
        if save_to.suffix != ".json":
            raise ValueError(
                f"Command line argument to save performance metrics to does "\
                f"not lead to a JSON (.json suffix) file name. Argument "\
                f"passed: {cla_args.save_err_to}"
            )
        
        logger.debug(
            "Get filepaths to files storing geo-encoded polygon mask "\
            "predictions and associated tiffs from which the labels were "\
            "derived."
        )
        ext = cla_args.ext
        preds_gts_tiffs_fps = [*preds_dir.glob(f"*{ext}")]
        logger.info(f"Found {len(preds_gts_tiffs_fps):,} prections files.")
        preds_gts_tiffs_fps = [
            (
                preds_path,
                gt_dir.joinpath(f"{stem}{ext}"),
                None if tiff_dir is None else tiff_dir.joinpath(f"{stem}.tif")
            )
            for preds_path in preds_gts_tiffs_fps
            if gt_dir.joinpath(f"{(stem := preds_path.stem)}{ext}").exists()
        ]
        logger.info(
            f"Found {len(preds_gts_tiffs_fps):,} prediction files with "\
            f"associated ground-truth files."
        )

        logger.debug("Iterate through geo-encoded prediction files")
        preds_gts_tiffs_filepaths_iter = progressbar.progressbar(
            preds_gts_tiffs_fps, 
            widgets = _WIDGETS,
            prefix = "Calculating metrics:"
        )
        preds_path: Path
        gt_path: Path
        tiff_path: Path | None
        # Initialising dictionary for global (across all images)
        # precision-recall metrics
        global_metrics = {
            "loose-detection": 0,
            "loose-recognition": 0,
            "strict-detection": 0,
            "strict-recognition": 0,
        }
        total_gt_toponym_count: int = 0
        total_pred_toponym_count: int = 0
        # Initialise metrics dictionary
        metrics = {"images": dict()}
        for pred_path, gt_path, tiff_path in preds_gts_tiffs_filepaths_iter:
            # initialize image specific metrics dictionary
            img_metrics = dict()
            if (tiff_path is not None) and tiff_path.exists():
                logger.debug(
                    f"Clip predictions and ground-truth labels to tif bounds "\
                    f"for TIFF: {tiff_path.name}"
                )
                with open_raster(tiff_path, mode = "r") as tif:
                    tif_crs = tif.read_crs()
                    tif_bounds = box(*tif.bounds)

            logger.debug(
                f"Loading geo-encoded predictions file: {pred_path.name}"
            )
            preds, labels = map(read_file, (pred_path, gt_path))

            # Normalize word column string
            preds["word"] = preds.word.astype("string").fillna("")
            labels["word"] = labels.word.astype("string").fillna("")

            if (tiff_path is not None) and tiff_path.exists():
                logger.debug(
                    "Filtering records and clipping geometries to within the "\
                    "tiff bounding box."
                )
                clipper = partial(
                    clip_geometries, shape = tif_bounds, shape_crs = tif_crs
                )
                preds, labels = map(clipper, (preds, labels))

            # Remove non-text entries from ground-truth labels
            labels = labels[~labels.word.str.contains("$", regex = False)]
            # Mark illegible ground-truth labels with a binary field
            labels["illegible"] = labels.status.str.contains("U")

            # Build ground-truth and prediction toponym GeoDataFrames
            preds_topo, labels_topo = map(build_toponym_gdf, (preds, labels))

            # Add truncated and illegible flags for labels_topo
            label_topo_flags = labels.groupby("groupid", as_index = True).agg(
                gt_topo_size = ("wordid", "count"),
                illegible = ("illegible", any),
                truncated = ("truncated", any)
            )
            labels_topo = labels_topo.join(
                label_topo_flags, how = "inner", on = "groupid"
            )

            # Get count of ground-truth toponyms
            gt_toponym_count = int(
                ((~labels_topo.illegible) & (~labels_topo.truncated)).sum()
            )

            # Spatial join prediction toponyms and label toponyms
            loose_topos = get_pred_ground_truth_overlaps(
                preds = preds_topo,
                gts = labels_topo,
                iou_thresh = cla_args.iou,
                preds_label_col = "toponym",
                gt_label_col = "toponym"
            )

            # Calculate loose metrics
            loose_detections = len(loose_topos)
            loose_recognitions = int(loose_topos.text_match.sum())
            ## loose recall metrics
            ### detection
            k = "loose-detection-recall"
            img_metrics[k] = loose_detections / gt_toponym_count
            ### recognition
            k = "loose-recognition-recall"
            img_metrics[k] = loose_recognitions / gt_toponym_count
            ## loose precision metrics
            ### detection
            k = "loose-detection-precision"
            img_metrics[k] = loose_detections / len(preds_topo)
            ### recognition
            k = "loose-recognition-precision"
            img_metrics[k] = loose_recognitions / len(preds_topo)


            # Spatial join prediction labels and ground-truth labels
            strict_topos = get_pred_ground_truth_overlaps(
                preds = preds,
                gts = labels,
                iou_thresh = cla_args.iou,
                preds_label_col = "word",
                gt_label_col = "word"
            )
            # Check whether the matched label's word order aligns between
            # prediction labels and ground-truth labels
            strict_topos["order_match"] =\
                (strict_topos.wordid_left == strict_topos.wordid_right)

            # Get counts of detected groud-truth word labels per toponym
            strict_topos =\
                strict_topos.groupby("groupid_right", as_index = True)
            strict_topos = strict_topos.agg(
                detection_count = ("order_match", "sum"),
                recognition_count = ("text_match", "sum")
            )
            # Join counts of ground-truth labels
            strict_topos = strict_topos.join(label_topo_flags, how = "inner")
            # Get flags of detected and recognised toponyms
            strict_topos["detected"] =\
                (strict_topos.detection_count == strict_topos.gt_topo_size)
            strict_topos["recognised"] =\
                (strict_topos.recognition_count == strict_topos.gt_topo_size)

            # Calculate strict metrics
            strict_detections = int(strict_topos.detected.sum())
            strict_recognitions = int(strict_topos.recognised.sum())
            ## strict recall metrics
            ### detection
            k = "strict-detection-recall"
            img_metrics[k] = strict_detections / gt_toponym_count
            ### recognition
            k = "strict-recognition-recall"
            img_metrics[k] = strict_recognitions / gt_toponym_count
            ## strict precision metrics
            ### detection
            k = "strict-detection-precision"
            img_metrics[k] = strict_detections / len(preds_topo)
            ### recognition
            k = "strict-recognition-precision"
            img_metrics[k] = strict_recognitions / len(preds_topo)

            # update running counts
            global_metrics["loose-detection"] += loose_detections
            global_metrics["loose-recognition"] += loose_recognitions
            global_metrics["strict-detection"] += strict_detections
            global_metrics["strict-recognition"] += strict_recognitions
            total_gt_toponym_count += gt_toponym_count
            total_pred_toponym_count += len(preds_topo)

            metrics["images"][tiff_path.name] = img_metrics

        # Calculate global precision-recall metrics
        for k in [*global_metrics.keys()]:
            temp = global_metrics.pop(k)
            global_metrics[f"{k}-recall"] = temp / total_gt_toponym_count
            global_metrics[f"{k}-precision"] = temp / total_pred_toponym_count
        metrics["results"] = global_metrics

        logger.info(f"Results:\n\n{pformat(metrics["results"])}")
        logger.debug("Saving metrics out to json.")
        with open(save_to, mode = "w") as dest:
            dump_json(metrics, dest)

    except Exception as e:
        logger.error(e, exc_info = True)
        raise

