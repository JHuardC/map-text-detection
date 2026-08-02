"""
Convert map text instances in geodata format to ICDAR 2025 -
Historical Map Text Recognition format.

ICDAR 2025 - Historical Map Text Recognition format is a JSON-friendly
format with the following structure for ground-truth labels, and a
similar structure except without "illegible" and "truncated" kwargs:

```
[ # Begin a list of images
    {
        "image": "IMAGE_NAME_1",
        "groups": [ # Begin a list of phrase groups for the image
            [  # A phrase: A list of word dictionaries for the phrase
                {
                    "vertices": [[x1, y1], [x2, y2], ..., [xN_1, yN_1]],
                    "text": "TEXT1",
                    "illegible": True/False,
                    "truncated": True/False
                },
                ...,
                {
                    "vertices": [[x1, y1], [x2, y2], ..., [xN_J, yN_J]],
                    "text": "TEXTJ",
                    "illegible": True/False,
                    "truncated": True/False
                }
            ],
            ...,
            [ # Another phrase
                {
                    "vertices": [[x1, y1], [x2, y2], ..., [xN_1, yN_1]],
                    "text": "TEXT1",
                    "illegible": True/False,
                    "truncated": True/False
                },
                ...
            ]
        ]
    },
    ...,
    {
        "image": "IMAGE_NAME_M",
        "groups": [
            [{"vertices": [[x1, y1], ..., [xN, yN]], ...}, ...], ...
        ]
    }
]
```

Historical Map Text Recognition Competition:
https://rrc.cvc.uab.es/?ch=32
"""
# Imports
from typing import Final
from collections.abc import Callable
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from os import environ
from numpy import ndarray
from shapely import get_coordinates
from geopandas import GeoDataFrame
from outputs import pixel_ref_geometries
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

def _get_word_order(d: dict) -> int | float:
    """
    Gets "wordid" key from the dictionary, used in ordering the words of
    the toponym in ICDAR 2025 format.
    """
    return d.get("wordid")


def get_predictions_in_ICDAR2025_format(
    gdf: GeoDataFrame
) -> list[list[dict[str, str | list[list[int]]]]]:
    """
    Extracts word predictions for a single image from a GeoDataFrame
    to ICDAR 2025 - Historical Map Text Recognition format.

    Parameters
    ----------
    gdf: GeodDataFrame.
        Required. GeoDataFrame for the predictions derived from a single
        image. Columns "groupid", "wordid", "text", and "geometry" are
        required. Before being passed, geometry objects should already
        be converted to `LINEARRING` by calling `.exterior`.
    
    Returns
    -------
    ICDAR 2025 - Historical Map Text Recognition format word
    predictions:

    ```
    [ # Begin a list of phrase groups for the image
        [  # A phrase: A list of word dictionaries for the phrase
            {
                "vertices": [[x1, y1], [x2, y2], ..., [xN_1, yN_1]],
                "text": "TEXT1"
            },
            ...,
            {
                "vertices": [[x1, y1], [x2, y2], ..., [xN_J, yN_J]],
                "text": "TEXTJ"
            }
        ],
        ...,
        [ # Another phrase
            {
                "vertices": [[x1, y1], [x2, y2], ..., [xN_1, yN_1]],
                "text": "TEXT1"
            },
            ...
        ]
    ]
    ```
    """
    toponyms = []
    for groupid in gdf.groupid.unique():
        toponyms.append(gdf\
            .loc[(gdf.groupid == groupid), ["wordid", "text", "geometry"]]\
            .to_dict(orient = "records")
        )

    for groupid in range(len(toponyms)):
        # Order words within each toponym
        toponyms[groupid] = sorted(toponyms[groupid], key = _get_word_order)
        for word in toponyms[groupid]:
            # remove wordid
            del word["wordid"]
            # Ensure text is in string format
            word["text"] = str(word["text"])
            # convert shape to list of xy pixel coordinates
            word["vertices"] = get_coordinates(word.pop("geometry")).tolist()
    
    return toponyms


def get_ground_truths_in_ICDAR2025_format(
    gdf: GeoDataFrame
) -> list[list[dict[str, str | list[list[int]]]]]:
    """
    Extracts word predictions for a single image from a GeoDataFrame
    to ICDAR 2025 - Historical Map Text Recognition format.

    Parameters
    ----------
    gdf: GeodDataFrame.
        Required. GeoDataFrame for the ground truth labels refering to
        text within a single image. Columns "groupid", "wordid", "text",
        "status", "truncated", and "geometry" are required. Before being
        passed, geometry objects should already be converted to
        `LinearRing` by calling `.exterior`.
    
    Returns
    -------
    ICDAR 2025 - Historical Map Text Recognition format word
    predictions:

    ```
    [ # Begin a list of phrase groups for the image
        [  # A phrase: A list of word dictionaries for the phrase
            {
                "vertices": [[x1, y1], [x2, y2], ..., [xN_1, yN_1]],
                "text": "TEXT1",
                "illegible": True/False,
                "truncated": True/False
            },
            ...,
            {
                "vertices": [[x1, y1], [x2, y2], ..., [xN_J, yN_J]],
                "text": "TEXTJ",
                "illegible": True/False,
                "truncated": True/False
            }
        ],
        ...,
        [ # Another phrase
            {
                "vertices": [[x1, y1], [x2, y2], ..., [xN_1, yN_1]],
                "text": "TEXT1",
                "illegible": True/False,
                "truncated": True/False
            },
            ...
        ]
    ]
    ```
    """
    cols = ["wordid", "text", "status", "truncated", "geometry"]
    toponyms = []
    for groupid in gdf.groupid.unique():
        toponyms.append(gdf\
            .loc[(gdf.groupid == groupid), cols]\
            .to_dict(orient = "records")
        )

    for groupid in range(len(toponyms)):
        # Order words within each toponym
        toponyms[groupid] = sorted(toponyms[groupid], key = _get_word_order)
        for word in toponyms[groupid]:
            # remove wordid
            del word["wordid"]
            # Ensure text is in string format
            word["text"] = str(word["text"])
            # convert shape to list of xy pixel coordinates
            word["vertices"] = get_coordinates(word.pop("geometry")).tolist()
            # record whether the word was illegible
            word["illegible"] = ("U" in word.pop("status"))
    
    return toponyms

if __name__ == "__main__":
    # Imports
    from functools import partial
    from json import load as load_json, dump as dump_json
    from shapely import box
    from rasterio import open as open_raster
    from rasterio.transform import AffineTransformer
    from geopandas import read_file
    from project_utils import parse_path
    from project_utils import parse_path, build_argument_parser, build_logger

    parser = build_argument_parser(
        filename = FILENAME,
        description =\
            "Convert map text instances in geodata format to ICDAR 2025 - "\
            "Historical Map Text Recognition format."
    )
    parser.add_argument(
        "geodata",
        action = "store",
        type = str,
        metavar = "read/geo/preds/from",
        help =\
            "Required. Specify directory to read geo-encoded labels "\
            "from. The filenames in the directory are expected to correspond "\
            "to the TIFF filenames the text instances belong within. Can "\
            "provide a relative or absolute path; relative paths will be set "\
            "relative to the path variable specified in config."
    )
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
        "dest",
        action = "store",
        type = str,
        metavar = "save/converted/to/json",
        help =\
            "Required. Specify JSON filepath to save ICDAR 2025 formatted "\
            "labels. Can provide a relative or absolute path; relative "\
            "paths will be set relative to the path variable specified in "\
            "config."
    )
    parser.add_argument(
        "--ext",
        action = "store",
        nargs = "+",
        type = str,
        metavar = ".EXT",
        default = [".gpkg"],
        dest = "exts",
        help =\
            "Optional. Specify file extensions for the files storing "\
            "geo-encoded labels. Can provide multiple extensions "\
            "separated by spaces. If no argument is passed, then the default "\
            "extension is \".gpkg\". The arguments provided will be combined "\
            "with the \"geodata\" directory argument to perform glob search."
    )
    parser.add_argument(
        "--ground-truth",
        action = "store_true",
        dest = "ground_truth",
        help =\
            "Optional. Argument that flags the labels to be processed as "\
            "ground-truth labels. These labels contain additional key-values "\
            "('illegible' and 'truncated' ) in their word dictionaries."
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
        src_dir = parse_path(cla_args.geodata, config["relative_path"])
        if not src_dir.exists():
            raise ValueError(
                f"Command line argument for the directory to the polygon "\
                f"mask predictions does not lead to an existing destination. "\
                f"Argument passed: {cla_args.geodata}"
            )
        if src_dir.is_file():
            raise ValueError(
                f"Command line argument for the directory to the polygon "\
                f"mask predictions to a file, rather than a directory. Value "\
                f"passed in command line: {cla_args.geodata}"
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
                f"Command line argument to save ICDAR 2025 formatted "\
                f"predictions to does not lead to a JSON (.json suffix) file "\
                f"name. Argument passed: {cla_args.save_err_to}"
            )
        
        logger.debug(
            "Get filepaths to files storing geo-encoded polygon mask "\
            "predictions and associated tiffs from which the labels were "\
            "derived."
        )
        geopreds_tiffs_fps = [
            (fp, tiff_path)
            for ext in cla_args.exts
            for fp in src_dir.glob(f"*{ext}")
            if (tiff_path := tiff_dir.joinpath(f"{fp.stem}.tif")).exists()
        ]

        logger.debug("Iterate through geo-encoded prediction files")
        labels_tiffs_filepaths_iter = progressbar.progressbar(
            geopreds_tiffs_fps,
            widgets = _WIDGETS,
            prefix = "Converting labels in files:"
        )
        icdar2025 = []
        for label_path, tiff_path in labels_tiffs_filepaths_iter:
            logger.debug(
                f"Getting relevant coordinate transformer and CRS from "\
                f"associacted associacted TIFF: {tiff_path.name}"
            )
            with open_raster(tiff_path, mode = "r") as tif:
                tif_transformer = AffineTransformer(tif.transform)
                tif_crs = tif.read_crs()
                tif_bounds = box(*tif.bounds)
            # provide tif_transformer to pixel_ref_transformer
            pxlref = partial(pixel_ref_geometries, tif_transformer)

            logger.debug(
                f"Loading geo-encoded predictions file: {label_path.name}"
            )
            gdf = read_file(label_path)
            gdf = gdf.to_crs(tif_crs)

            logger.debug(
                "Filtering and clipping geometries to within the tiff "
                "bounding box."
            )
            gdf = gdf[gdf.geometry.intersects(tif_bounds)]
            if not cla_args.ground_truth:
                gdf["geometry"] = gdf.geometry.intersection(tif_bounds)

            logger.debug(f"Converting geometries.")
            gdf["geometry"] = gdf\
                .geometry.transform(pxlref, include_z = False).exterior
            # change column name: word -> text
            gdf.rename(columns = {"word": "text"}, inplace = True)

            if cla_args.ground_truth:
                # remove non-text entries
                gdf = gdf[~gdf.text.str.contains("$", regex = False)]
                icdar2025.append({
                    "image": tiff_path.name,
                    "groups": get_ground_truths_in_ICDAR2025_format(gdf)
                })
            else:
                icdar2025.append({
                    "image": tiff_path.name,
                    "groups": get_predictions_in_ICDAR2025_format(gdf)
                })

        logger.debug("Saving toponyms in ICDAR 2025 format to json.")
        with open(save_to, mode = "w") as dest:
            dump_json(icdar2025, dest)

    except Exception as e:
        logger.error(e, exc_info = True)
        raise
