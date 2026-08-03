"""
Converts TIFF map images and associated labels to detectron2 training
data format.

For details surrounding detectron2 dataset formats, see:
https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html

Annotations format:
```
[ # list of dictionaries referencing images and their annotations
    { # A set of annotations for an image
        "file_name": example_img1.png,
        "height": Int (image height),
        "width": Int (image width),
        "image_id": Int,
        "annotations": [ # A list of dictionaries
            { # Each dictionary contains data for a specific anno
                "bbox": [xmin, ymin, xmax, ymax], # list of floats
                "bbox_mode": 0, # XYXY format
                "category_id": 1, # All annos are pos. inst. of text
                "text": [i1, ... , in], # ids representing chars
                "segmentation": [[x1, y1, ..., xm, ym]],
                "language": 6 # Language id value - always latin
            },
            ...,
            { # Another annotation record
                "bbox": [xmin, ymin, xmax, ymax],
                "bbox_mode": 0, # Always 0
                "category_id": 1, # Always 1
                "text": [i1, ... , ip],
                "segmentation": [[x1, y1, ..., xq, yq]],
                "language": 6 # Always 6 - latin
            }
        ]
    },
    ..., # more images
    ...,
    { # Another image
        "file_name": example_imgN.png,
        "height": Int,
        "width": Int,
        "image_id": Int,
        "annotations": [ # A list of dictionaries
            { # Each dictionary contains data for a specific anno
                "bbox": [xmin, ymin, xmax, ymax], # list of floats
                "bbox_mode": 0, # XYXY format
                "category_id": 1, # All annos are pos. inst. of text
                "text": [i1, ... , in], # ids representing chars
                "segmentation": [[x1, y1, ..., xm, ym]],
                "language": 6 # Language id value - always latin
            },
            ...,
            { # Another annotation record
                "bbox": [xmin, ymin, xmax, ymax],
                "bbox_mode": 0, # Always 0
                "category_id": 1, # Always 1
                "text": [i1, ... , ip],
                "segmentation": [[x1, y1, ..., xq, yq]],
                "language": 6 # Always 6 - latin
            }
        ]
    }
]
```
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
    from functools import partial
    from shutil import move
    from json import load as load_json, dump as dump_json
    from geopandas import read_file
    from edina import EDINATiffPNGConverter, get_transformer_from_geodataframe
    from outputs import pixel_ref_geometries, normalize_geometries
    from finetuning import get_annotations_in_detectron2_format
    from project_utils import parse_path, build_argument_parser, build_logger

    parser = build_argument_parser(
        filename = FILENAME,
        description =\
            "Converts TIFF map images and associated labels to detectron2 "\
            "training data format."
    )
    parser.add_argument(
        "tiff_dir",
        action = "store",
        type = str,
        metavar = "read/tiffs/from/dir",
        help =\
            "Required. Path to directory containing Edina downloaded tiff "\
            "files. Can provide relative or absolute paths; relative paths "\
            "will be set against path variable specified in config."
    )
    parser.add_argument(
        "labels_dir",
        action = "store",
        type = str,
        metavar = "read/geo/lbls/from/dir",
        help =\
            "Required. Specify directory to read geo-encoded ground-truth "\
            "labels from. The filenames in the directory are expected to "\
            "correspond to the TIFF filenames the text annotations belong to "\
            "Can provide a relative or absolute path; relative paths will be "\
            "set relative to the path variable specified in config."
    )
    parser.add_argument(
        "dest_dir",
        action = "store",
        type = str,
        metavar = "save/outputs/to/dir",
        help =\
            "Required. Specify directory to save clipped png files out to. "\
            "Can provide a relative or absolute path; relative paths will be "\
            "set relative to the path variable specified in config."
    )
    parser.add_argument(
        "--ext",
        action = "store",
        type = str,
        metavar = ".EXT",
        default = ".gpkg",
        dest = "ext",
        help =\
            "Optional. Specify file extension for the files storing the "\
            "geoencoded ground-truth labels. If no argument is passed, then "\
            "the default extension is \".gpkg\". The argument provided will "\
            "be combined with the predictions directory argument to perform "\
            "glob search."
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
        labels_dir = parse_path(cla_args.labels_dir, config["relative_path"])
        if not labels_dir.exists():
            raise ValueError(
                f"Command line argument for the directory to the polygon "\
                f"mask annotations does not lead to an existing destination. "\
                f"Argument passed: {cla_args.labels_dir}"
            )
        if labels_dir.is_file():
            raise ValueError(
                f"Command line argument for the directory to the polygon "\
                f"mask annotations to a file, rather than a directory. Value "\
                f"passed in command line: {cla_args.labels_dir}"
            )
        save_to = parse_path(cla_args.dest_dir, config["relative_path"])
        if not save_to.exists():
            raise ValueError(
                f"Command line argument to save detectron2 formatted data to "\
                f"does not lead to an existing destination. Argument passed: "\
                f"{cla_args.dest_dir}"
            )
        if save_to.is_file():
            raise ValueError(
                f"Command line argument to save detectron2 formatted data to "\
                f"leads to a file, rather than a directory. Value passed in "\
                f"command line: {cla_args.dest_dir}"
            )
        TRAIN_DIR: Final[Path] = save_to.joinpath("train_images")
        TEST_DIR: Final[Path] = save_to.joinpath("test_images")
        
        logger.debug(
            "Get filepaths to files storing geo-encoded annotations and "\
            "associated tiffs from which the labels were derived."
        )
        labels_tiffs_fps = [
            (fp, tiff_path)
            for fp in labels_dir.glob(f"*{cla_args.ext}")
            if (tiff_path := tiff_dir.joinpath(f"{fp.stem}.tif")).exists()
        ]
        logger.debug(f"Found {len(labels_tiffs_fps)} annotation-image pairs")

        # instance EDINATiffPNGConverter
        converter = EDINATiffPNGConverter(img_mode = config["image_mode"])

    except Exception as e:
        logger.error(e, exc_info = True)
        raise

    try:
        # set up kwargs for calling on converter
        converter_kwargs = {
            "tiff_paths": [el[1] for el in labels_tiffs_fps],
            "png_dest": save_to,
            "png_h": config["png_h"],
            "png_w": config["png_w"],
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
        ctrl_points = converter.convert_batch_tiff_to_pngs(**converter_kwargs)

        logger.debug("Build bounding boxes from PNG control points.")
        png_bounds = ctrl_points.dissolve(
            by = ["tiff_filename", "png_filename"], as_index = False
        )
        png_bounds["geometry"] = png_bounds.geometry.convex_hull
        png_bounds["image_id"] = [*range(len(png_bounds))]

        logger.debug("Add column to split images between train and test")
        png_bounds["split"] = "train"
        selection = png_bounds[["png_filename"]].sample(
            n = config["test_size"],
            random_state = config["random_state"],
            ignore_index = True
        )
        selection = png_bounds.png_filename.isin(set(selection.png_filename))
        png_bounds.loc[selection, "split"] = "test"

        logger.debug(
            "Iterate through geo-encoded label files and build annotations"
        )
        labels_tiffs_filepaths_iter = progressbar.progressbar(
            labels_tiffs_fps,
            widgets = _WIDGETS,
            prefix = "Converting data to detectron2 format:"
        )
        label_path: Path
        tiff_path: Path
        train_records = []
        test_records = []
        for label_path, tiff_path in labels_tiffs_filepaths_iter:
            logger.debug(
                f"Loading geo-encoded labels file: {label_path.name}"
            )
            labels = read_file(label_path)
            # remove non-text entries and illegible labels
            labels = labels[~labels.word.str.contains("$", regex = False)]
            labels = labels[~labels.status.str.contains("U", case = True)]
            labels = labels[~labels.word.isna()]
            labels["word"] = labels.word.astype("str")

            logger.debug(f"Get bounds for pngs derived from current TIFF.")
            selection = (png_bounds.tiff_filename == tiff_path.name)

            logger.debug(f"Iterate through pngs derived from current TIFF")
            for rec in png_bounds[selection].itertuples():
                png_fn: str = rec.png_filename
                logger.debug(f"Constrain control points to specific PNG")
                png_ctrl_points = ctrl_points[
                    (ctrl_points.png_filename == png_fn)
                ]

                # Construct records
                record = dict()
                record["file_name"] = png_fn
                record["height"] = int(png_ctrl_points.pixel_y.max() + 1)
                record["width"] = int(png_ctrl_points.pixel_x.max() + 1)
                record["image_id"] = int(rec.image_id)

                logger.debug(f"Constrain labels to specific png")
                png_labels = labels[labels.within(rec.geometry)]
                png_labels["geometry"] = normalize_geometries(
                    png_labels.geometry.intersection(rec.geometry)
                )

                logger.debug(f"Convert polygon coordinates to pixel locations")
                # Build transformer
                to_pixel_trans = partial(
                    pixel_ref_geometries,
                    get_transformer_from_geodataframe(png_ctrl_points)
                )
                # Apply transformer to geometries
                png_labels["geometry"] = png_labels.geometry.transform(
                    to_pixel_trans, include_z = False
                )

                # Add annotations to record
                record["annotations"] = get_annotations_in_detectron2_format(
                    gdf = png_labels, text_col = "word"
                )

                logger.debug(
                    f"Record creation complete, adding to train/test "\
                    f"collection"
                )
                move_file = partial(move, save_to.joinpath(png_fn))
                if rec.split == "train":
                    train_records.append(record)
                    _ = move_file(TRAIN_DIR.joinpath(png_fn))
                else:
                    test_records.append(record)
                    _ = move_file(TEST_DIR.joinpath(png_fn))

        logger.debug("Save train and test annotations out to JSON files.")
        with open(save_to.joinpath("train.json"), "w") as f:
            dump_json(train_records, f)
        with open(save_to.joinpath("test.json"), "w") as f:
            dump_json(test_records, f)

        logger.debug("Save out control points")
        ctrl_points.to_file(save_to.joinpath(f"control_points.{cla_args.ext}"))

    except Exception as e:
        logger.error(e, exc_info = True)
        raise
