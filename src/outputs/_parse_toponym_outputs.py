"""
Script contains the functions used for parsing outputs from the
ToponymExtractor model.

ToponymExtractor sourced from:
https://github.com/SesamePaste233/ToponymExtractor/tree/main
"""
# Imports
from typing import Final
from pathlib import Path
from logging import getLogger
from pickle import load as load_pickle
import progressbar
from numpy import array, clip, ndarray
from geopandas import GeoDataFrame
from pandas import DataFrame
from edina import get_transformer_from_geodataframe
from shapely import Polygon
from rasterio.transform import GCPTransformer

progressbar.streams.flush()

logger = getLogger(name = __name__)
logger.propagate = True

_WIDGETS: Final[list] = [
    ' [', progressbar.widgets.Counter(format='%(value)d of %(max_value)d'),
    ' (', progressbar.widgets.Percentage(), ')] ',
    progressbar.widgets.GranularBar(),
    ' ', progressbar.Timer(), ' | ',
     progressbar.ETA(), '|'
]


def read_pickle_queue(path: Path) -> list[dict]:
    """
    Reads specific pickle file format containing ToponymExtractor
    predictions.

    ToponymExtractor predictions were saved as a FIFO queue of
    dictionaries in pickle (.pkl) format.
    """
    outputs = []
    f =  open(path, "rb")
    try:
        while 1:
            outputs.append(load_pickle(f))
    except EOFError as _:
        # all outputs have been read.
        pass
    except Exception as e:
        # raise any other error
        raise
    finally:
        f.close()
    return outputs


def georeference_geometries(
    transformer: GCPTransformer, coords: ndarray
) -> ndarray:
    """
    Convert pixel coordinates to a geometric coordinate using the
    transformer.

    Parameters
    ----------
    transformer: GCPTransformer.
        Required. Transformer using georeferenced control points to
        convert pixel coordinates to geo coordinates. Can provide any
        object with an .xy() method.
    
    coords: numpy array.
        Required. 2d-array of shape [N, 2]. Column 0 representing x
        coords and column 1 representing y coords.

    Returns
    -------
    Numpy array of shape [N, 2].
        Converted coordinate values.
    """
    return array([*zip(*transformer.xy(coords[:,1], coords[:,0]))])


def _convert_ToponymExtractor_outputs_to_gdf_without_geotransforms(
    out: list[dict[str, str | list[list[dict[str, str | list[list[float]]]]]]],
    png_h: int,
    png_w: int
) -> tuple[GeoDataFrame, DataFrame]:
    """
    Create GeoDataFrame from ToponymExtractor outputs.

    ToponymExtractor sourced from:
    https://github.com/SesamePaste233/ToponymExtractor/tree/main

    Parameters
    ----------
    out: ToponymExtractor outputs.
        Required. ToponymExtractor outputs are saved in ICDAR 2025
        map text dataset format, which is a list of dictionaries:
            - "image": string. Name of image png the associated outputs
            are inferenced from.
            - "groups": list of list of dictionaries. Each list of
            dictionaries is a toponym - a collection of captured words
            or characters. Each dictionary contains:
                - "vertices": list of list of floats. Pixel x, y
                locations representing the corners for the corresponding
                text masks.
                - "text": string. Parsed text read from the map.
    png_h: int.
        Required. Height dimension of the image passed to the
        ToponymExtractor model. Needed to clip image vertices.
    png_w: int.
        Required. Width dimension of the image passed to the
        ToponymExtractor model. Needed to clip image vertices.
    

    Returns
    -------
    Two element tuple:
    1. GeoDataFrame. Extracted data, each record represents a word with
    the associated mask. Columns are:
        - png_filename: string. Name of image png the associated outputs
        are inferenced from.
        - groupid: int. Which toponym the word record belongs to. Count
        starts at 0.
        - wordid: int. Order the word record belongs to within the
        toponym group. Count starts at 0.
        - word: string. Parsed text from the ToponymExtractor.
        - score: string. Confidence score for the predicted word.
        - geometry: Polygon. Mask for the associated text, coordinates
        are in OSGB36 format (EPSG:27700).
    2. DataFrame. Records log images that ToponymExtractor did not
    produce outputs for due to errors at inference time. Columns are:
        - png_filename: string. Name of image png the ToponymExtractor
        errored on during inference.
        - error: string. Details the error that was thrown.
    """
    pngs, errors, data = set(), [], []
    progress = progressbar\
        .progressbar(out, widgets = _WIDGETS, prefix = "Reading predictions:")
    for image in progress:
        # Check record has not been seen before
        if image["image"] not in pngs:
            # Update seen pngs log
            pngs.add(image["image"])

            # some records errored out - so only upack those with expected
            # formats
            if isinstance(image.get("groups"), list):
                for i, group in enumerate(image["groups"]):
                    for j, word in enumerate(group):
                        # Create record
                        record = {
                            "png_filename": image["image"],
                            "groupid": i,
                            "wordid": j,
                            "word": word["text"],
                            "score": word["score"]
                        }
                        # Clip vertices to within the bounds of image
                        vertices = array(word["vertices"])
                        vertices[:, 0] =\
                            clip(vertices[:, 0], 0., float(png_w - 1))
                        vertices[:, 1] =\
                            clip(vertices[:, 1], 0., float(png_h - 1))
                        vertices = vertices.tolist()

                        record["geometry"] = Polygon(vertices).buffer(0)
                        
                        # Add record to data
                        data.append(record)

            elif "groups" in image:
                # Error handling - Format 1
                errors.append({
                    "png_filename": image["image"],
                    "error": image["groups"] + " - unspecified error."
                })
            elif "error" in image:
                # Error handling - Format 2
                errors.append({
                    "png_filename": image["image"], "error": image["error"]
                })
            else:
                # Error handling - catch all
                errors.append({
                    "png_filename": image["image"],
                    "error": "Parsing error - Unrecognised format."
                })

    # Convert record lists to frames
    data, errors = GeoDataFrame(data), DataFrame(errors)

    # log parsing details
    pngs = data["png_filename"].nunique()
    toponyms = int(data.groupby("png_filename")["groupid"].nunique().sum())
    words = len(data)
    logger.info(
        f"Outputs extracted for {pngs:,} PNGs: {words:,} text instances "\
        f"detected across {toponyms:,} groups (toponymns). "\
        f"{data["geometry"].isna().sum():,} records missing masks."
    )
    if len(errors):
        logger\
            .info(f"Encountered error records for {len(errors)} PNG records.")

    return data, errors


def _convert_ToponymExtractor_outputs_to_gdf_with_geotransforms(
    out: list[dict[str, str | list[list[dict[str, str | list[list[float]]]]]]],
    control_points: GeoDataFrame,
    png_h: int,
    png_w: int
) -> tuple[GeoDataFrame, DataFrame]:
    """
    Create GeoDataFrame from ToponymExtractor outputs.

    ToponymExtractor sourced from:
    https://github.com/SesamePaste233/ToponymExtractor/tree/main

    Parameters
    ----------
    out: ToponymExtractor outputs.
        Required. ToponymExtractor outputs are saved in ICDAR 2025
        map text dataset format, which is a list of dictionaries:
            - "image": string. Name of image png the associated outputs
            are inferenced from.
            - "groups": list of list of dictionaries. Each list of
            dictionaries is a toponym - a collection of captured words
            or characters. Each dictionary contains:
                - "vertices": list of list of floats. Pixel x, y
                locations representing the corners for the corresponding
                text masks.
                - "text": string. Parsed text read from the map.
    control_points: GeoDataFrame.
        Required. GeoDataFrame containing the details of the control
        points used for georeferencing each png. GeoDataFrame requires
        the fields: "png_filename", "pixel_x", "pixel_y", "geometry".
    png_h: int.
        Required. Height dimension of the image passed to the
        ToponymExtractor model. Needed to clip image vertices.
    png_w: int.
        Required. Width dimension of the image passed to the
        ToponymExtractor model. Needed to clip image vertices.
    

    Returns
    -------
    Two element tuple:
    1. GeoDataFrame. Extracted data, each record represents a word with
    the associated mask. Columns are:
        - png_filename: string. Name of image png the associated outputs
        are inferenced from.
        - groupid: int. Which toponym the word record belongs to. Count
        starts at 0.
        - wordid: int. Order the word record belongs to within the
        toponym group. Count starts at 0.
        - word: string. Parsed text from the ToponymExtractor.
        - score: string. Confidence score for the predicted word.
        - geometry: Polygon. Mask for the associated text, coordinates
        are in OSGB36 format (EPSG:27700).
    2. DataFrame. Records log images that ToponymExtractor did not
    produce outputs for due to errors at inference time. Columns are:
        - png_filename: string. Name of image png the ToponymExtractor
        errored on during inference.
        - error: string. Details the error that was thrown.
    """
    pngs, errors, data = set(), [], []
    progress = progressbar\
        .progressbar(out, widgets = _WIDGETS, prefix = "Reading predictions:")
    for image in progress:
        # Check record has not been seen before
        if image["image"] not in pngs:
            # Update seen pngs log
            pngs.add(image["image"])

            # some records errored out - so only upack those with expected
            # formats
            if isinstance(image.get("groups"), list):
                for i, group in enumerate(image["groups"]):
                    for j, word in enumerate(group):
                        # Create record
                        record = {
                            "png_filename": image["image"],
                            "groupid": i,
                            "wordid": j,
                            "word": word["text"],
                            "score": word["score"]
                        }
                        # Clip vertices to within the bounds of image
                        vertices = array(word["vertices"])
                        vertices[:, 0] =\
                            clip(vertices[:, 0], 0., float(png_w - 1))
                        vertices[:, 1] =\
                            clip(vertices[:, 1], 0., float(png_h - 1))
                        vertices = vertices.tolist()

                        # Get georeference control points transformer
                        gcp_trans = control_points.loc[
                            control_points["png_filename"] == image["image"]
                        ]
                        if len(gcp_trans):
                            gcp_trans =\
                                get_transformer_from_geodataframe(gcp_trans)
                            # Convert pixel location coordinates to latitude/
                            # longitude and create geometry field
                            record["geometry"] = Polygon([
                                gcp_trans.xy(r, c) for c, r in vertices
                            ])
                            # Documentation recommends calling close on
                            # GCPTransformer after calling transforms
                            gcp_trans.close()
                        else:
                            logger.warning(
                                f"Georeferencing control points not found "\
                                f"for image {image["image"]}"
                            )
                            record["geometry"] = None
                        
                        # Add record to data
                        data.append(record)

            elif "groups" in image:
                # Error handling - Format 1
                errors.append({
                    "png_filename": image["image"],
                    "error": image["groups"] + " - unspecified error."
                })
            elif "error" in image:
                # Error handling - Format 2
                errors.append({
                    "png_filename": image["image"], "error": image["error"]
                })
            else:
                # Error handling - catch all
                errors.append({
                    "png_filename": image["image"],
                    "error": "Parsing error - Unrecognised format."
                })

    # Convert record lists to frames
    data = GeoDataFrame(data, crs = control_points.crs)
    errors = DataFrame(errors)

    # log parsing details
    pngs = data["png_filename"].nunique()
    toponyms = int(data.groupby("png_filename")["groupid"].nunique().sum())
    words = len(data)
    logger.info(
        f"Outputs extracted for {pngs:,} PNGs: {words:,} text instances "\
        f"detected across {toponyms:,} groups (toponymns). "\
        f"{data["geometry"].isna().sum():,} records missing masks."
    )
    if len(errors):
        logger\
            .info(f"Encountered error records for {len(errors)} PNG records.")

    return data, errors


def convert_ToponymExtractor_outputs_to_gdf(
    out: list[dict[str, str | list[list[dict[str, str | list[list[float]]]]]]],
    png_h: int,
    png_w: int,
    control_points: GeoDataFrame | None = None
) -> tuple[GeoDataFrame, DataFrame]:
    """
    Create GeoDataFrame from ToponymExtractor outputs.

    ToponymExtractor sourced from:
    https://github.com/SesamePaste233/ToponymExtractor/tree/main

    Parameters
    ----------
    out: ToponymExtractor outputs.
        Required. ToponymExtractor outputs are saved in ICDAR 2025
        map text dataset format, which is a list of dictionaries:
            - "image": string. Name of image png the associated outputs
            are inferenced from.
            - "groups": list of list of dictionaries. Each list of
            dictionaries is a toponym - a collection of captured words
            or characters. Each dictionary contains:
                - "vertices": list of list of floats. Pixel x, y
                locations representing the corners for the corresponding
                text masks.
                - "text": string. Parsed text read from the map.
    png_h: int.
        Required. Height dimension of the image passed to the
        ToponymExtractor model. Needed to clip image vertices.
    png_w: int.
        Required. Width dimension of the image passed to the
        ToponymExtractor model. Needed to clip image vertices.
    control_points: GeoDataFrame or None.
        Optional. GeoDataFrame containing the details of the control
        points used for georeferencing each png. GeoDataFrame requires
        the fields: "png_filename", "pixel_x", "pixel_y", "geometry". If
        no control points are passed, then polygons are not
        georeferenced.

    Returns
    -------
    Two element tuple:
    1. GeoDataFrame. Extracted data, each record represents a word with
    the associated mask. Columns are:
        - png_filename: string. Name of image png the associated outputs
        are inferenced from.
        - groupid: int. Which toponym the word record belongs to. Count
        starts at 0.
        - wordid: int. Order the word record belongs to within the
        toponym group. Count starts at 0.
        - word: string. Parsed text from the ToponymExtractor.
        - score: string. Confidence score for the predicted word.
        - geometry: Polygon. Mask for the associated text, if 
        georeferencing was applied, then coordinates are in OSGB36
        format (EPSG:27700), otherwise polygon coordinates are pixel
        coordinate values.
    2. DataFrame. Records log images that ToponymExtractor did not
    produce outputs for due to errors at inference time. Columns are:
        - png_filename: string. Name of image png the ToponymExtractor
        errored on during inference.
        - error: string. Details the error that was thrown.
    """
    if control_points is not None:
        return _convert_ToponymExtractor_outputs_to_gdf_with_geotransforms(
            out = out,
            control_points = control_points,
            png_h = png_h,
            png_w = png_w
        )
    else:
        return _convert_ToponymExtractor_outputs_to_gdf_without_geotransforms(
            out = out, png_h = png_h, png_w = png_w
        )
