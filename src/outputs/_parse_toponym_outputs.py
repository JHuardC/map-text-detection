"""
Script contains the functions used for parsing outputs from the
ToponymExtractor model.

ToponymExtractor sourced from:
https://github.com/SesamePaste233/ToponymExtractor/tree/main
"""
# Imports
from typing import Final
from logging import getLogger
import progressbar
from geopandas import GeoDataFrame
from pandas import DataFrame
from edina import get_transformer_from_geodataframe
from shapely import Polygon

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

def convert_ToponymExtractor_outputs_to_gdf(
    out: list[dict[str, str | list[list[dict[str, str | list[list[float]]]]]]],
    control_points: GeoDataFrame
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
                            "word": word["text"]
                        }

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
                                gcp_trans.xy(x, y) for x, y in word["vertices"]
                            ])
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
