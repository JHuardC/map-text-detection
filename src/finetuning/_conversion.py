"""
Function for converting georeferenced polygon masks to detectron2
format.
"""
# Imports
from typing import Final
from collections.abc import Hashable
from shapely import Polygon, get_coordinates
from geopandas import GeoDataFrame

# Type Aliases
_KV = tuple[str, int]

# Constants
_ANNOTATION_CONSTANTS: Final[tuple[_KV, _KV, _KV]] = (
    ("bbox_mode", 0), ("category_id", 1), ("language", 6)
)
_ID_CHAR_LOOKUP: dict[int, str] = dict(enumerate([
    ' ','!','"','#','$','%','&','\'','(',')','*','+',',','-','.','/','0','1',
    '2','3','4','5','6','7','8','9',':',';','<','=','>','?','@','A','B','C',
    'D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U',
    'V','W','X','Y','Z','[','\\',']','^','_','`','a','b','c','d','e','f','g',
    'h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y',
    'z','{','|','}','~'
]))
_CHAR_ID_LOOKUP: dict[str, int] = {v: k for k, v in _ID_CHAR_LOOKUP.items()}


def _get_dict_order(d: dict, ord_k: Hashable) -> int | float:
    """
    Used to order a collection of dictionaries; gets `ord_k` key whose
    values are used to order the dictionaries.
    """
    return d[ord_k]


def get_annotations_in_detectron2_format(
    gdf: GeoDataFrame,
    text_col: Hashable
) -> dict[
    str,
    str | int | dict[str, list[float] | list[list[float]] | int | str]
]:
    """
    Extracts the word polygon masks for a single image, contained in a
    GeoDataFrame, to detectron2 datasets format.

    Parameters
    ----------
    gdf: GeodDataFrame.
        Required. GeoDataFrame containig the word-level polygon
        annotations for a single image. Must contain column name
        matching the argument passed to `text_col`, and must contain a
        "geometry" column containing the Polygons.
    
    text_col: Hashable.
        Required. Column name for the annotated text value.
    
    Returns
    -------
    detectron2 formatted annotations. For details see:
    https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html

    Format:
    ```
    [ # list of dictionaries referencing the annotations for an image
        { # Each dictionary contains data for a specific anno
            "bbox": [xmin, ymin, xmax, ymax], # list of floats
            "bbox_mode": 0, # XYXY format
            "category_id": 1, # All annos are pos. inst. of text
            "text": [i1, ... , in], # ids representing chars
            "segmentation": [[x1, y1, ..., xm, ym]],
            "language": 6 # Language id value - always latin
        },
        ..., # more annotations
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
    ```
    """
    annotations = []
    # Cycle through each annotation dictionary
    for anno in gdf[[text_col, "geometry"]].to_dict(orient = "records"):

        # Extract the polygon from the record
        poly: Polygon = anno.pop("geometry")
        # add polygon details to annotation
        anno["bbox"] = list(poly.bounds)
        poly: list[list[float]] = get_coordinates(poly.exterior).tolist()
        if len(poly) < 3:
            # if there are less than 3 edges in the polygon, do not include
            # annotation
            continue
        anno["segmentation"] = [poly]

        # add text label - Converting characters to indices
        anno["text"] = [
            idx for el in anno.pop(text_col)
            if (idx := _CHAR_ID_LOOKUP.get(el, 0))
        ]
        # NOTE may need to pad text length to a fixed size?
        if sum(anno["text"]) == 0:
            # If there is no text, do not include the annotation
            continue

        # Combine annotation with annotation constants and add to annotations
        annotations.append(dict((*anno.items(), *_ANNOTATION_CONSTANTS)))
    
    return annotations
