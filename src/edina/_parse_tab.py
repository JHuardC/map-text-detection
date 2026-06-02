from typing import Final
from pathlib import Path
from re import compile, Pattern
from shapely import from_wkt
from geopandas import GeoDataFrame

_TBL_REGEX: Final[Pattern]
_NEWLINE_REGEX: Final[Pattern]
_STR_REGEX: Final[Pattern]
_DIGIT_REGEX: Final[Pattern]
_TAB_KEYS: Final[tuple]

_TBL_REGEX = compile(r"(?s)(?<=\"RASTER\"\n  ).+?(?=\n  CoordSys)")
_NEWLINE_REGEX = compile(r",?s*?\n\s*")
_STR_REGEX = compile(r"(?s)(\".*?\")")
_DIGIT_REGEX = compile(r"\d+")
_TAB_KEYS = ("geometry", "pixels", 0, 1)

def parse_edina_tab_file(fp: Path, crs: str = "EPSG:27700") -> GeoDataFrame:
    """
    Read and parse .tab file containing the control points used for
    georeferencing tiffs downloaded from edina.

    Parameters
    ----------
    fp: Path.
        Required. Path instance to .tab file containing the metadata
        required for georeferencing the corresponding tiff images.
    
    crs: str. Default: "EPSG:27700".
        Optional. Coordinate reference system the control points within
        the .tab file uses.

    Returns
    -------
    GeoDataFrame. Each record contains the details of a control point
    used for georeferencing. Fields:
        - "geometry": shapely.Point. Contains the coordinates for the
        control points encoded under the crs provided.
        - "pixel_x": int. Column value for the pixel corresponding to
        the coordinate stored in geometry.
        - "pixel_y": int. Row value for the pixel corresponding to the
        coordinate stored in geometry.
        - "Label": string. Label provided with the control point.
    """
    with open(fp, "r") as tab_file:
        data = tab_file.read()
    
    # extract table data
    data = _TBL_REGEX.search(data).group()
    # Split into records
    data = _NEWLINE_REGEX.split(data)
    # Split records into separate fields
    data = [_STR_REGEX.split(record) for record in data]
    data = [
        [
            v
            for string in record
            for v in ([string[1: -1]] if "\"" in string else string.split(" "))
            if len(v)
        ]
        for record in data
    ]
    # Convert records to dictionaries
    data = [dict(zip(_TAB_KEYS, record)) for record in data]

    for record in data:
        # generate "Label" key
        record[record.pop(0)] = record.pop(1)

        # convert geomtry  to an actual geometric data type
        record["geometry"] =\
            from_wkt(f"POINT {record["geometry"].replace(",", " ")}")
        
        # separate pixel values into x, y
        px = record.pop("pixels")
        px = tuple(int(el) for el in _DIGIT_REGEX.findall(px))
        record["pixel_x"], record["pixel_y"] = px
        
        # Add filename
        record["filename"] = fp.stem

    data = GeoDataFrame.from_records(data).set_crs(crs = crs)
    return data