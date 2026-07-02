"""
Edina module contains functions and classes used to extract and parse
the geodata from tiff files downloaded from EDINA.
"""
from ._parse_tab import parse_edina_tab_file
from ._geo_transforms import\
    get_transformer_from_geodataframe, get_png_overlaps
from ._tiffs_to_pngs import EDINATiffPNGConverter