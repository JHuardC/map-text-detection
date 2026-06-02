from geopandas import GeoDataFrame
from rasterio.control import GroundControlPoint
from rasterio.transform import GCPTransformer

def get_transformer_from_geodataframe(gdf: GeoDataFrame) -> GCPTransformer:
    """
    Build Tranformer for georeferencing edina .tif files using
    Ground Control Points extracted for .tab file.

    Parameters
    ----------
    gdf: GeoDataFrame.
        Required. GeoDataFrame containing the details of the control
        points used for georeferencing. GeoDataFrame requires fields:
        "geometry", "pixel_x", "pixel_y".

    Returns
    -------
        GCPTransformer. Control point transformer. Convert pixel
        row-column values to coordinates using .xy() method; convert
        coordinates to pixel row-column values using .rowcol() method.
    """
    # Generate collection of ground control points
    gcps: tuple[GroundControlPoint] = tuple(
        GroundControlPoint(
            row = tup.pixel_y,
            col = tup.pixel_x,
            x = tup.geometry.x,
            y = tup.geometry.y
        )
        for tup in gdf.itertuples(index = False)
    )
    # Create GCPTransformer instance
    transformer = GCPTransformer(gcps = gcps, tps = True)
    return transformer