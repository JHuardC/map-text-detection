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
    # Check all pixel coordinate values are unique
    if gdf.duplicated(["pixel_x", "pixel_y"]).any():
        raise ValueError(
            "GeoDataFrame contains duplicate control points. Each record "\
            "in the GeoDataFrame should contain unique 'pixel_x', pixel_y "\
            "pairs."
        )
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
    return GCPTransformer(gcps = gcps, tps = True)
