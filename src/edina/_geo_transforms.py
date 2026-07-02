from geopandas import GeoDataFrame, sjoin
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
    return GCPTransformer(gcps = gcps, tps = False)


def get_png_overlaps(ctrl_points: GeoDataFrame) -> GeoDataFrame:
    """
    Combine PNG boundary shapes to retrieve the overlapping areas for
    each PNG.

    Parameter
    ---------
    ctrl_points: GeoDataFrame.
        Required. GeoDataFrame containing the details of the control
        points used for georeferencing. GeoDataFrame requires fields:
        "geometry", "pixel_x", "pixel_y", "png_filename", and
        "tiff_filename".
    
    Returns
    -------
    GeoDataFrame. Each record contains a polygon that represents the
    overlap between each PNG. Includes the fields:
    - tiff_filename: str. Filename for the TIFF that the PNGs were
    derived from.
    - png_filename_left, png_filename_right: str, str. Filenames for the
    PNGs that the overlapping polygon geometry was derived from.
    """
    # Group control points by tiff_filename and png_filename
    png_bounds = ctrl_points[["tiff_filename", "png_filename", "geometry"]]\
        .dissolve(by = ["tiff_filename","png_filename"], as_index = False)
    # Convert multipoint object to a polygon - polygon represents the
    # area covered by the png
    png_bounds["geometry"] = png_bounds.geometry.convex_hull

    # Find which PNG areas overlap by joining PNGs with intersecting
    # geometries
    overlaps = png_bounds[["tiff_filename", "png_filename", "geometry"]]
    overlaps = sjoin(
        overlaps,
        overlaps[["png_filename", "geometry"]],
        how = "inner",
        predicate = "intersects"
    )
    # Remove duplicate intersection pairs
    overlaps = overlaps[overlaps.index < overlaps.index_right]
    # Get overlapping space for each PNG pair
    overlaps[ "geometry"] = overlaps[ "geometry"].intersection(
        png_bounds.loc[overlaps.index_right, "geometry"], align = False
    )
    # Select inmportant fields
    overlaps = overlaps[[
        "tiff_filename", "png_filename_left", "png_filename_right", "geometry"
    ]]
    return overlaps
