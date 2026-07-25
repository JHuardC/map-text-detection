"""
Funtions for manipulating polygon word labels
"""
from pandas import DataFrame, merge
from geopandas import GeoDataFrame
from ._parse_toponym_outputs import normalize_geometries

def aggregate_words_to_toponym(df_group: DataFrame) -> str:
    """
    Function to be used with `DataFrameGroupBy.apply`.

    Combines words in the order of the `wordid` column to create a
    toponym string.

    Parameter
    ---------
    df_group: DataFrame.
        Required. Collection of records with matching groupid values.
        Must contain the fields "wordid", and "word".
    
    Returns
    -------
    str. Toponym string; words are concatenated with spaces as
    delimeters.
    """
    toponym = df_group\
        .sort_values(by = "wordid", ascending = True)\
        .word\
        .to_list()
    toponym = " ".join(toponym)

    return toponym


def build_toponym_text(df: DataFrame) -> DataFrame:
    """
    Creates a Dataframe of toponyms.

    Parameter
    ---------
    df: DataFrame.
        Required. Records of words to be grouped into toponyms. Must
        include the fields "groupid", "wordid" and "word".
    
    Returns
    -------
    DataFrame. Contains 2-columns:
        - groupid: Containing the ID values for each group of words that
        make up the toponymns
        - toponym: str. Combined words, grouped by "groupid" and ordered
        by "wordid".
    """
    return df\
        .groupby(by = "groupid", as_index = False)\
        .apply(aggregate_words_to_toponym)\
        .rename(columns = {None: "toponym"})


def build_toponym_gdf(gdf: GeoDataFrame) -> GeoDataFrame:
    """
    Combine word labels into toponyms with combined polygon masks.

    Parameters
    ----------
    gdf: DataFrame.
        Required. Records of words to be grouped into toponyms. Must
        include the fields "groupid", "wordid", "word", and "geometry".
    
    Returns
    -------
    DataFrame. Contains 3-columns:
        - groupid: Containing the ID values for each group of words that
        make up the toponymns
        - toponym: str. Combined words, grouped by "groupid" and ordered
        by "wordid".
        - geometry: Polygons. Combined polygon mask covering all the
        word polygons.
    """
    cols = ["groupid", "geometry"]
    toponyms = gdf[cols].dissolve(by = ["groupid"], as_index = False)
    toponyms["geometry"] = normalize_geometries(toponyms.geometry)

    toponyms = merge(
        toponyms, build_toponym_text(gdf), how = "inner", on = "groupid"
    )
    return toponyms[["groupid", "toponym", "geometry"]]
