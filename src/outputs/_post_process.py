"""
Collection of functions to post-process ToponymExtractor predictions.
"""
from typing import Any
from pathlib import Path
from shapely import Polygon, Point
from numpy import ndarray, hstack, floor as np_floor, ceil as np_ceil
from PIL.Image import Image, fromarray as image_fromarray
from rasterio import open as open_raster
from rasterio.transform import AffineTransformer
from pandas import concat, Series
from geopandas import GeoDataFrame, sjoin
from edina import get_png_overlaps

# Functions
def _get_intersecting_png_masks(
    gdf1: GeoDataFrame, gdf2: GeoDataFrame
) -> GeoDataFrame:
    """
    Get intersecting masks from the predictions for different PNGs.

    Parameters
    ----------
    gdf1, gdf2: GeoDataFrame, GeoDataFrame.
        Required. Contains the text predictions with polygons for their
        masks. Each GeoDataFrame must include the field: "png_filename".
    
    Returns
    -------
    GeoDataFrame. Each record represents a pair of predictions from
    different PNGs whose masks intersect. The returned geometry field
    contains the gdf1 geometry. Includes the fields:

    - png_filename_left, png_filename_right: str, str. These are the
    filenames for the images the intersecting polygons were derived
    from.
    - iou: float. Intersection over union between the polygon pairs.
    """
    overlapping = sjoin(gdf1, gdf2, how = "inner", predicate = "intersects")
    # Do not include overlapping predictions from the same png - leveraging
    # bipartite matching used within DeepSolo that minimized the likelihood
    # for duplicate predictions
    overlapping = overlapping\
        .loc[overlapping.png_filename_left != overlapping.png_filename_right]

    # Calculate intersection over union for the intersecting predictions
    intersection_area = gdf2.loc[overlapping.index_right, "geometry"]
    intersection_area = overlapping\
        .geometry.intersection(intersection_area, align = 0).area
    
    union_area = gdf2.loc[overlapping.index_right, "geometry"]
    union_area = overlapping.geometry.union(union_area, align = 0).area
    
    overlapping["iou"] = intersection_area / union_area

    return overlapping


def get_intersecting_png_masks(
    gdf1: GeoDataFrame, gdf2: GeoDataFrame
) -> GeoDataFrame:
    """
    Get intersecting masks from the predictions for different PNGs.

    Parameters
    ----------
    gdf1, gdf2: GeoDataFrame, GeoDataFrame.
        Required. Contains the text predictions with polygons for their
        masks. Each GeoDataFrame must include the field: "png_filename".
    
    Returns
    -------
    GeoDataFrame. Each record represents a pair of predictions from
    different PNGs whose masks intersect. The returned geometry field
    contains the gdf1 geometry. Includes the fields:

    - png_filename_left, png_filename_right: str, str. These are the
    filenames for the images the intersecting polygons were derived
    from.
    - iou: float. Intersection over union between the polygon pairs.
    """
    if "png_filename" not in gdf1.columns:
        raise ValueError(
            "The 'png_filename' field is missing from gdf1 GeoDataFrame"
        )
    if "png_filename" not in gdf2.columns:
        raise ValueError(
            "The 'png_filename' field is missing from gdf2 GeoDataFrame"
        )
    return _get_intersecting_png_masks(gdf1 = gdf1, gdf2 = gdf2)


def get_intersecting_polygon_pairs(predictions: GeoDataFrame) -> GeoDataFrame:
    """
    Used in non-maximal suppression.

    Parameters
    ----------
    predictions: GeoDataFrame.
        Required. Contains the text predictions with polygons for their
        masks. The GeoDataFrame must include the fields: "png_filename",
        "png_overlap", "word", "score", and "geometry".

    Returns
    -------
    GeoDataFrame. Contains a record for each pair of intersecting
    prediction masks. The "geometry" field contains the polygon mask for
    the "left" fields, these fields are:

    - png_filename_left, png_filename_right: str, str. These are the
    filenames for the images the intersecting polygons were derived
    from.
    - png_overlap_left, png_overlap_right: str, str. These fields
    describe the relationship between predicted polygons and the
    overlapping space of the PNGs. Can be one of "disjoint",
    "intersect", or "subset".
    - word_left, word_right: str, str. Predicted words for the
    overlapping polygons.
    - score_left, score_right: float, float. Confidence scores for the
    overlapping predictions.
    - iou: float. Intersection over union between the polygon pairs.
    """
    # Select specific columns
    overlapping = predictions[[
        "png_filename", "png_overlap", "word", "score", "geometry"
    ]]
    # Join GeoDataFrame with itself based on intersecting predictions
    overlapping = sjoin(
        overlapping, overlapping, how = "inner", predicate = "intersects"
    )
    # Do not include overlapping predictions from the same png - leveraging
    # bipartite matching used within DeepSolo that minimized the likelihood
    # for duplicate predictions
    overlapping = overlapping\
        .loc[overlapping.png_filename_left != overlapping.png_filename_right]
    # Do not include duplicate prediction pairs
    overlapping = overlapping.loc[overlapping.index < overlapping.index_right]

    # Calculate intersection over union for the intersecting predictions
    intersection_area = predictions.loc[overlapping.index_right, "geometry"]
    intersection_area = overlapping\
        .geometry.intersection(intersection_area, align = 0).area
    
    union_area = predictions.loc[overlapping.index_right, "geometry"]
    union_area = overlapping.geometry.union(union_area, align = 0).area
    
    overlapping["iou"] = intersection_area / union_area

    return overlapping


class ProcessToponymExtractorPredictions:
    def __init__(
        self,
        ctrl_points: GeoDataFrame,
        tiff_dir: Path
    ):
        self._tiff_height: int
        self._tiff_width: int
        self._tiff_transformer: AffineTransformer
        self._tiff_crs: Any
        self._tiff_data: ndarray
        self._undetermined: dict[str, list[Image|tuple[int,int]]|GeoDataFrame]
        self._clique_count: int

        # Get mask of overlapping space for each TIFF
        png_overlaps = get_png_overlaps(ctrl_points = ctrl_points)
        self.tif_overlaps = png_overlaps[["tiff_filename", "geometry"]]\
            .dissolve("tiff_filename", as_index = False)
        self.tif_overlaps["geometry"] = self.tif_overlaps.geometry.buffer(0)
        # store tiff directory
        self.tiff_directory: Path = tiff_dir

    def _get_tiff_details(self, tiff_fn: str) -> None:
        tiff_fp = self.tiff_directory.joinpath(tiff_fn)
        with open_raster(tiff_fp, mode = "r") as src:
            self._tiff_height = src.height # image height
            self._tiff_width = src.width # image width
            self._tiff_transformer = AffineTransformer(src.transform)
            self._tiff_crs = src.read_crs() # coordinate reference system
            self._tiff_data = (-src.read() + 1) * 255 # image array


    def _add_png_overlap_column(self) -> None:
        """
        Determine the relationship between the predictions and the
        overlapping spaces of the TIFF file.

        Adds column "png_overlap" to the current predictions
        GeoDataFrame.
        """
        # default to all predictions belonging to a single png
        self._current_predictions["png_overlap"] = "disjoint"

        # update predictions that intersect with png overlap area
        selection = self._current_predictions.geometry\
            .intersects(self._current_tiff_overlap)
        self._current_predictions.loc[selection, "png_overlap"] = "intersect"

        # update predictions that are contained entirely within png overlap
        # area
        selection = self._current_predictions.geometry\
            .within(self._current_tiff_overlap)
        self._current_predictions.loc[selection, "png_overlap"] = "subset"


    def _process_indeterminant_predictions(self, gdf: GeoDataFrame) -> None:
        # Get all instances of overlapping predictions
        selection = gdf.index.union(gdf.index_right).unique().sort_values()
        pred_words = self._current_predictions.loc[selection]

        # Retrieve the word groups the overlapping predictions belong to
        pred_words["key"] =\
            pred_words.png_filename + pred_words.groupid.astype("string")
        selection = (
            self._all_predictions.png_filename
            + self._all_predictions.groupid.astype("string")
        )
        selection = selection.isin(set(pred_words.key))
        word_groups = self._all_predictions[selection].copy()

        # Group together the polygons for each word into their
        # respective groups
        word_groups["key"] =\
            word_groups.png_filename + word_groups.groupid.astype("string")

        bounds = word_groups[["key", "png_filename", "groupid", "geometry"]]\
            .dissolve(["key", "png_filename", "groupid"], as_index = True)\
            .convex_hull\
            .reset_index(drop = False, name = "geometry")

        # Get key fields for left and right overlaps
        gdf_key_l = gdf.png_filename_left + gdf.groupid_left.astype("string")
        gdf_key_r = gdf.png_filename_right + gdf.groupid_right.astype("string")

        # Collect overlapping word-group polygons
        connected_components: list[set] = []
        vertex_iter = zip(gdf_key_l.tolist(), gdf_key_r.tolist())
        try:
            connected_components.append(set(next(vertex_iter)))
            while 1:
                v = set(next(vertex_iter))
                clique = next(
                    (e for e in connected_components if not v.isdisjoint(e)),
                    set()
                )
                if clique:
                    clique.update(v)
                else:
                    connected_components.append(v)
        except StopIteration as _:
            pass
        except Exception as e:
            raise

        # Combine grouped word geometries from predictions and retrieve
        # metadata
        grouped_polygons = []
        for idx, clique in enumerate(connected_components, self._clique_count):
            grouped_polygons.append(dict())
            grouped_polygons[-1]["clique_idx"] = idx
            grouped_polygons[-1]["geometry"] = bounds\
                .loc[bounds.key.isin(clique)]\
                .geometry\
                .union_all()\
                .convex_hull\
                .buffer(5)
            
            grouped_polygons[-1]["words"] = word_groups\
                .loc[word_groups.key.isin(clique)]\
                .copy()
            grouped_polygons[-1]["words"]["clique_idx"] = idx
        # update clique count
        self._clique_count += len(connected_components)

        grouped_polygons_meta = concat(
            [el.pop("words") for el in grouped_polygons],
            axis = 0,
            ignore_index = False
        )
        grouped_polygons = GeoDataFrame(
            grouped_polygons, crs = grouped_polygons_meta.crs
        )

        # Get bounding box points for each pair_bounds using tiff CRS
        bboxes = grouped_polygons[["clique_idx"]].copy()
        bboxes[["minx", "miny", "maxx", "maxy"]] =\
            grouped_polygons.geometry.to_crs(self._tiff_crs).bounds

        # Convert coordinates to PNG row-column indices
        temp = self._tiff_transformer\
            .rowcol(bboxes["minx"], bboxes["maxy"], op = np_floor)
        bboxes[["min_row", "min_col"]] =\
            hstack([el.reshape((-1, 1)) for el in temp])
        
        temp = self._tiff_transformer\
            .rowcol(bboxes["maxx"], bboxes["miny"], op = np_ceil)
        bboxes[["max_row", "max_col"]] =\
            hstack([el.reshape((-1, 1)) for el in temp])
        
        # clip row-col index values to fall within the PNG dimensions
        bboxes[["min_row", "max_row"]] =\
            bboxes[["min_row", "max_row"]].clip(0, self._tiff_height - 1)
        bboxes[["min_col", "max_col"]] =\
            bboxes[["min_col", "max_col"]].clip(0, self._tiff_width - 1)
        bboxes[["min_row", "min_col", "max_row", "max_col"]] =\
            bboxes[["min_row", "min_col", "max_row", "max_col"]]\
            .astype("int64")
        
        # Get image and metadata for each image segment to be passed back
        # to the model
        undetermined = {"image": [], "image_hw": [], "control_points": []}
        for tup in bboxes.itertuples(index = False):
            # get image snippet
            temp = self._tiff_data[
                0, tup.min_row: tup.max_row + 1, tup.min_col: tup.max_col + 1
            ]
            undetermined["image"].append(
                image_fromarray(obj = temp, mode =  "L")
            )
            undetermined["image_hw"].append(temp.shape)

            # create georeferencing control points for image
            undetermined["control_points"].append({
                "clique_idx": tup.clique_idx,
                "pixel_x": tup.min_col,
                "pixel_y": tup.min_row,
                "geometry": Point(tup.minx, tup.maxy)
            })
            undetermined["control_points"].append({
                "clique_idx": tup.clique_idx,
                "pixel_x": tup.min_col,
                "pixel_y": tup.max_row,
                "geometry": Point(tup.minx, tup.miny)
            })
            undetermined["control_points"].append({
                "clique_idx": tup.clique_idx,
                "pixel_x": tup.max_col,
                "pixel_y": tup.min_row,
                "geometry": Point(tup.maxx, tup.maxy)
            })
            undetermined["control_points"].append({
                "clique_idx": tup.clique_idx,
                "pixel_x": tup.max_col,
                "pixel_y": tup.max_row,
                "geometry": Point(tup.maxx, tup.miny)
            })

        undetermined["control_points"] =\
            GeoDataFrame(undetermined["control_points"], crs = self._tiff_crs)\
            .to_crs(grouped_polygons_meta.crs)
        undetermined["word_groups"] = grouped_polygons_meta.copy()

        # update _undetermined attributes
        self._undetermined["image"].extend(undetermined["image"])
        self._undetermined["image_hw"].extend(undetermined["image_hw"])
        self._undetermined["control_points"] = concat(
            [
                self._undetermined["control_points"],
                undetermined["control_points"]
            ],
            axis = 0
        )
        self._undetermined["word_groups"] = concat(
            [self._undetermined["word_groups"], undetermined["word_groups"]],
            axis = 0
        )
        # Update _current_predictions and _suppressed_predictions attributes
        selection =\
            self._current_predictions.index.intersection(word_groups.index)
        self._suppressed_predictions = concat(
            [
                self._suppressed_predictions,
                self._current_predictions.loc[selection]
            ],
            axis = 0
        )
        self._current_predictions = self._current_predictions\
            .loc[self._current_predictions.index.difference(word_groups.index)]


    def _process_intercept_intercept_predictions(
        self, gdf: GeoDataFrame
    ) -> None:
        """
        Edits `_current_predictions` GeoDataFrame attribute. Creates
        `_suppressed_predictions` GeoDataFrame attribute and an
        `_undetermined` dictionary.
        """
        # Keep predctions with IoU scores less than or equal to .1
        gdf = gdf[gdf.iou > .1]
        # Apply non-maximal supression for masks with the same text
        # labels
        selection = ((gdf.iou >= .8) & (gdf.word_left == gdf.word_right))
        suppress_idxs = gdf.loc[selection, "index_right"].tolist()

        # Log predictions that are removed and update _current_predictions
        self._suppressed_predictions =\
            self._current_predictions.loc[suppress_idxs].copy()
        self._current_predictions = self._current_predictions\
            .loc[self._current_predictions.index.difference(suppress_idxs)]
        
        # Update gdf
        gdf = gdf.loc[~selection]
        gdf = gdf.loc[gdf.index.difference(suppress_idxs)]

        # Get indeterminate predictions
        self._process_indeterminant_predictions(gdf = gdf)
        

    def _process_subset_subset_predictions(self, gdf: GeoDataFrame) -> None:
        """
        Edits the `_current_predictions` GeoDataFrame attribute, and the
        `_suppressed_predictions` GeoDataFrame attribute.
        """
        # Apply non-maximal supression for IoU greater than .8
        selection = gdf.loc[(gdf.iou >= .8), "index_right"].tolist()
        self._suppressed_predictions = concat(
            [
                self._suppressed_predictions,
                self._current_predictions.loc[selection]
            ],
            axis = 0
        )
        self._current_predictions = self._current_predictions\
            .loc[self._current_predictions.index.difference(selection)]
        

    def _process_intercept_subset_predictions(self, gdf: GeoDataFrame) -> None:
        """
        Edits the `_current_predictions` GeoDataFrame attribute, the
        `_suppressed_predictions` GeoDataFrame attribute, and the
        `_undetermined` dictionary attribute.
        """
        # Suppress "subset" predictions when words are the same
        selection = (gdf.word_left == gdf.word_right)
        suppression_idxs = gdf.loc[selection, "index_right"].to_list()
        self._suppressed_predictions = concat(
            [
                self._suppressed_predictions,
                self._current_predictions.loc[suppression_idxs]
            ],
            axis = 0
        )
        self._current_predictions = self._current_predictions\
            .loc[self._current_predictions.index.difference(suppression_idxs)]
        # update gdf
        gdf = gdf.loc[~selection]
        # Keep predictions where IoU is less than .1
        gdf = gdf.loc[gdf.iou >= .1]

        # Suppress "subset" strings contained within the "intersect" prediction
        selection = [
            (
                tup.word_left.startswith(tup.word_right)
                or tup.word_left.endswith(tup.word_right)
            )
            for tup in gdf.itertuples()
        ]
        selection = Series(selection, index = gdf.index, dtype = "bool")
        suppression_idxs = gdf.loc[selection, "index_right"].to_list()
        self._suppressed_predictions = concat(
            [
                self._suppressed_predictions,
                self._current_predictions.loc[suppression_idxs]
            ],
            axis = 0
        )
        self._current_predictions = self._current_predictions\
            .loc[self._current_predictions.index.difference(suppression_idxs)]
        # Update gdf
        gdf = gdf.loc[~selection]

        # Get indeterminate predictions
        self._process_indeterminant_predictions(gdf = gdf)


    def process_predictions(
        self, tiff_fn: str, predictions: GeoDataFrame
    ):
        # Get tiff image details
        self._get_tiff_details(tiff_fn = tiff_fn)

        # initialize undetermined attribute
        self._undetermined = {
            "image": [],
            "image_hw": [],
            "control_points": GeoDataFrame(),
            "word_groups": GeoDataFrame()
        }
        # initialize clique count
        self._clique_count = 0

        self._current_tiff_overlap: Polygon = self.tif_overlaps\
            .loc[self.tif_overlaps["tiff_filename"] == tiff_fn, "geometry"]\
            .iloc[0]
        
        # Sort predictions by confidence score - descending
        self._current_predictions: GeoDataFrame = predictions\
            .sort_values("score", ascending = False, ignore_index = True)
        # Keep a copy of all predictions to track word groups
        self._all_predictions = self._current_predictions.copy()
        self._add_png_overlap_column()

        # Process intersect - intersect predictions
        selection = (self._current_predictions["png_overlap"] == "intersect")
        intersect_intersect_predictions = get_intersecting_png_masks(
            self._current_predictions[selection],
            self._current_predictions[selection]
        )
        intersect_intersect_predictions = intersect_intersect_predictions[(
            intersect_intersect_predictions.index
            < intersect_intersect_predictions.index_right
        )]
        self._process_intercept_intercept_predictions(
            intersect_intersect_predictions
        )

        # Process subset - subset predictions
        selection = (self._current_predictions["png_overlap"] == "subset")
        subset_subset_predictions = get_intersecting_png_masks(
            self._current_predictions[selection],
            self._current_predictions[selection]
        )
        subset_subset_predictions = subset_subset_predictions[(
            subset_subset_predictions.index
            < subset_subset_predictions.index_right
        )]
        self._process_subset_subset_predictions(subset_subset_predictions)

        # process intersect - subset predictions
        intersect_subset_predictions = get_intersecting_png_masks(
            self._current_predictions, self._current_predictions
        )
        intersect_subset_predictions = intersect_subset_predictions[(
            (intersect_subset_predictions.png_overlap_left == "intersect")
            & (intersect_subset_predictions.png_overlap_right == "subset")
        )]
        self._process_intercept_subset_predictions(
            intersect_subset_predictions
        )
        return (
            self._current_predictions,
            self._suppressed_predictions,
            self._undetermined
        )
