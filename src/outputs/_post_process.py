"""
Collection of functions to post-process ToponymExtractor predictions.
"""
from typing import Any
from pathlib import Path
from shapely import Polygon, Point
from numpy import ndarray, hstack, floor as np_floor, ceil as np_ceil
# from PIL.Image import Image, fromarray as image_fromarray, new as new_image
from rasterio import open as open_raster
from rasterio.transform import AffineTransformer
from pandas import concat, Series
from geopandas import GeoDataFrame, GeoSeries, sjoin
from edina import get_png_overlaps

# Functions
def _get_intersecting_png_masks(
    gdf1: GeoDataFrame, gdf2: GeoDataFrame
) -> tuple[GeoDataFrame, GeoSeries]:
    """
    Get intersecting masks from the predictions for different PNGs.

    Parameters
    ----------
    gdf1, gdf2: GeoDataFrame, GeoDataFrame.
        Required. Contains the text predictions with polygons for their
        masks. Each GeoDataFrame must include the field: "png_filename",
        and "groupid".
    
    Returns
    -------
    tuple: (GeoDataFrame, GeoSeries).
    
    GeoDataFrame:
    
    Each record represents a pair of predictions from different PNGs
    whose masks intersect. The returned geometry field contains the gdf1
    geometry. Includes the fields:

    - png_filename_left, png_filename_right: str, str. These are the
    filenames for the images the intersecting polygons were derived
    from.
    - groupid_left, groupid_right: int, int. These are the group indices
    for each png representing the toponym the word instance was assigned
    to.
    - iou: float. Intersection over union between the polygon pairs.

    Geoseries:

    Each record is aligned to the records in the returned GeoDataFrame
    and contains the intersecting area between `gdf1` and `gdf2`
    joint geometries.
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

    return overlapping, intersection_area


def _check_intersecting_png_masks_gdfs(
    gdf1: GeoDataFrame, gdf2: GeoDataFrame
) -> None:
    """
    Checks `gdf1` and `gdf2` contain "png_filename" and "groupid"
    columns.

    Returns
    -------
    None.

    Raises
    ------
    ValueErrors if desired column names do not exist.
    """
    if "png_filename" not in gdf1.columns:
        raise ValueError(
            "The 'png_filename' field is missing from gdf1 GeoDataFrame"
        )
    if "png_filename" not in gdf2.columns:
        raise ValueError(
            "The 'png_filename' field is missing from gdf2 GeoDataFrame"
        )
    if "groupid" not in gdf1.columns:
        raise ValueError(
            "The 'groupid' field is missing from gdf1 GeoDataFrame"
        )
    if "groupid" not in gdf2.columns:
        raise ValueError(
            "The 'groupid' field is missing from gdf2 GeoDataFrame"
        )
    return None


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
    _check_intersecting_png_masks_gdfs(gdf1 = gdf1, gdf2 = gdf2)
    output, _ = _get_intersecting_png_masks(gdf1 = gdf1, gdf2 = gdf2)
    return output


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


class ToponymExtractorProcessor:

    @property
    def current_tiff_overlap(self) -> Polygon:
        """
        Polygon representing the geographic areas overlapped by PNGs
        for a specific TIFF file.
        """
        try:
            return self._current_tiff_overlap
        except AttributeError as _:
            msg = "Attribute current_tiff_overlap has not been assigned yet. "\
                "Attribute current_tiff_overlap is assigned when "\
                "process_predictions() is called."
            raise AttributeError(msg) from None
        except Exception as e:
            raise
    @current_tiff_overlap.setter
    def current_tiff_overlap(self, v) -> None:
        msg = "Cannot directly set current_tiff_overlap attribute"
        raise AttributeError(msg)
    @current_tiff_overlap.deleter
    def current_tiff_overlap(self) -> None:
        msg = "Cannot directly delete current_tiff_overlap attribute"
        raise AttributeError(msg)

    @property
    def buffer(self) -> int:
        """
        Number of pixels to buffer ambiguous images by.
        """
        return self._buffer
    @buffer.setter
    def buffer(self, v: int) -> None:
        msg = "Cannot directly set buffer attribute; can only be set on "\
            "initialisation."
        raise AttributeError(msg)
    @buffer.deleter
    def buffer(self) -> None:
        msg = "Cannot directly delete buffer attribute"
        raise AttributeError(msg)


    def __init__(
        self, ctrl_points: GeoDataFrame, tiff_dir: Path, buffer: int = 10
    ):
        """
        Post-process ToponymExtractor outputs

        Parameters
        ----------
        ctrl_points: GeoDataFrame.
            Required. Contains the georeferencing control points for
            each image passed to the ToponymExtractor model.
        tiff_dir: Path.
            Required. Directory path to the Edina downloaded tiff files.
        buffer: int. Default: 10.
            Optional. Number of pixels to buffer ambiguous images by.
        """
        self._tiff_height: int
        self._tiff_width: int
        self._tiff_transformer: AffineTransformer
        self._tiff_crs: Any
        self._tiff_data: ndarray
        self._undetermined: dict[str, list[ndarray] | GeoDataFrame]
        self._clique_count: int
        # self.img_h: int
        # self.img_w: int

        # Get mask of overlapping space for each TIFF
        png_overlaps = get_png_overlaps(ctrl_points = ctrl_points)
        self.tif_overlaps = png_overlaps[["tiff_filename", "geometry"]]\
            .dissolve("tiff_filename", as_index = False)
        self.tif_overlaps["geometry"] = self.tif_overlaps.geometry.buffer(0)
        # store tiff directory
        if not tiff_dir.is_dir():
            raise ValueError(
                f"Argument passed to tiff_dir is not a directory: {tiff_dir}"
            )
        self.tiff_directory: Path = tiff_dir
        self._buffer = buffer
        # store standardized image sizes
        # self.img_h, self.img_w = img_h, img_w

    def _get_tiff_details(self, tiff_fn: str) -> None:
        tiff_fp = self.tiff_directory.joinpath(tiff_fn)
        with open_raster(tiff_fp, mode = "r") as src:
            self._tiff_height = src.height # image height
            self._tiff_width = src.width # image width
            self._tiff_transformer = AffineTransformer(src.transform)
            self._tiff_crs = src.read_crs() # coordinate reference system
            # self._tiff_data = (-src.read() + 1) * 255 # image array
            self._tiff_data = src.read() # binary array


    def _process_indeterminant_predictions(
        self, gdf: GeoDataFrame, current_predictions: GeoDataFrame
    ) -> GeoDataFrame:
        """
        Edits `current_predictions` GeoDataFrame,
        `_suppressed_predictions` GeoDataFrame attribute, and the 
        `_undetermined` dictionary attribute.

        Returns
        -------
        `current_predictions` GeoDataFrame with suppressed indeterminate
        predictions removed.
        """
        if len(gdf) == 0:
            # Nothing to process
            return current_predictions
        # Get all instances of overlapping predictions
        selection = gdf.index.union(gdf.index_right).unique().sort_values()
        pred_words = current_predictions.loc[selection]

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
                .convex_hull
            
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

        # Convert coordinates to PNG row-column indices and include buffer
        temp = self._tiff_transformer\
            .rowcol(bboxes["minx"], bboxes["maxy"], op = np_floor)
        bboxes[["min_row", "min_col"]] =\
            hstack([el.reshape((-1, 1)) for el in temp]) - self.buffer
        
        temp = self._tiff_transformer\
            .rowcol(bboxes["maxx"], bboxes["miny"], op = np_ceil)
        bboxes[["max_row", "max_col"]] =\
            hstack([el.reshape((-1, 1)) for el in temp]) + self.buffer
        
        # clip row-col index values to fall within the TIFF dimensions
        bboxes[["min_row", "max_row"]] =\
            bboxes[["min_row", "max_row"]].clip(0, self._tiff_height - 1)
        bboxes[["min_col", "max_col"]] =\
            bboxes[["min_col", "max_col"]].clip(0, self._tiff_width - 1)
        bboxes[["min_row", "min_col", "max_row", "max_col"]] =\
            bboxes[["min_row", "min_col", "max_row", "max_col"]]\
            .astype("int64")
        
        # Get image and metadata for each image segment to be passed back
        # to the model
        undetermined = {"image": [], "control_points": []}
        for tup in bboxes.itertuples(index = False):
            # get image snippet
            temp = self._tiff_data[
                0, tup.min_row: tup.max_row + 1, tup.min_col: tup.max_col + 1
            ].copy()
            undetermined["image"].append(temp)

            # create georeferencing control points for image
            undetermined["control_points"].append({
                "clique_idx": tup.clique_idx,
                "pixel_x": 0,
                "pixel_y": 0,
                "geometry": Point(tup.minx, tup.maxy)
            })
            undetermined["control_points"].append({
                "clique_idx": tup.clique_idx,
                "pixel_x": 0,
                "pixel_y": temp.shape[0] - 1,
                "geometry": Point(tup.minx, tup.miny)
            })
            undetermined["control_points"].append({
                "clique_idx": tup.clique_idx,
                "pixel_x": temp.shape[1] - 1,
                "pixel_y": 0,
                "geometry": Point(tup.maxx, tup.maxy)
            })
            undetermined["control_points"].append({
                "clique_idx": tup.clique_idx,
                "pixel_x": temp.shape[1] - 1,
                "pixel_y": temp.shape[0] - 1,
                "geometry": Point(tup.maxx, tup.miny)
            })

        undetermined["control_points"] =\
            GeoDataFrame(undetermined["control_points"], crs = self._tiff_crs)\
            .to_crs(grouped_polygons_meta.crs)
        undetermined["word_groups"] = grouped_polygons_meta.copy()

        # update _undetermined attributes
        self._undetermined["image"].extend(undetermined["image"])
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
            current_predictions.index.intersection(word_groups.index)
        self._suppressed_predictions = concat(
            [
                self._suppressed_predictions,
                current_predictions.loc[selection]
            ],
            axis = 0
        )
        current_predictions = current_predictions\
            .loc[current_predictions.index.difference(word_groups.index)]
        
        return current_predictions


    def process_predictions(
        self, tiff_fn: str, predictions: GeoDataFrame
    ) -> GeoDataFrame:
        # Get tiff image details
        self._get_tiff_details(tiff_fn = tiff_fn)

        # initialised suppressed predictions attribute
        self._suppressed_predictions = GeoDataFrame()

        # initialize undetermined predictions attribute
        self._undetermined = {
            "image": [],
            "control_points": GeoDataFrame(),
            "word_groups": GeoDataFrame()
        }
        # initialize clique count
        self._clique_count = 0

        self._current_tiff_overlap: Polygon = self.tif_overlaps\
            .loc[self.tif_overlaps["tiff_filename"] == tiff_fn, "geometry"]\
            .iloc[0]
        
        # Initialize current predictions log (retained predictions)
        # Sort predictions by confidence score - descending
        predictions: GeoDataFrame = predictions\
            .sort_values("score", ascending = False, ignore_index = True)
        # Keep a copy of all predictions to track word groups
        self._all_predictions = predictions.copy()

        return predictions


class ProcessToponymExtractorPredictions(ToponymExtractorProcessor):
    def __init__(self, ctrl_points, tiff_dir, buffer = 10):
        """
        Post-process ToponymExtractor outputs

        Parameters
        ----------
        ctrl_points: GeoDataFrame.
            Required. Contains the georeferencing control points for
            each image passed to the ToponymExtractor model.
        tiff_dir: Path.
            Required. Directory path to the Edina downloaded tiff files.
        buffer: int. Default: 10.
            Optional. Number of pixels to buffer ambiguous images by.
        """
        super().__init__(ctrl_points, tiff_dir, buffer)


    def _add_png_overlap_column(
        self, current_predictions: GeoDataFrame
    ) -> GeoDataFrame:
        """
        Determine the relationship between the predictions and the
        overlapping spaces of the TIFF file.

        Adds column "png_overlap" to the current predictions
        GeoDataFrame.

        Returns
        -------
        `current_predictions` GeoDataFrame with "png_overlap" column.
        """
        # default to all predictions belonging to a single png
        current_predictions["png_overlap"] = "disjoint"

        # update predictions that intersect with png overlap area
        selection = current_predictions.geometry\
            .intersects(self.current_tiff_overlap)
        current_predictions.loc[selection, "png_overlap"] = "intersect"

        # update predictions that are contained entirely within png overlap
        # area
        selection = current_predictions.geometry\
            .within(self.current_tiff_overlap)
        current_predictions.loc[selection, "png_overlap"] = "subset"

        return current_predictions
        

    def _update_gdf_current_suppressed(
        self,
        gdf: GeoDataFrame,
        current_predictions: GeoDataFrame,
        selection: Series
    ) -> tuple[GeoDataFrame, GeoDataFrame]:
        """
        Edits overlapping predicitions GeoDataFrame `gdf`,
        `current_predictions` GeoDataFrame, and
        `_suppressed_predictions` GeoDataFrame attribute.

        Returns
        -------
        2-element tuple:
        - `gdf` overlapping predictions GeoDataFrame,
        - `current_predictions` GeoDataFrame
        """
        # Edit suppressed predictions GeoDataFrame with new predictions being
        # suppressed
        suppress_idxs = gdf.loc[selection, "index_right"].tolist()
        self._suppressed_predictions = concat(
            [
                current_predictions.loc[suppress_idxs],
                self._suppressed_predictions
            ],
            axis = 0
        )
        # Remove suppressed predictions from current predictions GeoDataFrame
        current_predictions = current_predictions\
            .loc[current_predictions.index.difference(suppress_idxs)]
        # Remove records of overlapping predictions for suppressed predictions
        gdf = gdf.loc[~gdf.index_right.isin(suppress_idxs)]
        gdf = gdf.loc[gdf.index.difference(suppress_idxs)]
        return gdf, current_predictions


    def _process_intersect_intersect_predictions(
        self, gdf: GeoDataFrame, current_predictions: GeoDataFrame
    ) -> GeoDataFrame:
        """
        Edits `current_predictions` GeoDataFrame,
        `_suppressed_predictions` GeoDataFrame attribute and
        `_undetermined` dictionary.

        Returns
        -------
        `current_predictions` GeoDataFrame with suppressed predictions
        removed.
        """
        # Keep predctions with IoU scores less than or equal to .1
        gdf = gdf[gdf.iou > .1]
        # Apply non-maximal supression for masks with the same text
        # labels
        selection = ((gdf.iou >= .8) & (gdf.word_left == gdf.word_right))
        if selection.sum():
            gdf, current_predictions = self._update_gdf_current_suppressed(
                gdf = gdf,
                current_predictions = current_predictions,
                selection = selection
            )

        if len(gdf):
            # Get indeterminate predictions
            current_predictions = self._process_indeterminant_predictions(
                gdf = gdf, current_predictions = current_predictions
            )
        
        return current_predictions
        

    def _process_subset_subset_predictions(
        self, gdf: GeoDataFrame, current_predictions: GeoDataFrame
    ) -> GeoDataFrame:
        """
        Edits the `current_predictions` GeoDataFrame, and the
        `_suppressed_predictions` GeoDataFrame attribute.

        Returns
        -------
        `current_predictions` GeoDataFrame with suppressed predictions
        removed.
        """
        # Apply non-maximal supression for IoU greater than .8
        selection = (gdf.iou >= .8)
        if selection.sum():
            gdf, current_predictions = self._update_gdf_current_suppressed(
                gdf = gdf,
                current_predictions = current_predictions,
                selection = selection
            )
        return current_predictions
        

    def _process_intercept_subset_predictions(
        self, gdf: GeoDataFrame, current_predictions: GeoDataFrame
    ) -> GeoDataFrame:
        """
        Edits the `current_predictions` GeoDataFrame, the
        `_suppressed_predictions` GeoDataFrame attribute, and the
        `_undetermined` dictionary attribute.

        Returns
        -------
        `current_predictions` GeoDataFrame with suppressed predictions
        removed.
        """
        # Suppress "subset" predictions when words are the same
        selection = (gdf.word_left == gdf.word_right)
        if selection.sum():
            gdf, current_predictions = self._update_gdf_current_suppressed(
                gdf = gdf,
                current_predictions = current_predictions,
                selection = selection
            )
        # Keep predictions where IoU is less than .1
        gdf = gdf[(gdf.iou >= .1)]
        # Suppress "subset" strings contained within the "intersect" prediction
        selection = [
            (
                tup.word_left.startswith(tup.word_right)
                or tup.word_left.endswith(tup.word_right)
            )
            for tup in gdf.itertuples()
        ]
        selection = Series(selection, index = gdf.index, dtype = "bool")
        if selection.sum():
            gdf, current_predictions = self._update_gdf_current_suppressed(
                gdf = gdf,
                current_predictions = current_predictions,
                selection = selection
            )
        if len(gdf):
            # Get indeterminate predictions
            current_predictions = self._process_indeterminant_predictions(
                gdf = gdf, current_predictions = current_predictions
            )
        
        return current_predictions


    def process_predictions(
        self, tiff_fn: str, predictions: GeoDataFrame
    ) -> tuple[
        GeoDataFrame,
        GeoDataFrame,
        dict[str, list[ndarray] | GeoDataFrame]
    ]:
        """
        Post-processes ToponymExtractor outputs. Suppresses overlapping
        predictions from overlapping pngs.

        Parameters
        ----------
        tiff_fn: str.
            Required. Filename for the tiff file the predictions were
            derived from, with the .tif extension.
        predicitons: GeoDataFrame.
            Required. GeoDataFrame containing the word predictions and
            their associated polygon masks, derived from a specific tiff
            file whose filename is passed to `tiff_fn`. Must include the
            fields: "png_filename", "groupid", "word", "score", and
            "geometry".
        
        Returns
        -------
        3-element tuple:

        1. GeoDataFrame. Word predictions with their associated polygon
        masks that were not suppressed.
        2. GeoDataFrame. Suppressed word predictions with their
        associated polygon masks.
        3. Dictionary. Contains an undetermined collection of words and
        their associated metadata required to be passed back to the
        ToponymExtractor. Items include:
        - "image": List of binary ndarrays. Array images containing the
        map text segments that were undetermined. These images contain
        the group text the undetermined words belonged to.
        - "control_points": GeoDataFrame. Contains the georeferencing
        control points for the image snippets. Contains the fields:
        "clique_idx", "pixel_x", "pixel_y", and "geometry".
        - "word_groups": GeoDataFrame. Contains the text instances
        suppressed for each image snippet. Contains the same fields that
        were contained within the `predictions` GeoDataFrame, alongside
        the "clique_idx" field.
        """
        # Initialize attributes
        predictions = super()\
            .process_predictions(tiff_fn = tiff_fn, predictions = predictions)
        
        predictions = self._add_png_overlap_column(predictions)

        # Process intersect - intersect predictions
        selection = (predictions["png_overlap"] == "intersect")
        intersect_intersect_predictions = get_intersecting_png_masks(
            predictions[selection],
            predictions[selection]
        )
        intersect_intersect_predictions = intersect_intersect_predictions[(
            intersect_intersect_predictions.index
            < intersect_intersect_predictions.index_right
        )]
        predictions = self._process_intersect_intersect_predictions(
            intersect_intersect_predictions, predictions
        )

        # Process subset - subset predictions
        selection = (predictions["png_overlap"] == "subset")
        subset_subset_predictions = get_intersecting_png_masks(
            predictions[selection],
            predictions[selection]
        )
        subset_subset_predictions = subset_subset_predictions[(
            subset_subset_predictions.index
            < subset_subset_predictions.index_right
        )]
        predictions = self._process_subset_subset_predictions(
            subset_subset_predictions, predictions
        )

        # process intersect - subset predictions
        intersect_subset_predictions = get_intersecting_png_masks(
            predictions, predictions
        )
        intersect_subset_predictions = intersect_subset_predictions[(
            (intersect_subset_predictions.png_overlap_left == "intersect")
            & (intersect_subset_predictions.png_overlap_right == "subset")
        )]
        predictions = self._process_intercept_subset_predictions(
            intersect_subset_predictions, predictions
        )
        # self._pad_image_snippets()
        return (
            predictions,
            self._suppressed_predictions,
            self._undetermined
        )


class ProcessToponymExtractorPredictionsV2(ToponymExtractorProcessor):
    def __init__(
        self,
        ctrl_points,
        tiff_dir,
        iou_threshold: float,
        a_in_b_threshold: float,
        buffer = 10
    ):
        """
        Post-process ToponymExtractor outputs.

        V2 streamlines the suppression process significantly.
        Non-maximal suppression is not applied; instead, all
        intersecting text instances whose intersection thresholds reach
        the values specified are considered as ambiguous and passed on
        to be predicted again.

        Parameters
        ----------
        ctrl_points: GeoDataFrame.
            Required. Contains the georeferencing control points for
            each image passed to the ToponymExtractor model.
        tiff_dir: Path.
            Required. Directory path to the Edina downloaded tiff files.
        iou_threshold: float.
            Required. Intersection over union threshold value (range:
            [0., 1.]), above which overlapping polygons will be
            suppressed; interacts with `a_in_b_threshold` and in an OR
            manner.
        a_in_b_threshold: float.
            Required. Given a pair of overlapping polygons, 'a in b'
            represents the proportion that one polygon (polygon 'a')
            intersects with the other (polygon 'b') (range: [0., 1.]).
            `a_in_b_threshold` marks threshold value for 'a in b' above
            which overlapping polygons will be suppressed; interacts
            with `iou` in an OR manner.
        buffer: int. Default: 10.
            Optional. Number of pixels to buffer ambiguous images by.
        """
        super().__init__(ctrl_points, tiff_dir, buffer)
    
        if (iou_threshold > 1.) or (iou_threshold < 0.):
            raise ValueError(
                f"Argument passed to iou_threshold parameter must be between"\
                f"0. and 1. (inclusive). Argument passed: {iou_threshold}"
            )
        self._iou_threshold: float = iou_threshold

        if (a_in_b_threshold > 1.) or (a_in_b_threshold < 0.):
            raise ValueError(
                f"Argument passed to a_in_b_threshold parameter must be "\
                f"between 0. and 1. (inclusive). Argument passed: "\
                f"{a_in_b_threshold}"
            )
        self._a_in_b_threshold: float = a_in_b_threshold


    def process_predictions(
        self, tiff_fn: str, predictions: GeoDataFrame
    ) -> tuple[
        GeoDataFrame,
        GeoDataFrame,
        dict[str, list[ndarray] | GeoDataFrame]
    ]:
        """
        Post-processes ToponymExtractor outputs. Suppresses overlapping
        predictions from overlapping pngs.

        Parameters
        ----------
        tiff_fn: str.
            Required. Filename for the tiff file the predictions were
            derived from, with the .tif extension.
        predicitons: GeoDataFrame.
            Required. GeoDataFrame containing the word predictions and
            their associated polygon masks, derived from a specific tiff
            file whose filename is passed to `tiff_fn`. Must include the
            fields: "png_filename", "groupid", "word", "score", and
            "geometry".
        
        Returns
        -------
        3-element tuple:

        1. GeoDataFrame. Word predictions with their associated polygon
        masks that were not suppressed.
        2. GeoDataFrame. Suppressed word predictions with their
        associated polygon masks.
        3. Dictionary. Contains an undetermined collection of words and
        their associated metadata required to be passed back to the
        ToponymExtractor. Items include:
        - "image": List of binary ndarrays. Array images containing the
        map text segments that were undetermined. These images contain
        the group text the undetermined words belonged to.
        - "control_points": GeoDataFrame. Contains the georeferencing
        control points for the image snippets. Contains the fields:
        "clique_idx", "pixel_x", "pixel_y", and "geometry".
        - "word_groups": GeoDataFrame. Contains the text instances
        suppressed for each image snippet. Contains the same fields that
        were contained within the `predictions` GeoDataFrame, alongside
        the "clique_idx" field.
        """
        # Initialize attributes
        predictions = super()\
            .process_predictions(tiff_fn = tiff_fn, predictions = predictions)
        
        overlapping_predictions, intersect_area =\
            _get_intersecting_png_masks(predictions, predictions)

        overlapping_predictions["l_in_r_area_pc"] =\
            intersect_area / overlapping_predictions.geometry.area
        
        overlapping_predictions["r_in_l_area_pc"] = self._all_predictions\
            .loc[overlapping_predictions.index_right, "geometry"].area.array
        overlapping_predictions["r_in_l_area_pc"] =\
            intersect_area / overlapping_predictions.pop("r_in_l_area_pc")
        
        # select predictions for suppression by the thresholds passed on init
        selection = (
            (overlapping_predictions.iou >= self._iou_threshold)
            |(overlapping_predictions.l_in_r_area_pc >= self._a_in_b_threshold)
            |(overlapping_predictions.r_in_l_area_pc >= self._a_in_b_threshold)
        )
        overlapping_predictions = overlapping_predictions[selection]

        # All retained predictions are classified as
        # indeterminate/ambiguous, their image snippets are passed back
        # to be re-predicted.
        predictions = self._process_indeterminant_predictions(
            gdf = overlapping_predictions, current_predictions = predictions
        )
        return (
            predictions,
            self._suppressed_predictions,
            self._undetermined
        )
