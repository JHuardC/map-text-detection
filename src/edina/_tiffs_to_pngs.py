"""
Converts directory of EDINA tiff files to PNGs.
"""
# Imports
from typing import Final
from logging import getLogger
from collections.abc import Iterable
from pathlib import Path
import progressbar
from numpy import ndarray
from shapely import Point
from PIL.Image import Image, fromarray as image_fromarray
from pandas import concat as pd_concat
from geopandas import GeoDataFrame
from rasterio import open as open_raster
from rasterio.transform import AffineTransformer
from rasterio.crs import CRS

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

class EDINATiffPNGConverter:
    def __init__(self):
        pass

    def _get_raster_data(
        self, tiff_path: Path
    ) -> tuple[ndarray, AffineTransformer, CRS, tuple[int, int]]:
        """Open EDINA tiff file and extract relevant data."""
        with open_raster(tiff_path, mode = "r") as src:
            tiff_height = src.height # image height
            tiff_width = src.width # image width
            tiff_transformer = AffineTransformer(src.transform) # transformer
            tiff_crs = src.read_crs() # coordinate reference system
            data = (-src.read() + 1) * 255 # image array
        
        return data, tiff_transformer, tiff_crs, (tiff_height, tiff_width)

    def _get_arr_split_idxs(
        self, img_dim: int, partsize_dim: int, overlap: int, start: int = 0
    ) -> list[int]:
        """
        Get index values used to split the image array along a specific
        dimension.
        """
        # Initial index values to split along - final split index will
        # need adjustment
        split_idxs = [*range(start, img_dim, partsize_dim - overlap)]

        # Check if end of image can be fully captured by penultimate partition
        if (img_dim - split_idxs[-1]) <= overlap:
            # Drop final split index
            split_idxs = split_idxs[: -1]

        # Adjust final split to make the remainder of img_dim equal to
        # partsize dim
        split_idxs = [*split_idxs[: -1], img_dim - partsize_dim]
        return split_idxs
    
    def _get_pixel_rowcol_idxs(
        self,
        arr_h: int,
        arr_w: int,
        part_h: int,
        part_w: int,
        overlap: int,
        start_h: int,
        start_w: int
    ) -> list[tuple[int, int]]:
        """
        Get index values used to split the image array.
        """
        row_splits = self._get_arr_split_idxs(arr_h, part_h, overlap, start_h)
        col_splits = self._get_arr_split_idxs(arr_w, part_w, overlap, start_w)
        return [(row, col) for row in row_splits for col in col_splits]
    
    def _create_image_and_meta(
        self,
        img_arr: ndarray,
        row: int,
        col: int,
        png_h: int,
        png_w: int,
        transformer: AffineTransformer,
        tiff_filename: str,
        png_filename: str
    ) -> tuple[Image, list[dict[str, str | int | Point]]]:
        """
        Splits image array and converts into sub-arrays into pngs and
        provides control points metadata.
        """
        control_points_meta = [
            {
                "tiff_filename": tiff_filename,
                "png_filename": png_filename,
                "pixel_x": i * (png_w - 1),
                "pixel_y": j * (png_h - 1),
                "geometry": Point(transformer.xy(x, y))
            }
            for i, x in enumerate([col, col + png_w - 1])
            for j, y in enumerate([row, row + png_h - 1])
        ]
        img = img_arr[0, row: row + png_h, col: col + png_w].copy()
        img = image_fromarray(obj = img, mode =  "L")

        return img, control_points_meta
    
    def _create_georef_df(
        self,
        control_point_records: list[dict[str, str | int | Point]],
        crs: CRS,
        to_crs: str | None = None
    ) -> GeoDataFrame:
        # Convert control points records to GeoDataFrame
        gdf = GeoDataFrame.from_records(control_point_records)
        
        if to_crs is None:
            # if no crs conversion, then record current crs using WKT
            gdf["crs"] = crs.to_wkt()
        else:
            gdf = gdf.set_crs(crs).to_crs(to_crs)
        
        return gdf
        
        
    def convert_tiff_to_pngs(
        self,
        tiff_path: Path,
        png_dest: Path,
        png_h: int | None = None,
        png_w: int | None = None,
        overlap: int = 0,
        to_crs: str | None = None,
        start_h: int = 0,
        start_w: int = 0
    ) -> GeoDataFrame:
        """
        Reads Edina tiff files and saves out splits of the image as
        pngs. Returns metadata containing control points for each png.

        Parameters
        ----------
        tiff_path: Path.
            Required. Path to EDINA tiff file containing georeferenced
            image.
        
        png_dest: Path.
            Required. Path to save PNG files out to. Should be a
            directory, unless the tiff file is not being split.
        
        png_h: int or None. Default: None.
            Optional. Height dimension of the PNG images being saved
            out. None means the tiff file will not be split height-wise
            and output PNGs will have the same height as the original
            tiff file.
        
        png_w: int or None. Default: None.
            Optional. Width dimension of the PNG images being saved
            out. None means the tiff file will not be split width-wise
            and output PNGs will have the same width as the original
            tiff file.
        
        overlap: int. Default: 0.
            Optional. Number of pixels adjacent PNGs will overlap by
            (both horizontally and vertically adjacent PNGs overlap by
            the same amount).

        to_crs: str or None. Default: None.
            Optional. If an argument is provided, convert the crs of the
            control points for georeferencing the PNGs.
        
        start_h: int. Default: 0.
            Optional. Specify where to start splits along the vertical
            axis. Effectively crops the top of the tiff image by
            start_h pixels.
        
        start_w: int. Default: 0.
            Optional. Specify where to start splits along the horizontal
            axis. Effectively crops the left side of the tiff image by
            start_w pixels.

        Returns
        -------
        GeoDataFrame. Each record contains the details of a control
        point used for georeferencing for each png saved out. Fields:
        - "tiff_filename": str. Name of tiff file control point was
        derived from.
        - "png_filename": str. Name of PNG file control point
        georeferences.
        - "pixel_x": int. Column value for the pixel corresponding to
        the coordinate stored in geometry.
        - "pixel_y": int. Row value for the pixel corresponding to the
        coordinate stored in geometry.
        - "geometry": shapely.Point. Contains the coordinates for the
        control points encoded under the crs provided.
        """
        tiffarr, transformer, crs, hw = self._get_raster_data(tiff_path)
        
        # If png dimensions were not specified, make them equal to the
        # tiff dimensions
        png_h = hw[0] if png_h is None else png_h
        png_w = hw[1] if png_w is None else png_w

        pixel_rowcol_split_idxs = self._get_pixel_rowcol_idxs(
            *hw, png_h, png_w, overlap, start_h, start_w
        )

        control_points_meta = []
        for idx, (row, col) in enumerate(pixel_rowcol_split_idxs, start = 1):
            # derive png filename
            if png_dest.is_dir():
                # png filenames derived from tiff filename
                png_filename = f"{tiff_path.stem}-{idx}.png"
            elif len(pixel_rowcol_split_idxs) == 1:
                # use provided png filename
                png_filename = png_dest.name
                png_dest = png_dest.parent
            else:
                # File name provided, but need to save more than one png
                raise ValueError(
                    "png_dest argument must be a directory to save out "
                    "multiple pngs."
                )
            # get image and control points used for georeferencing the image
            img, ctrl_points = self._create_image_and_meta(
                tiffarr,
                row,
                col,
                png_h,
                png_w,
                transformer,
                tiff_path.name,
                png_filename
            )
            control_points_meta += ctrl_points
            img.save(png_dest.joinpath(png_filename))
            logger.info(
                f"Image {png_dest.joinpath(png_filename).stem} saved out"
            )

        # Convert control points records to GeoDataFrame
        return self._create_georef_df(control_points_meta, crs, to_crs)

    def convert_batch_tiff_to_pngs(
        self,
        tiff_paths: Iterable[Path],
        png_dest: Path,
        png_h: int | None = None,
        png_w: int | None = None,
        overlap: int = 0,
        to_crs: str | None = None,
        start_h: int = 0,
        start_w: int = 0
    ) -> GeoDataFrame:
        """
        Reads a collection Edina tiff files and saves out splits of each
        image as pngs. Returns metadata containing control points for
        each png.

        Parameters
        ----------
        tiff_paths: Iterable of Path instances.
            Required. Paths to EDINA tiff files containing georeferenced
            images.
        
        png_dest: Path.
            Required. Path to save PNG files out to, must be a
            directory.
        
        png_h: int or None. Default: None.
            Optional. Height dimension of the PNG images being saved
            out. None means tiff files will not be split height-wise
            and output PNGs will have the same height as the tiff file
            the PNG was derived from.
        
        png_w: int or None. Default: None.
            Optional. Width dimension of the PNG images being saved
            out. None means tiff files will not be split width-wise
            and output PNGs will have the same width as the tiff file
            the PNG was derived from.
        
        overlap: int. Default: 0.
            Optional. Number of pixels adjacent PNGs will overlap by
            (both horizontally and vertically adjacent PNGs overlap by
            the same amount).

        to_crs: str or None. Default: None.
            Optional. If an argument is provided, convert the crs of the
            control points used to georeference the PNGs.
        
        start_h: int. Default: 0.
            Optional. Specify where to start splits along the vertical
            axis. Effectively crops the top of each tiff image by
            start_h pixels.
        
        start_w: int. Default: 0.
            Optional. Specify where to start splits along the horizontal
            axis. Effectively crops the left side of each tiff image by
            start_w pixels.

        Returns
        -------
        GeoDataFrame. Each record contains the details of a control
        point used for georeferencing for each png saved out. Fields:
        - "tiff_filename": str. Name of tiff file control point was
        derived from.
        - "png_filename": str. Name of PNG file control point
        georeferences.
        - "pixel_x": int. Column value for the pixel corresponding to
        the coordinate stored in geometry.
        - "pixel_y": int. Row value for the pixel corresponding to the
        coordinate stored in geometry.
        - "geometry": shapely.Point. Contains the coordinates for the
        control points encoded under the crs provided.
        - "crs": str. Coordinate Reference System for control points in
        well known text (WKT) format.
        """
        # Argument check for png_dest
        if png_dest.is_file():
            raise ValueError(
                "png_dest argument must be a directory, not an individual "\
                "file."
            )
        
        tiff_paths = [*iter(tiff_paths)]
        if len(tiff_paths) == 0:
            raise ValueError(
                "No tiff files found in tiff_paths iterable passed"
            )

        # Construct progressbar
        progress = progressbar\
            .ProgressBar(0, len(tiff_paths), _WIDGETS, prefix = "Read Tiffs:")
        control_points_meta = None
        try:
            # start cycling through tiffs
            tiff_paths = iter(tiff_paths)
            tiff_path = next(tiff_paths)
            progress.start()

            tiffarr, transformer, crs, self._hw =\
                self._get_raster_data(tiff_path)
            progress.increment(1)
            
            # store arguments for deriving row-col split indices.
            args = (
                *self._hw,
                self._hw[0] if png_h is None else png_h,
                self._hw[1] if png_w is None else png_w,
                overlap,
                start_h,
                start_w
            )
            pxl_rc_split_idxs = self._get_pixel_rowcol_idxs(*args)

            control_points_records = []
            for idx, (row, col) in enumerate(pxl_rc_split_idxs, start = 1):
                # derive png filename
                png_filename = f"{tiff_path.stem}-{idx}.png"
                # get image and control points used for georeferencing
                img, ctrl_points = self._create_image_and_meta(
                    tiffarr,
                    row,
                    col,
                    png_h,
                    png_w,
                    transformer,
                    tiff_path.name,
                    png_filename
                )
                control_points_records += ctrl_points
                img.save(png_dest.joinpath(png_filename))
                logger.info(
                    f"Image {png_dest.joinpath(png_filename).stem} saved "\
                    "out."
                )

            # Convert control points records to GeoDataFrame
            control_points_meta =\
                self._create_georef_df(control_points_records, crs, to_crs)

            while 1:

                tiff_path = next(tiff_paths)
                tiffarr, transformer, crs, hw =\
                    self._get_raster_data(tiff_path)
                progress.increment(1)
                
                # Check new tiff has same height-width dimensions as the
                # previous tiff
                if self._hw != hw:
                    # When dimensions are different, need to recalculate
                    # split locations
                    self._hw = hw
                    args = (
                        *self._hw,
                        self._hw[0] if png_h is None else png_h,
                        self._hw[1] if png_w is None else png_w,
                        overlap,
                        start_h,
                        start_w
                    )
                    pxl_rc_split_idxs = self._get_pixel_rowcol_idxs(*args)
                
                control_points_records = []
                for idx, (row, col) in enumerate(pxl_rc_split_idxs, start = 1):
                    # derive png filename
                    png_filename = f"{tiff_path.stem}-{idx}.png"
                    # get image and control points used for georeferencing
                    img, ctrl_points = self._create_image_and_meta(
                        tiffarr,
                        row,
                        col,
                        png_h,
                        png_w,
                        transformer,
                        tiff_path.name,
                        png_filename
                    )
                    control_points_records += ctrl_points
                    # Save image out
                    img.save(png_dest.joinpath(png_filename))
                    logger.info(
                        f"Image {png_dest.joinpath(png_filename).stem} saved "\
                        "out."
                    )

                # Convert control points records to GeoDataFrame
                control_points_meta = pd_concat([
                    control_points_meta,
                    self._create_georef_df(control_points_records, crs, to_crs)
                ])

        except StopIteration as _:
            progress.finish()
        except Exception as e:
            progress.finish(dirty = True)
            raise
        # Convert control points records to GeoDataFrame
        return control_points_meta
