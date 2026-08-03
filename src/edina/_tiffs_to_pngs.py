"""
Converts directory of EDINA tiff files to PNGs.
"""
# Imports
from typing import Final
from abc import abstractmethod
from functools import partial
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

def get_edina_tiff_data(
    tiff_path: Path | str
) -> tuple[ndarray, AffineTransformer, CRS, tuple[int, int]]:
    """
    Load TIFF file downloaded from EDINA and retrieve image, geodata,
    and metadata.

    Image is converted from binary to greyscale format.

    Parameters
    ----------
    raster_path: Path or string.
        Required. Path to EDINA downloaded TIFF file, the images in this
        file are in binary format.
    
    Returns
    -------
    4-element tuple:

    1. ndarray: (1, H, W) Grey-scale representation of the raster image.
    2. AffineTransformer: Georeferencing transformer, converting pixel
    locations to map crs coordinates and vice versa.
    3. CRS: Coordinate reference system for map coordinates.
    4. (H, W): Dimensions of the image, matches dimensions two and three
    of ndarray in element 1.
    """
    with open_raster(tiff_path, mode = "r") as src:
        tiff_height = src.height # image height
        tiff_width = src.width # image width
        tiff_transformer = AffineTransformer(src.transform) # transformer
        tiff_crs = src.read_crs() # coordinate reference system
        data = (-src.read() + 1) * 255 # image array
    
    return data, tiff_transformer, tiff_crs, (tiff_height, tiff_width)


class _ImageArraySplitter:

    @abstractmethod
    def _get_arr_split_idxs(self, *args, **kwargs) -> list[int]:
        pass
    
    def get_pixel_rowcol_idxs(
        self, row_args: tuple, col_args: tuple
    ) -> list[tuple[int, int]]:
        """
        Override this function to unpack row arguments and column
        arguments.
        """
        row_splits = self._get_arr_split_idxs(*row_args)
        col_splits = self._get_arr_split_idxs(*col_args)
        return [(row, col) for row in row_splits for col in col_splits]


class _SplitWithOverlap(_ImageArraySplitter):

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

    def get_pixel_rowcol_idxs(
        self,
        arr_h: int,
        arr_w: int,
        part_h: int,
        part_w: int,
        overlap: int,
        start_h: int,
        start_w: int,
        **kwargs
    ) -> list[tuple[int, int]]:
        """
        Get index values used to split the image array.
        """
        row_args = arr_h, part_h, overlap, start_h
        col_args = arr_w, part_w, overlap, start_w
        return super().get_pixel_rowcol_idxs(row_args, col_args)


class _SplitWithoutOverlap(_ImageArraySplitter):

    def _get_arr_split_idxs(
        self, img_dim: int, partsize_dim: int, start: int = 0
    ) -> list[int]:
        """
        Get index values used to split the image array along a specific
        dimension.
        """
        # Initial index values to split along - final split index will
        # need adjustment
        split_idxs = [*range(start, img_dim, partsize_dim)]

        # Drop final split index if remainder is not a full partsize image
        if img_dim - split_idxs[-1] < partsize_dim:
            split_idxs = split_idxs[:-1]

        return split_idxs

    def get_pixel_rowcol_idxs(
        self,
        arr_h: int,
        arr_w: int,
        part_h: int,
        part_w: int,
        start_h: int,
        start_w: int,
        **kwargs
    ) -> list[tuple[int, int]]:
        """
        Get index values used to split the image array.
        """
        row_args = arr_h, part_h, start_h
        col_args = arr_w, part_w, start_w
        return super().get_pixel_rowcol_idxs(row_args, col_args)


class _ImageSaver:
    @staticmethod
    @abstractmethod
    def save_image(img: Image, path: str | Path, *args, **kwargs) -> None:
        pass

class _AutoSaver(_ImageSaver):
    @staticmethod
    def save_image(img: Image, path: str | Path, *args, **kwargs) -> None:
        img.save(path)
        return None

class _SaveInMode(_ImageSaver):
    "Converts image to specified mode"
    @staticmethod
    def save_image(
        img: Image, path: str | Path, img_mode: str, *args, **kwargs
    ) -> None:
        img.convert(mode = img_mode).save(path)
        return None


class EDINATiffPNGConverter:
    def __init__(self, img_mode: str | None = None):
        """
        Converts EDINA downloaded TIFF files to clipped PNG Images with
        georeferencing metadata.

        Parameters
        ----------
        img_mode: str or None. Default: None.
            Optional. Mode the image should be saved as, e.g. "L", or
            "RGB". If no argument is passed, image will not be
            converted before being saved out.
        """
        self.image_mode = img_mode
        # Construct image saver function
        check = (img_mode is None)
        self._image_saver = (_AutoSaver if check else _SaveInMode).save_image
    
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
                "geometry": Point(transformer.xy(r, c))
            }
            for i, c in enumerate([col, col + png_w - 1])
            for j, r in enumerate([row, row + png_h - 1])
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
        # Get image clipper
        clipper = (
            _SplitWithOverlap() if overlap > 0 else _SplitWithoutOverlap()
        )
        clipper = partial(
            clipper.get_pixel_rowcol_idxs,
            overlap = overlap,
            start_h = start_h,
            start_w = start_w
        )

        tiffarr, transformer, crs, hw = get_edina_tiff_data(tiff_path)
        
        # If png dimensions were not specified, make them equal to the
        # tiff dimensions
        png_h = hw[0] if png_h is None else png_h
        png_w = hw[1] if png_w is None else png_w

        pixel_rowcol_split_idxs = clipper(*hw, png_h, png_w)

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
            self._image_saver(
                img,
                png_dest.joinpath(png_filename),
                img_mode = self.image_mode
            )
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
        - "crs": str. Optional. If no crs argument is provided then
        this field contains the coordinate Reference System for control
        points in well known text (WKT) format.
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
        
        # Get image clipper
        clipper = (
            _SplitWithOverlap() if overlap > 0 else _SplitWithoutOverlap()
        )
        clipper = partial(
            clipper.get_pixel_rowcol_idxs,
            overlap = overlap,
            start_h = start_h,
            start_w = start_w
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
                get_edina_tiff_data(tiff_path)
            progress.increment(1)
            
            # store arguments for deriving row-col split indices.
            args = (
                *self._hw,
                self._hw[0] if png_h is None else png_h,
                self._hw[1] if png_w is None else png_w
            )
            pxl_rc_split_idxs = clipper(*args)

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
                self._image_saver(
                    img,
                    png_dest.joinpath(png_filename),
                    img_mode = self.image_mode
                )
                logger.info(
                    f"Image {png_dest.joinpath(png_filename).stem} saved "\
                    "out."
                )

            # Convert control points records to GeoDataFrame
            control_points_meta =\
                self._create_georef_df(control_points_records, crs, to_crs)

            while 1:

                tiff_path = next(tiff_paths)
                tiffarr, transformer, crs, hw = get_edina_tiff_data(tiff_path)
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
                    )
                    pxl_rc_split_idxs = clipper(*args)
                
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
                    self._image_saver(
                        img,
                        png_dest.joinpath(png_filename),
                        img_mode = self.image_mode
                    )
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
