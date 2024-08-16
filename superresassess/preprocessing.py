from pathlib import Path
from typing import Sequence

import torch.utils.data

from monai.transforms.compose import Compose
from monai.transforms.io.array import LoadImage, SaveImage
from monai.transforms.utility.array import EnsureChannelFirst
from monai.transforms.spatial.array import Orientation, Spacing, ResampleToMatch
from monai.transforms.intensity.array import (
    ScaleIntensityRangePercentiles,
    GaussianSmooth,
)
from monai.utils.misc import first
from monai.data.dataset import ArrayDataset


def _get_preprocessing_loader(
    lower_percentile: float = 0.01,
    upper_percentile: float = 99.99,
    lower_out: int | float = 0,
    upper_out: int | float = 255,
    orientation: str = "RAS",
):
    return Compose(
        [
            LoadImage(),
            EnsureChannelFirst(),
            Orientation(axcodes=orientation),
            ScaleIntensityRangePercentiles(
                lower_percentile, upper_percentile, lower_out, upper_out
            ),
        ]
    )


def _get_downsampler(sigma: float, lr_pixdim: Sequence[float]) -> Compose:
    return Compose(
        [
            GaussianSmooth(sigma=sigma),
            Spacing(lr_pixdim),
        ]
    )


class PrepareHRLRData:
    """A class that prepares the high resolution (HR) and low resolution (LR) data for
    the experiment.


    Args:
        lower_percentile (float):
        upper_percentile (float):
        lower_range_output (float):
        upper_range_output (float):
        output_orientation (float):
        smoothing_sigma (float):
        low_resolution_voxel_dimension (float):
    """

    def __init__(
        self,
        lower_percentile: float,
        upper_percentile: float,
        lower_range_output: int | float,
        upper_range_output: int | float,
        smoothing_sigma: float,
        downsampling_scale: int,
        output_orientation: str = "RAS",
    ):
        self.preprocessing_loader = _get_preprocessing_loader(
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
            lower_out=lower_range_output,
            upper_out=upper_range_output,
            orientation=output_orientation,
        )
        self.sigma = smoothing_sigma
        self.scale = downsampling_scale

    def prepare_data(self, file_ending: str, raw_path: Path, processed_path: Path):
        files = list(raw_path.glob(f"**/*{file_ending}"))
        hr_path = processed_path.joinpath("hr")
        hr_path.mkdir(parents=True, exist_ok=True)
        lr_path = processed_path.joinpath("lr")
        lr_path.mkdir(parents=True, exist_ok=True)

        lr_saver = SaveImage(
            output_dir=lr_path,
            output_postfix="",
            separate_folder=False,  # already created separate folders
        )
        hr_saver = SaveImage(
            output_dir=hr_path,
            output_postfix="",
            separate_folder=False,  # already created separate folders
        )

        ds = ArrayDataset(files, self.preprocessing_loader)

        # this must be the torch DataLoader, the monai DataLoader is breaking the tests
        loader = torch.utils.data.DataLoader(ds, batch_size=None)

        img = first(loader)
        hr_pixdim = img.pixdim  # type: ignore
        lr_pixdim = tuple([dim * self.scale for dim in hr_pixdim])

        downsampler = _get_downsampler(self.sigma, lr_pixdim)
        matcher = ResampleToMatch(mode="bilinear")

        for im in loader:
            im = im[0, ...]
            im_lr = downsampler(im)
            im_interpolated = matcher(im_lr, im)  # type: ignore

            hr_saver(im)
            lr_saver(im_interpolated)
