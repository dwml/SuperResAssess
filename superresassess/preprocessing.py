import monai
import torch

from pathlib import Path


def _get_preprocessing_loader(
    lower_percentile: float = 0.01,
    upper_percentile: float = 99.99,
    lower_out: int | float = 0,
    upper_out: int | float = 255,
    orientation: str = "RAS",
):
    return monai.transforms.Compose(
        [
            monai.transforms.LoadImage(),
            monai.transforms.EnsureChannelFirst(),
            monai.transforms.Orientation(axcodes=orientation),
            monai.transforms.ScaleIntensityRangePercentiles(
                lower_percentile, upper_percentile, lower_out, upper_out
            ),
        ]
    )


def _get_downsampler(
    sigma: float, lr_pixdim: tuple[float, float, float]
) -> torch.nn.Module:
    return monai.transforms.Compose(
        [
            monai.transforms.GaussianSmooth(sigma=sigma),
            monai.transforms.Spacing(lr_pixdim),
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
        print(files)
        hr_path = processed_path.joinpath("hr")
        hr_path.mkdir(parents=True, exist_ok=True)
        lr_path = processed_path.joinpath("lr")
        lr_path.mkdir(parents=True, exist_ok=True)

        lr_saver = monai.transforms.SaveImage(
            output_dir=lr_path,
            output_postfix="",
            separate_folder=False,  # already created separate folders
        )
        hr_saver = monai.transforms.SaveImage(
            output_dir=hr_path,
            output_postfix="",
            separate_folder=False,  # already created separate folders
        )

        ds = monai.data.ArrayDataset(files, self.preprocessing_loader)
        loader = torch.utils.data.DataLoader(ds, batch_size=None)

        img = monai.utils.first(loader)
        hr_pixdim = img.pixdim
        lr_pixdim = [dim * self.scale for dim in hr_pixdim]

        downsampler = _get_downsampler(self.sigma, lr_pixdim)
        matcher = monai.transforms.ResampleToMatch(mode="bilinear")

        for im in loader:
            im = im[0, ...]
            im_lr = downsampler(im)
            im_interpolated = matcher(im_lr, im)

            hr_saver(im)
            lr_saver(im_interpolated)
