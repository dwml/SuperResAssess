from superresassess.preprocessing import PrepareHRLRData

import torch
from pathlib import Path
from monai.data import ITKWriter
from monai.transforms import LoadImage
import pytest
import numpy as np


@pytest.fixture
def prepare_example_image(tmp_path) -> Path:
    random_data = torch.randn(50, 50, 50)
    test_affine = torch.as_tensor(
        [[0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 0.5, 0], [0, 0, 0, 1]],
        dtype=torch.float16,
    )
    writer = ITKWriter(output_dtype=np.float32)
    writer.set_data_array(random_data, channel_dim=None)
    writer.set_metadata({"affine": test_affine})
    file_name = tmp_path.joinpath("test.nii.gz")
    writer.write(file_name)
    return file_name


@pytest.mark.slow
def test_preprocessing_pipeline(prepare_example_image):
    parent_path = prepare_example_image.parent
    preprocessor = PrepareHRLRData(
        lower_percentile=0.01,
        upper_percentile=99.99,
        lower_range_output=0,
        upper_range_output=255,
        smoothing_sigma=1.0,
        downsampling_scale=2,
    )
    preprocessor.prepare_data(".nii.gz", parent_path, parent_path)

    hr_path = parent_path.joinpath("hr/test.nii.gz")
    lr_path = parent_path.joinpath("lr/test.nii.gz")
    reader = LoadImage()
    hr_image = reader(hr_path)
    lr_image = reader(lr_path)

    assert hr_path.is_file()
    assert lr_path.is_file()
    assert (np.diag(hr_image.meta["affine"]) == np.array([0.5, 0.5, 0.5, 1.0])).all()
    # because of the resampling, the assertion below is 0.5, 0.5, 0.5,
    # which is the original spacing. Maybe come up with some better test.
    assert (np.diag(lr_image.meta["affine"]) == np.array([0.5, 0.5, 0.5, 1.0])).all()
    assert (lr_image.data != hr_image.data).any()
