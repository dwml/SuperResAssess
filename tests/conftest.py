from pathlib import Path
import pytest
import nibabel as nib
import numpy as np
from lightning import LightningModule
from superresassess.model import LitReCNN, ReCNNConfiguration


@pytest.fixture(scope="session")
def mock_data(tmpdir_factory) -> list[dict[[str], Path]]:
    """Create 10 hr/lr pairs"""
    # Setup path
    hr_path = Path(tmpdir_factory.mktemp("hr"))
    lr_path = Path(tmpdir_factory.mktemp("lr"))

    hr_list = []
    lr_list = []
    for i in range(10):
        hr_filename = hr_path.joinpath(f"{i:0>5}.nii.gz")
        hr_image = nib.Nifti1Image(
            np.random.randint(0, 2, size=[32, 32, 32]).astype(float), np.eye(4)
        )
        nib.save(hr_image, hr_filename)
        hr_list.append(hr_filename)
        lr_filename = lr_path.joinpath(f"{i:0>5}.nii.gz")
        lr_image = nib.Nifti1Image(
            np.random.randint(0, 2, size=[32, 32, 32]).astype(float), np.eye(4)
        )
        nib.save(lr_image, lr_filename)
        lr_list.append(lr_filename)

    return [{"img": image, "lab": label} for image, label in zip(lr_list, hr_list)]


@pytest.fixture(scope="session")
def mock_recnn_config() -> ReCNNConfiguration:
    return ReCNNConfiguration(
        n_layers=10,
        spatial_dims=3,
        in_channels=1,
        intermediate_channels=4,
        out_channels=1,
        kernel_size=3,
        stride=1,
        padding="same",
    )


@pytest.fixture(scope="session")
def mock_model(mock_recnn_config) -> LightningModule:
    return LitReCNN(configuration=mock_recnn_config)
