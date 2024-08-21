from pathlib import Path
import pytest
import nibabel as nib
import numpy as np
from lightning import LightningModule
from superresassess.model import LitReCNN, ReCNNConfiguration
from superresassess.data import DataListType, DataLoaderConfig, CroppedDataLoaderConfig


def _mock_data(n: int, save_dir: Path) -> DataListType:
    """Create 10 hr/lr pairs"""
    # Setup path
    hr_path = save_dir.joinpath("hr")
    hr_path.mkdir(parents=True, exist_ok=True)
    lr_path = save_dir.joinpath("lr")
    lr_path.mkdir(parents=True, exist_ok=True)

    hr_list = []
    lr_list = []
    for i in range(n):
        sg = np.random.SeedSequence(1988)
        rng = np.random.Generator(np.random.MT19937(sg))
        hr_filename = hr_path.joinpath(f"{i:0>5}.nii.gz")
        hr_image = nib.Nifti1Image(rng.random(size=[32, 32, 32]), np.eye(4))
        nib.save(hr_image, hr_filename)
        hr_list.append(hr_filename)
        lr_filename = lr_path.joinpath(f"{i:0>5}.nii.gz")
        lr_image = nib.Nifti1Image(rng.random(size=[32, 32, 32]), np.eye(4))
        nib.save(lr_image, lr_filename)
        lr_list.append(lr_filename)

    return [{"img": image, "lab": label} for image, label in zip(lr_list, hr_list)]


@pytest.fixture(scope="session")
def mock_data(tmpdir_factory) -> DataListType:
    """Create 20 hr/lr pairs"""
    # Setup path
    save_dir = Path(tmpdir_factory.mktemp("data"))

    return _mock_data(20, save_dir)


@pytest.fixture(scope="session")
def mock_data_with_external(tmpdir_factory) -> DataListType:
    """Create 40 hr/lr pairs"""
    # Setup path
    save_dir = Path(tmpdir_factory.mktemp("data"))

    return _mock_data(40, save_dir)


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


N_INTERNAL_IMAGES = 20
SEED = 2024


@pytest.fixture
def cropped_dataloader_config():
    return CroppedDataLoaderConfig(
        seed=SEED,
        dict_keys=("img", "lab"),
        batch_size=32,
        num_workers=41,
        roi_size=(32, 32, 32),
        samples_per_image=200,
        limit_train_batches=20,
    )


@pytest.fixture
def dataloader_config():
    return DataLoaderConfig(
        seed=SEED,
        dict_keys=("img", "lab"),
        batch_size=1,
        num_workers=41,
    )
