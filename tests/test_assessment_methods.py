from pathlib import Path

import pytest

from superresassess.assessment.base import _check_ratios_with_folds
from superresassess.assessment.enums import AssessmentEnum
from superresassess.data import (
    ImageDatasetd,
    get_image_loader,
    DataLoaderConfig,
    CroppedDataLoaderConfig,
)
from superresassess.experiments import ExperimentConfiguration

from lightning import Trainer
from lightning.pytorch.loggers import CSVLogger

ASSESSMENT_CONFIG = Path("./configurations/assessment.yaml")
ALL_EXISTING_METHODS = [
    "three_way_holdout",
    "k_fold_cross_validation",
    "nested_cross_validation",
]

NON_EXISTING_METHODS = [
    "failing_method",
    "grid_search",
]


@pytest.mark.parametrize("existing_method", ALL_EXISTING_METHODS)
def test_existing_assessment_methods(existing_method):
    assessment = AssessmentEnum(existing_method)
    assert assessment.name == existing_method


@pytest.mark.parametrize("non_existing_method", NON_EXISTING_METHODS)
def test_non_existing_assessment_methods(non_existing_method):
    with pytest.raises(Exception):
        _ = AssessmentEnum(non_existing_method)


@pytest.fixture
def image_dataset(mock_data):
    """Returns a ImageDatasetd containing 10 low-res/high-res pairs"""
    return ImageDatasetd(mock_data, get_image_loader(dict_keys=("img", "lab")))


@pytest.mark.parametrize(
    "incorrect_ratios, number_images",
    [((0.8, 0.05, 0.15), 10), ((0.67, 0.1, 0.23), 20), ((0.9, 0.1, 0.1), 10)],
)
def test_check_ratios_raises_value_error(incorrect_ratios, number_images):
    with pytest.raises(ValueError):
        _check_ratios_with_folds(incorrect_ratios, number_images)


@pytest.mark.parametrize("correct_ratios", [(0.8, 0.1, 0.1), (0.6, 0.2, 0.2)])
def test_check_ratios_raises_no_error(
    correct_ratios,
):
    # Check that this does not raise an error
    _check_ratios_with_folds(correct_ratios, 20)


class AssessmentTestSetup:
    my_seed = 2024
    my_max_epochs = 2
    my_n_internal_images = 20
    my_experiment_id = "00001"
    my_learning_rate = 1e-3
    my_fast_training_dataloader = CroppedDataLoaderConfig(
        seed=my_seed,
        dict_keys=("img", "lab"),
        batch_size=32,
        num_workers=41,
        roi_size=(32, 32, 32),
        samples_per_image=200,
        limit_train_batches=1,
    )
    my_not_so_fast_training_dataloader = CroppedDataLoaderConfig(
        seed=my_seed,
        dict_keys=("img", "lab"),
        batch_size=32,
        num_workers=41,
        roi_size=(32, 32, 32),
        samples_per_image=200,
        limit_train_batches=20,
    )
    my_validation_dataloader = DataLoaderConfig(
        seed=my_seed,
        dict_keys=("img", "lab"),
        batch_size=1,
        num_workers=41,
    )
    my_testing_dataloader = DataLoaderConfig(
        seed=my_seed,
        dict_keys=("img", "lab"),
        batch_size=1,
        num_workers=41,
    )

    @pytest.fixture(scope="function")
    def assessment_config(self, tmp_path):
        return ExperimentConfiguration(
            assessment_method=AssessmentEnum("three_way_holdout"),
            train_val_test_ratio=(0.6, 0.2, 0.2),
            log_path=tmp_path,
            learning_rate=self.my_learning_rate,
            max_epochs=self.my_max_epochs,
            training_dataloader_config=self.my_not_so_fast_training_dataloader,
            validation_dataloader_config=self.my_validation_dataloader,
            testing_dataloader_config=self.my_testing_dataloader,
            n_internal_images=self.my_n_internal_images,
            experiment_id="00001",
            seed=self.my_seed,
        )

    @pytest.fixture(scope="function")
    def trainer(self, tmp_path, assessment_config):
        # logger will be set in the assessment classes
        return Trainer(
            logger=CSVLogger(tmp_path, self.my_experiment_id),
            max_epochs=assessment_config.data_config.max_epochs,
            limit_train_batches=assessment_config.data_config.limit_train_batches,
        )

    @pytest.fixture(scope="function")
    def test_trainer(self, holdout, assessment_config):
        # logger will be set in the assessment classes
        return Trainer(
            devices=1,
            num_nodes=1,
            logger=CSVLogger(
                holdout.root_log_dir, holdout.experiment_id, version="testing"
            ),
            max_epochs=assessment_config.data_config.max_epochs,
            limit_train_batches=assessment_config.data_config.limit_train_batches,
        )
