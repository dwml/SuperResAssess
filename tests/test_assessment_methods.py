from superresassess.assessment_methods import (
    AssessmentEnum,
    ThreeWayHoldout,
    AssessmentConfig,
)
from superresassess.model import LitReCNN
from superresassess.data import ImageDatasetd, get_image_loader
from superresassess.validation import ValidationConfig
import pytest
from pathlib import Path

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


class TestThreeWayHoldout:
    my_seed = 2024
    my_max_epochs = 5
    my_n_internal_images = 20
    my_fast_validation_config = ValidationConfig(
        max_epochs=my_max_epochs,
        seed=my_seed,
        samples_per_image=200,
        train_batch_size=32,
        val_batch_size=1,
        train_workers=1,
        val_workers=1,
        train_roi_size=(32, 32, 32),
        learning_rate=1e-3,
        dict_keys=("img", "lab"),
        limit_train_batches=1,
    )
    my_not_so_fast_validation_config = ValidationConfig(
        max_epochs=my_max_epochs,
        seed=my_seed,
        samples_per_image=200,
        train_batch_size=32,
        val_batch_size=1,
        train_workers=1,
        val_workers=1,
        train_roi_size=(32, 32, 32),
        learning_rate=1e-3,
        dict_keys=("img", "lab"),
        limit_train_batches=10,
    )
    my_assessment_config = AssessmentConfig(
        train_val_test_ratio=(0.6, 0.2, 0.2),
        validation_config=my_not_so_fast_validation_config,
        test_batch_size=1,
        test_workers=1,
        n_internal_images=my_n_internal_images,
    )

    @pytest.mark.slow
    def test_internal_and_external_test_values(
        self, mock_recnn_config, mock_data_with_external, tmp_path
    ):
        holdout = ThreeWayHoldout(
            LitReCNN,
            mock_recnn_config,
            mock_data_with_external,
            self.my_assessment_config,
            root_log_dir=tmp_path,
            experiment_id="00001",
        )
        holdout.assess()
        assert holdout.internal_testing_values == pytest.approx(0.166, abs=1e-3)
        assert holdout.external_testing_values == pytest.approx(0.166, abs=1e-3)
        assert holdout.internal_testing_values != holdout.external_testing_values
