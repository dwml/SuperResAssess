from pathlib import Path

import pytest

from superresassess.assessment_methods import (
    AssessmentConfig,
    AssessmentEnum,
    KFoldCrossValidation,
    ThreeWayHoldout,
)
from superresassess.data import ImageDatasetd, get_image_loader
from superresassess.model import LitReCNN
from superresassess.validation import ValidationConfig

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

    @pytest.fixture
    def holdout(self, mock_recnn_config, mock_data_with_external, tmp_path):
        return ThreeWayHoldout(
            LitReCNN,
            mock_recnn_config,
            mock_data_with_external,
            self.my_assessment_config,
            root_log_dir=tmp_path,
            experiment_id="00001",
        )

    @pytest.mark.slow
    def test_internal_and_external_test_values(self, holdout):
        holdout.assess()
        assert holdout.internal_testing_values == pytest.approx(0.166, abs=1e-3)
        assert holdout.external_testing_values == pytest.approx(0.166, abs=1e-3)
        assert holdout.internal_testing_values != holdout.external_testing_values


class TestKFoldCrossValidation:
    my_seed = 2024
    my_max_epochs = 5
    my_n_internal_images = 20
    my_fast_validation_config = ValidationConfig(
        max_epochs=1,
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

    @pytest.fixture
    def assessment_config(self):
        return AssessmentConfig(
            train_val_test_ratio=(0.6, 0.2, 0.2),
            validation_config=self.my_fast_validation_config,
            test_batch_size=1,
            test_workers=1,
            n_internal_images=self.my_n_internal_images,
        )

    @pytest.mark.parametrize(
        "train_val_test_ratio, split_length",
        [((0.6, 0.2, 0.2), 5), ((0.8, 0.1, 0.1), 10)],
    )
    def test_folds(
        self,
        mock_recnn_config,
        mock_data_with_external,
        assessment_config,
        tmp_path,
        train_val_test_ratio,
        split_length,
    ):
        assessment_config.train_val_test_ratio = train_val_test_ratio
        cross_validation = KFoldCrossValidation(
            LitReCNN,
            mock_recnn_config,
            mock_data_with_external,
            assessment_config,
            root_log_dir=tmp_path,
            experiment_id="00001",
        )
        assert len(cross_validation._splits) == split_length

    @pytest.mark.parametrize(
        "train_val_test_ratio",
        [(0.62, 0.1, 0.28), (0.8, 0.05, 0.15)],
    )
    def test_failing_train_val_test_ratio(
        self,
        mock_recnn_config,
        mock_data_with_external,
        assessment_config,
        tmp_path,
        train_val_test_ratio,
    ):
        assessment_config.train_val_test_ratio = train_val_test_ratio
        with pytest.raises(ValueError):
            _ = KFoldCrossValidation(
                LitReCNN,
                mock_recnn_config,
                mock_data_with_external,
                assessment_config,
                root_log_dir=tmp_path,
                experiment_id="00001",
            )

    @pytest.mark.slow
    def test_cross_validation(
        self, mock_recnn_config, mock_data_with_external, assessment_config, tmp_path
    ):
        experiment_id = "00001"
        cross_validation = KFoldCrossValidation(
            LitReCNN,
            mock_recnn_config,
            mock_data_with_external,
            assessment_config,
            root_log_dir=tmp_path,
            experiment_id=experiment_id,
        )
        cross_validation.assess()
        experiment_log_path = tmp_path.joinpath(experiment_id)
        assert (
            experiment_log_path.joinpath("iteration0").joinpath("metrics.csv").is_file()
        )
        assert (
            experiment_log_path.joinpath("iteration1").joinpath("metrics.csv").is_file()
        )
        assert (
            experiment_log_path.joinpath("iteration2").joinpath("metrics.csv").is_file()
        )
        assert (
            experiment_log_path.joinpath("iteration3").joinpath("metrics.csv").is_file()
        )
        assert (
            experiment_log_path.joinpath("iteration4").joinpath("metrics.csv").is_file()
        )
        assert isinstance(cross_validation.internal_testing_values, float)
        assert isinstance(cross_validation.external_testing_values, float)
