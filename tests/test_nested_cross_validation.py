import pytest

from superresassess.assessment_base import DataConfig, AssessmentConfig
from superresassess.nested_cross_validation import NestedCrossValidation
from superresassess.model import LitReCNN


class TestNestedCrossValidation:
    my_seed = 2024
    my_max_epochs = 5
    my_n_internal_images = 20
    my_fast_validation_config = DataConfig(
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
    my_not_so_fast_validation_config = DataConfig(
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

    @pytest.fixture
    def nested(
        self, mock_recnn_config, mock_data_with_external, assessment_config, tmp_path
    ):
        experiment_id = "00001"
        return NestedCrossValidation(
            LitReCNN,
            mock_recnn_config,
            mock_data_with_external,
            assessment_config,
            root_log_dir=tmp_path,
            experiment_id=experiment_id,
        )

    def test_nested_cross_validation(self, nested):
        nested.assess()
        assert nested
