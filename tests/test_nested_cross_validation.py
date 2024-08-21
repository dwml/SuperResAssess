import pytest

from superresassess.assessment.enums import AssessmentEnum
from superresassess.data import DataLoaderConfig, CroppedDataLoaderConfig
from superresassess.experiments import ExperimentConfiguration
from superresassess.nested_cross_validation import NestedCrossValidation
from superresassess.model import LitReCNN

N_INTERNAL_IMAGES = 20
SEED = 2024
TRAINING_DATALOADER_CONFIG = CroppedDataLoaderConfig(
    seed=SEED,
    dict_keys=("img", "lab"),
    batch_size=32,
    num_workers=41,
    roi_size=(32, 32, 32),
    samples_per_image=200,
    limit_train_batches=20,
)
VALIDATION_DATALOADER_CONFIG = DataLoaderConfig(
    seed=SEED,
    dict_keys=("img", "lab"),
    batch_size=1,
    num_workers=41,
)
TESTING_DATALOADER_CONFIG = DataLoaderConfig(
    seed=SEED,
    dict_keys=("img", "lab"),
    batch_size=1,
    num_workers=41,
)


@pytest.fixture
def assessment_config(tmp_path):
    return ExperimentConfiguration(
        assessment_method=AssessmentEnum("k_fold_cross_validation"),
        train_val_test_ratio=(0.6, 0.2, 0.2),
        log_path=tmp_path,
        learning_rate=1e-3,
        max_epochs=2,
        training_dataloader_config=TRAINING_DATALOADER_CONFIG,
        validation_dataloader_config=VALIDATION_DATALOADER_CONFIG,
        testing_dataloader_config=TESTING_DATALOADER_CONFIG,
        n_internal_images=N_INTERNAL_IMAGES,
        experiment_id="00001",
        seed=SEED,
    )


class TestNestedCrossValidation:
    @pytest.fixture
    def nested(
        self,
        mock_recnn_config,
        mock_data_with_external,
        assessment_config,
    ):
        return NestedCrossValidation(
            LitReCNN,
            mock_recnn_config,
            mock_data_with_external,
            assessment_config,
        )

    def test_nested_cross_validation(self, nested):
        nested.assess()
        assert nested
