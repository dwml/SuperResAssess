import pytest

from superresassess.assessment.enums import AssessmentEnum
from superresassess.experiments import ExperimentConfiguration
from superresassess.kfold_cross_validation import KFoldCrossValidation
from superresassess.model import LitReCNN
from .conftest import N_INTERNAL_IMAGES, SEED


@pytest.fixture
def assessment_config(tmp_path, cropped_dataloader_config, dataloader_config):
    return ExperimentConfiguration(
        assessment_method=AssessmentEnum("k_fold_cross_validation"),
        train_val_test_ratio=(0.6, 0.2, 0.2),
        log_path=tmp_path,
        learning_rate=1e-3,
        max_epochs=2,
        training_dataloader_config=cropped_dataloader_config,
        validation_dataloader_config=dataloader_config,
        testing_dataloader_config=dataloader_config,
        n_internal_images=N_INTERNAL_IMAGES,
        experiment_id="00001",
        seed=SEED,
    )


class TestKFoldCrossValidation:
    @pytest.mark.parametrize(
        "train_val_test_ratio, split_length",
        [((0.6, 0.2, 0.2), 5), ((0.8, 0.1, 0.1), 10)],
    )
    def test_folds(
        self,
        mock_recnn_config,
        mock_data_with_external,
        assessment_config,
        train_val_test_ratio,
        split_length,
    ):
        assessment_config.train_val_test_ratio = train_val_test_ratio
        cross_validation = KFoldCrossValidation(
            LitReCNN,
            mock_recnn_config,
            mock_data_with_external,
            assessment_config,
        )
        assert len(cross_validation._splits) == split_length

    @pytest.mark.slow
    def test_cross_validation(
        self, mock_recnn_config, mock_data_with_external, assessment_config
    ):
        assessment_config.log_path
        cross_validation = KFoldCrossValidation(
            LitReCNN,
            mock_recnn_config,
            mock_data_with_external,
            assessment_config,
        )

        experiment_log_path = assessment_config.log_path.joinpath(
            assessment_config.experiment_id
        )

        cross_validation.assess()

        assessments = [f"assessment{ii}" for ii in range(5)]
        for assessment in assessments:
            assert (
                experiment_log_path.joinpath(assessment)
                .joinpath("metrics.csv")
                .is_file()
            )
