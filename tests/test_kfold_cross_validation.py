import os
from threading import local

import pytest

from superresassess.assessment_base import AssessmentConfig
from superresassess.kfold_cross_validation import KFoldCrossValidation
from superresassess.model import LitReCNN
from superresassess.data import DataConfig

DATA_CONFIG = DataConfig(
    max_epochs=2,
    seed=2024,
    samples_per_image=200,
    train_batch_size=32,
    val_batch_size=1,
    test_batch_size=1,
    train_workers=41,
    val_workers=41,
    test_workers=1,
    train_roi_size=(32, 32, 32),
    learning_rate=1e-3,
    dict_keys=("img", "lab"),
    limit_train_batches=20,
)

N_INTERNAL_IMAGES = 20


@pytest.fixture
def assessment_config(tmp_path):
    return AssessmentConfig(
        train_val_test_ratio=(0.6, 0.2, 0.2),
        log_path=tmp_path,
        data_config=DATA_CONFIG,
        n_internal_images=N_INTERNAL_IMAGES,
        experiment_id="00001",
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

        # The assess method above, uses data distributed parallelization to use all GPUs
        # in the machine. It does this by creating separated processes and sending them
        # individually to each GPU. The downside is that the processes continue
        # executing the code. The process with local rank 0 is the main process, that
        # is way we exit all processes that do not have local rank 0
        local_rank_key = "LOCAL_RANK"
        if local_rank_key in os.environ.keys():
            if int(os.environ[local_rank_key]) > 0:
                pytest.exit("Not main process, exiting...")

        assert (
            experiment_log_path.joinpath("assessment0")
            .joinpath("metrics.csv")
            .is_file()
        )
        assert (
            experiment_log_path.joinpath("assessment1")
            .joinpath("metrics.csv")
            .is_file()
        )
        assert (
            experiment_log_path.joinpath("assessment2")
            .joinpath("metrics.csv")
            .is_file()
        )
        assert (
            experiment_log_path.joinpath("assessment3")
            .joinpath("metrics.csv")
            .is_file()
        )
        assert (
            experiment_log_path.joinpath("assessment4")
            .joinpath("metrics.csv")
            .is_file()
        )
