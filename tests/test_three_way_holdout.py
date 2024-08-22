import pytest

from superresassess.model import LitReCNN
from superresassess.three_way_holdout import ThreeWayHoldout

from .test_assessment_methods import AssessmentTestSetup


class TestThreeWayHoldout(AssessmentTestSetup):
    @pytest.fixture
    def holdout(self, mock_recnn_config, mock_data_with_external, experiment_config):
        return ThreeWayHoldout(
            LitReCNN,
            mock_recnn_config,
            mock_data_with_external,
            experiment_config,
        )

    @pytest.mark.slow
    def test_assessment(self, holdout):
        holdout.assess()

        assert (
            holdout.experiment_config.log_path.joinpath(
                holdout.experiment_config.experiment_id
            )
            .joinpath("validation")
            .joinpath("metrics.csv")
            .exists()
        )

    def test_testing_raises_error_without_assess(self, holdout):
        with pytest.raises(AttributeError):
            holdout.test()

    @pytest.mark.slow
    def test_internal_and_external_test_values(
        self, mock_recnn_config, mock_data_with_external, experiment_config
    ):
        holdout = ThreeWayHoldout(
            LitReCNN,
            mock_recnn_config,
            mock_data_with_external,
            experiment_config,
        )
        holdout.assess()

        holdout.test()

        assert holdout.internal_testing_loss == pytest.approx(0.166, abs=1e-3)
        assert holdout.external_testing_loss == pytest.approx(0.166, abs=1e-3)
