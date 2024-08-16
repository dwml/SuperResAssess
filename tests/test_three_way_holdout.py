import pytest

from .test_assessment_methods import AssessmentTestSetup
from superresassess.three_way_holdout import ThreeWayHoldout
from superresassess.model import LitReCNN


class TestThreeWayHoldout(AssessmentTestSetup):
    @pytest.fixture
    def holdout(self, mock_recnn_config, mock_data_with_external, assessment_config):
        return ThreeWayHoldout(
            LitReCNN,
            mock_recnn_config,
            mock_data_with_external,
            assessment_config,
        )

    @pytest.mark.slow
    def test_assessment(self, holdout):
        holdout.assess()

        assert (
            holdout.assessment_config.log_path.joinpath(
                holdout.assessment_config.experiment_id
            )
            .joinpath("assessment")
            .joinpath("metrics.csv")
            .exists()
        )

    def test_testing_raises_error_without_assess(self, holdout):
        with pytest.raises(AttributeError):
            holdout.test()

    @pytest.mark.slow
    def test_internal_and_external_test_values(self, holdout):
        holdout.assess()

        holdout.test()

        assert holdout.internal_testing_values == pytest.approx(0.166, abs=1e-3)
        assert holdout.external_testing_values == pytest.approx(0.166, abs=1e-3)
