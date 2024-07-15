from superresassess.assessment_methods import AssessmentEnum

import pytest

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
