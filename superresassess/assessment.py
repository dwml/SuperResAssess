from typing import Mapping
from superresassess.assessment_base import AssessmentEnum, AssessmentMethod
from superresassess.three_way_holdout import ThreeWayHoldout
from superresassess.kfold_cross_validation import KFoldCrossValidation
from superresassess.nested_cross_validation import NestedCrossValidation


assessment_mapper: Mapping[AssessmentEnum, type[AssessmentMethod]] = {
    AssessmentEnum.three_way_holdout: ThreeWayHoldout,
    AssessmentEnum.k_fold_cross_validation: KFoldCrossValidation,
    AssessmentEnum.nested_cross_validation: NestedCrossValidation,
}
