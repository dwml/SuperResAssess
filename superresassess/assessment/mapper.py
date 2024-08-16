from typing import Mapping

from superresassess.assessment.base import AssessmentMethod
from superresassess.assessment.enums import AssessmentEnum
from superresassess.three_way_holdout import ThreeWayHoldout
from superresassess.kfold_cross_validation import KFoldCrossValidation
from superresassess.nested_cross_validation import NestedCrossValidation


# This needs to be a separate file, since if this is placed in assessment_base
# we get a circular import
assessment_mapper: Mapping[AssessmentEnum, type[AssessmentMethod]] = {
    AssessmentEnum.three_way_holdout: ThreeWayHoldout,
    AssessmentEnum.k_fold_cross_validation: KFoldCrossValidation,
    AssessmentEnum.nested_cross_validation: NestedCrossValidation,
}
