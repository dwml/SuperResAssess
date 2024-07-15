from enum import Enum


class AssessmentEnum(str, Enum):
    three_way_holdout = "three_way_holdout"
    nested_cross_validation = "nested_cross_validation"
    k_fold_cross_validation = "k_fold_cross_validation"
