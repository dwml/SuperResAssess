from abc import abstractmethod
from enum import Enum
from pathlib import Path
from typing import Optional

from lightning import LightningModule
from pydantic import BaseModel

from superresassess.model import ReCNNConfiguration
from superresassess.data import DataConfig, DataListType


def _check_ratios_with_folds(
    train_val_test_ratio: tuple[float, float, float], n_internal_images: int
):
    ratio_sum = sum(train_val_test_ratio)
    if ratio_sum != 1.0:
        raise ValueError(f"Train val test ratios don't sum to 1, but to {ratio_sum}")

    train_val_test_length = [
        ratio * n_internal_images for ratio in train_val_test_ratio
    ]

    train_val_test_integer = [length.is_integer() for length in train_val_test_length]

    if not all(train_val_test_integer):
        raise ValueError(
            "Train, validation and test ratios do not give integer"
            f"train val and test sets, but {train_val_test_length}"
        )


class AssessmentEnum(str, Enum):
    three_way_holdout = "three_way_holdout"
    nested_cross_validation = "nested_cross_validation"
    k_fold_cross_validation = "k_fold_cross_validation"


HoldoutProportion = tuple[float, float, float]


class AssessmentConfig(BaseModel):
    train_val_test_ratio: HoldoutProportion
    log_path: Path
    data_config: DataConfig
    n_internal_images: int
    experiment_id: str


class AssessmentMethod:
    def __init__(
        self,
        lightning_module: type[LightningModule],
        lightning_module_config: ReCNNConfiguration,
        dataset: DataListType,
        assessment_config: AssessmentConfig,
    ):
        self.lightning_module_type = lightning_module
        self.lightning_module_config = lightning_module_config
        self.dataset = dataset
        self.assessment_config = assessment_config
        self.best_model: Optional[LightningModule] = None
        self.assessment: Optional[float] = None
        self.n_epochs: Optional[int] = None

        # This might need to move to the assess method
        _check_ratios_with_folds(
            self.assessment_config.train_val_test_ratio,
            self.assessment_config.n_internal_images,
        )
        self._setup_file_splitting(
            self.assessment_config.train_val_test_ratio,
            self.assessment_config.n_internal_images,
        )

    @abstractmethod
    def _setup_file_splitting(
        self, train_val_test_ratio: tuple[float, float, float], n_internal_images: int
    ) -> None:
        pass

    @abstractmethod
    def assess(self) -> None:
        pass

    @abstractmethod
    def test(self) -> None:
        pass
