from enum import Enum
from pathlib import Path
from typing import Optional

from lightning import LightningModule
from lightning.pytorch.loggers import CSVLogger
from pydantic import BaseModel
from torch.utils.data import Dataset

from superresassess.validation import Validation, ValidationConfig
from superresassess.model import ReCNNConfiguration


class AssessmentEnum(str, Enum):
    three_way_holdout = "three_way_holdout"
    nested_cross_validation = "nested_cross_validation"
    k_fold_cross_validation = "k_fold_cross_validation"


HoldoutProportion = tuple[float, float, float]


class AssessmentConfig(BaseModel):
    train_val_test_ratio: HoldoutProportion
    validation_config: ValidationConfig
    test_batch_size: int
    test_workers: int
    n_internal_images: int


class ThreeWayHoldout:
    def __init__(
        self,
        lightning_module: type[LightningModule],
        lightning_module_config: ReCNNConfiguration,
        dataset: Dataset,
        assessment_config: AssessmentConfig,
        root_log_dir: Path,
        experiment_id: str,
    ):
        self._method = AssessmentEnum.three_way_holdout
        self.lightning_module_type = lightning_module
        self.lightning_module_config = lightning_module_config
        self.dataset = dataset
        self.assessment_config = assessment_config
        self.root_log_dir = root_log_dir
        self.experiment_id = experiment_id
        self.best_model: Optional[LightningModule] = None
        self.assessment: Optional[float] = None
        self.n_epochs: Optional[int] = None

        self._split_data(
            self.assessment_config.train_val_test_ratio,
            self.assessment_config.n_internal_images,
        )

    def _split_data(
        self, train_val_test_ratio: tuple[float, float, float], n_internal_images: int
    ) -> None:
        """For now I assume that the dataset is already randomly sampled from a bigger
        dataset, randomizing again thus makes no sense."""
        internal_images = self.dataset[:n_internal_images]
        self.external_test_data = self.dataset[n_internal_images:]

        ratio_sum = sum(list(train_val_test_ratio))
        if ratio_sum != 1.0:
            raise ValueError(
                f"Train val test ratios don't sum to 1, but to {ratio_sum}"
            )
        len_dataset = len(internal_images)
        train_size = int(train_val_test_ratio[0] * len_dataset)
        val_size = int(train_val_test_ratio[1] * len_dataset)
        self.train_data = internal_images[:train_size]
        self.val_data = internal_images[train_size : train_size + val_size]
        self.internal_test_data = internal_images[train_size + val_size :]

    def assess(self) -> None:
        logger = CSVLogger(self.root_log_dir)
        validator = Validation(
            self.lightning_module_type,
            self.lightning_module_config,
            logger=logger,
            train_data=self.train_data,
            val_data=self.val_data,
            validation_config=self.assessment_config.validation_config,
        )
        validator.validate()
        testing_values = validator.test(
            self.internal_test_data,
            self.external_test_data,
            batch_size=self.assessment_config.test_batch_size,
            num_workers=self.assessment_config.test_workers,
        )
        self.internal_testing_values = testing_values[0]["test_loss/dataloader_idx_0"]
        self.external_testing_values = testing_values[1]["test_loss/dataloader_idx_1"]
