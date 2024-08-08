from abc import abstractmethod
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from lightning import LightningModule
from lightning.pytorch.loggers import CSVLogger
from pydantic import BaseModel
from torch.utils.data import Dataset

from superresassess.model import ReCNNConfiguration
from superresassess.validation import Validation, ValidationConfig


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


class AssessmentMethod:
    def __init__(
        self,
        lightning_module: type[LightningModule],
        lightning_module_config: ReCNNConfiguration,
        dataset: Dataset,
        assessment_config: AssessmentConfig,
        root_log_dir: Path,
        experiment_id: str,
    ):
        self.lightning_module_type = lightning_module
        self.lightning_module_config = lightning_module_config
        self.dataset = dataset
        self.assessment_config = assessment_config
        self.root_log_dir = root_log_dir
        self.experiment_id = experiment_id
        self.best_model: Optional[LightningModule] = None
        self.assessment: Optional[float] = None
        self.n_epochs: Optional[int] = None

        # This might need to move to the assess method
        self._setup_data(
            self.assessment_config.train_val_test_ratio,
            self.assessment_config.n_internal_images,
        )

    @abstractmethod
    def _setup_data(
        self, train_val_test_ratio: tuple[float, float, float], n_internal_images: int
    ) -> None:
        pass

    @abstractmethod
    def assess(self) -> None:
        pass


class ThreeWayHoldout(AssessmentMethod):
    _train_data = None
    _val_data = None
    _internal_test_data = None
    _external_test_data = None

    def _setup_data(
        self, train_val_test_ratio: tuple[float, float, float], n_internal_images: int
    ) -> None:
        """For now I assume that the dataset is already randomly sampled from a bigger
        dataset, randomizing again thus makes no sense."""
        internal_images = self.dataset[:n_internal_images]
        self._external_test_data = self.dataset[n_internal_images:]

        ratio_sum = sum(list(train_val_test_ratio))
        if ratio_sum != 1.0:
            raise ValueError(
                f"Train val test ratios don't sum to 1, but to {ratio_sum}"
            )
        len_dataset = len(internal_images)
        train_size = int(train_val_test_ratio[0] * len_dataset)
        val_size = int(train_val_test_ratio[1] * len_dataset)
        self._train_data = internal_images[:train_size]
        self._val_data = internal_images[train_size : train_size + val_size]
        self._internal_test_data = internal_images[train_size + val_size :]

    def assess(self) -> None:
        logger = CSVLogger(self.root_log_dir)
        validator = Validation(
            self.lightning_module_type,
            self.lightning_module_config,
            logger=logger,
            train_data=self._train_data,
            val_data=self._val_data,
            validation_config=self.assessment_config.validation_config,
        )
        validator.validate()
        testing_values = validator.test(
            internal_test_data=self._internal_test_data,
            external_test_data=self._external_test_data,
            batch_size=self.assessment_config.test_batch_size,
            num_workers=self.assessment_config.test_workers,
        )
        self.internal_testing_values = testing_values[0]["test_loss/dataloader_idx_0"]
        self.external_testing_values = testing_values[1]["test_loss/dataloader_idx_1"]


class KFoldCrossValidation(AssessmentMethod):
    _splits = None
    _external_test_data = None

    def _setup_data(
        self, train_val_test_ratio: tuple[float, float, float], n_internal_images: int
    ):
        # train_val_test_ratio is a tuple of floats summing to 1 since in k-fold
        # cross-validation we only use test and training, we can sum the first two
        # fractions
        train_fraction = train_val_test_ratio[0] + train_val_test_ratio[1]
        test_fraction = train_val_test_ratio[2]

        internal_data = self.dataset[:n_internal_images]
        self._external_test_data = self.dataset[n_internal_images:]

        train_folds = train_fraction / test_fraction
        all_folds = train_folds + 1

        if not all_folds.is_integer():
            raise ValueError(
                "The training fraction and the testing fraction does not result in"
                f" integer folds but in {all_folds} folds"
            )

        all_folds = int(all_folds)

        if not (n_internal_images % all_folds) == 0:
            raise ValueError(
                f"The {n_internal_images} internal testing images cannot be separated"
                f" in {all_folds} folds."
            )

        fold_length = int(n_internal_images // all_folds)

        self._splits = [
            internal_data[i * fold_length : (i + 1) * fold_length]
            for i in range(all_folds)
        ]

    def assess(self) -> None:
        # list to keep track of the epochs and losses for every iteration
        epochs = []
        losses = []
        for ii in range(len(self._splits)):
            train_set = self._splits.copy()
            test_set = train_set.pop(ii)

            # join lists of paths in train_set
            train_set = [j for i in train_set for j in i]

            logger = CSVLogger(
                save_dir=self.root_log_dir,
                name=str(self.experiment_id),
                version=f"iteration{ii}",
            )
            validator = Validation(
                self.lightning_module_type,
                self.lightning_module_config,
                logger=logger,
                train_data=train_set,
                val_data=test_set,
                validation_config=self.assessment_config.validation_config,
            )
            validator.validate()

            df = pd.read_csv(
                self.root_log_dir.joinpath(str(self.experiment_id))
                .joinpath(f"iteration{ii}")
                .joinpath("metrics.csv")
            )
            best_row = df.iloc[df["validation_loss"].idxmax()]
            losses.append(best_row["validation_loss"])
            epochs.append(best_row["epoch"])

        self.internal_testing_values = np.asarray(losses).mean()
        validator = Validation(
            self.lightning_module_type,
            self.lightning_module_config,
            logger=logger,
            train_data=train_set,
            validation_config=self.assessment_config.validation_config,
        )
        median_epochs = int(np.median(np.asarray(epochs)))
        validator.train(median_epochs)
        testing_values = validator.test(
            external_test_data=self._external_test_data,
            batch_size=self.assessment_config.test_batch_size,
            num_workers=self.assessment_config.test_workers,
        )
        self.external_testing_values = testing_values[0]["test_loss"]


class NestedCrossValidation(AssessmentMethod): ...


assessment_mapper = {
    AssessmentEnum.three_way_holdout: ThreeWayHoldout,
    AssessmentEnum.nested_cross_validation: NestedCrossValidation,
    AssessmentEnum.k_fold_cross_validation: KFoldCrossValidation,
}
