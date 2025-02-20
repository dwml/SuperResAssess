from abc import abstractmethod
from typing import Optional
from pathlib import Path

from lightning import LightningModule

from superresassess.assessment.results import (
    AssessResult,
    InternalTestResult,
    ExternalTestResult,
)
from superresassess.model import ReCNNConfiguration
from superresassess.data import DataListType
from superresassess.experiments import ExperimentConfiguration
from superresassess.seeded import (
    SeededDataLoaderProvider,
    SeededCroppedDataLoaderProvider,
    SeededModelProvider,
)
from superresassess.fold import Fold


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


class AssessmentMethod:
    def __init__(
        self,
        lightning_module_type: type[LightningModule],
        lightning_module_config: ReCNNConfiguration,
        dataset: DataListType,
        experiment_config: ExperimentConfiguration,
    ):
        self.experiment_config = experiment_config
        self.seeded_model_provider = SeededModelProvider(
            lightning_module_type, lightning_module_config, self.experiment_config.seed
        )
        self.dataset = dataset
        self.best_model: Optional[LightningModule] = None
        self.best_model_path: Optional[Path] = None
        self.assessment: Optional[float] = None
        self.n_epochs: Optional[int] = None
        self.internal_testing_loss: Optional[float] = None
        self.external_testing_loss: Optional[float] = None
        self.fold = Fold(
            seeded_model_provider=SeededModelProvider(
                lightning_module_type,
                lightning_module_config,
                self.experiment_config.seed,
            ),
            seeded_training_provider=SeededCroppedDataLoaderProvider(
                self.experiment_config.training_dataloader_config
            ),
            seeded_validation_provider=SeededDataLoaderProvider(
                self.experiment_config.validation_dataloader_config
            ),
            seeded_testing_provider=SeededDataLoaderProvider(
                self.experiment_config.testing_dataloader_config
            ),
            log_path=self.experiment_config.log_path,
            experiment_id=self.experiment_config.experiment_id,
            max_epochs=self.experiment_config.max_epochs,
            limit_train_batches=self.experiment_config.training_dataloader_config.limit_train_batches,
        )

        # This might need to move to the assess method
        _check_ratios_with_folds(
            self.experiment_config.train_val_test_ratio,
            self.experiment_config.n_internal_images,
        )
        self._set_internal_external_test_data()
        self._split_files()

    def _set_internal_external_test_data(self):
        self._internal_images = self.dataset[: self.experiment_config.n_internal_images]
        self._external_test_data = self.dataset[
            self.experiment_config.n_internal_images :
        ]

    @abstractmethod
    def _split_files(self) -> None:
        pass

    @abstractmethod
    def assess(self) -> AssessResult:
        pass

    @abstractmethod
    def internal_test(
        self, log_versions: list[str], best_model_path: Path
    ) -> InternalTestResult:
        pass

    @abstractmethod
    def external_test(self, best_model_path: Path) -> ExternalTestResult:
        pass
