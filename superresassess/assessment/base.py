from abc import abstractmethod
from typing import Optional

from lightning import LightningModule

from superresassess.model import ReCNNConfiguration
from superresassess.data import DataListType
from superresassess.experiments import ExperimentConfiguration


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
        lightning_module: type[LightningModule],
        lightning_module_config: ReCNNConfiguration,
        dataset: DataListType,
        experiment_config: ExperimentConfiguration,
    ):
        self.lightning_module_type = lightning_module
        self.lightning_module_config = lightning_module_config
        self.dataset = dataset
        self.experiment_config = experiment_config
        self.best_model: Optional[LightningModule] = None
        self.assessment: Optional[float] = None
        self.n_epochs: Optional[int] = None

        # This might need to move to the assess method
        _check_ratios_with_folds(
            self.experiment_config.train_val_test_ratio,
            self.experiment_config.n_internal_images,
        )
        self._setup_file_splitting(
            self.experiment_config.train_val_test_ratio,
            self.experiment_config.n_internal_images,
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
