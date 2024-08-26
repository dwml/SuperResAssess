from pathlib import Path

from superresassess.assessment.base import AssessmentMethod
from superresassess.assessment.results import (
    AssessResult,
    InternalTestResult,
    ExternalTestResult,
)


class ThreeWayHoldout(AssessmentMethod):
    best_validation_loss = None
    best_model_path = None

    def _split_files(self) -> None:
        """For now I assume that the dataset is already randomly sampled from a bigger
        dataset, randomizing again thus makes no sense."""
        len_dataset = len(self._internal_images)
        train_size = int(self.experiment_config.train_val_test_ratio[0] * len_dataset)
        val_size = int(self.experiment_config.train_val_test_ratio[1] * len_dataset)
        self._train_data = self._internal_images[:train_size]
        self._val_data = self._internal_images[train_size : train_size + val_size]
        self._internal_test_data = self._internal_images[train_size + val_size :]

    def assess(self) -> AssessResult:
        self.fold.version = "validation"

        best_validation_loss, best_model_path = self.fold.train_and_validate_model(
            self._train_data, self._val_data
        )

        self.best_validation_loss = best_validation_loss
        self.best_model_path = best_model_path

        return AssessResult(
            best_model_path=self.best_model_path, log_versions=["validation"]
        )

    def internal_test(self, log_versions, best_model_path) -> InternalTestResult:
        self.fold.version = "internal_testing"
        self.internal_testing_loss = self.fold.test_model(
            best_model_path, self._internal_test_data
        )
        return InternalTestResult(
            internal_testing_loss=self.internal_testing_loss,
            num_epochs_trained=-99,
        )

    def external_test(self, best_model_path: Path) -> ExternalTestResult:
        self.fold.version = "external_testing"
        self.external_testing_loss = self.fold.test_model(
            best_model_path, self._external_test_data
        )
        return ExternalTestResult(external_testing_loss=self.external_testing_loss)
