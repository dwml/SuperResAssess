from pathlib import Path

from superresassess.assessment.base import AssessmentMethod


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

    def assess(self) -> None:
        self.fold.version = "validation"

        if self.fold.trainer.logger:
            self.fold.trainer.logger.log_hyperparams(
                {
                    "train_data": [str(datum) for datum in self._train_data],
                    "val_data": [str(datum) for datum in self._val_data],
                }
            )

        self.fold.train_and_validate_model(self._train_data, self._val_data)

        # Checking for global zero makes sure that all spawned processes have finished
        self.best_validation_loss = (
            self.fold.trainer.checkpoint_callback.best_model_score  # type: ignore
        )  # type: ignore
        self.best_model_path = Path(
            self.fold.trainer.checkpoint_callback.best_model_path  # type: ignore
        )  # type: ignore

    def test(self) -> None:
        """Make sure to set devices and nodes to 1 in the trainer, since this makes for
        reporducible test data."""
        if not self.best_model_path:
            raise AttributeError(
                "Best model path is needed for testing, but does not exist. Run"
                " ThreeWayHoldout(...).assess() first."
            )

        # internal testing
        self.fold.version = "internal_testing"
        self.internal_testing_loss = self.fold.test_model(
            self.best_model_path, self._internal_test_data
        )
        print(f"Self internal testing loss {self.internal_testing_loss}")

        # external testing
        self.fold.version = "external_testing"
        self.external_testing_loss = self.fold.test_model(
            self.best_model_path, self._external_test_data
        )
        print(f"Self external testing loss {self.external_testing_loss}")
