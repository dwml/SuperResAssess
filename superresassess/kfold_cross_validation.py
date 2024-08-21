from pathlib import Path
from typing import MutableSequence, TypeVar

import pandas as pd
import numpy as np

from superresassess.assessment.base import AssessmentMethod
from superresassess.data import DataListType


def _prepare_kfold_sets(
    splits: list[DataListType],
) -> tuple[list[DataListType], list[DataListType]]:
    train_sets: list[DataListType] = []
    test_sets: list[DataListType] = []
    for ii in range(len(splits)):
        splits_copy = splits.copy()
        test_sets.append(splits_copy.pop(ii))
        train_sets.append([j for i in splits_copy for j in i])
    return train_sets, test_sets


def _load_best_row_from_csv(
    csv: Path, key: str = "validation_loss", min: bool = True
) -> pd.Series:
    df = pd.read_csv(csv)
    if min:
        return df.iloc[df[key].idxmin()]
    return df.iloc[df[key].idxmax()]


T = TypeVar("T")


def _append_to_list_from_series(
    row: pd.Series, list_to_append: MutableSequence[T], key: str
) -> MutableSequence[T]:
    # it is unclear what type row will return. For now I don't know how to fix it, so
    # I'll ignore it.
    val: T = row[key]  # type: ignore
    list_to_append.append(val)
    return list_to_append


class KFoldCrossValidation(AssessmentMethod):
    best_validation_loss = None
    best_model_path = None
    internal_testing_values = None
    external_testing_values = None
    num_epochs_for_refit = None

    def _even_splitting_possible_error_check(self, number_of_folds: int) -> None:
        if not (self.experiment_config.n_internal_images % number_of_folds) == 0:
            raise ValueError(
                f"The {self.experiment_config.n_internal_images} internal testing"
                f" images cannot be separated in {number_of_folds} folds."
            )

    def _get_train_test_fraction(self) -> tuple[float, float]:
        # train_val_test_ratio is a tuple of floats summing to 1 since in k-fold
        # cross-validation we only use test and training, we can sum the first two
        # fractions
        train_fraction = (
            self.experiment_config.train_val_test_ratio[0]
            + self.experiment_config.train_val_test_ratio[1]
        )
        test_fraction = self.experiment_config.train_val_test_ratio[2]
        return train_fraction, test_fraction

    def _split_files(self):
        train_fraction, test_fraction = self._get_train_test_fraction()

        number_of_folds = int(train_fraction / test_fraction + 1)

        # Error checking
        self._even_splitting_possible_error_check(number_of_folds)

        fold_length = int(self.experiment_config.n_internal_images // number_of_folds)

        self._splits = [
            self._internal_images[i * fold_length : (i + 1) * fold_length]
            for i in range(number_of_folds)
        ]

    def assess(self) -> None:
        # list to keep track of the epochs and losses for every iteration
        train_sets, test_sets = _prepare_kfold_sets(self._splits)

        log_versions = []
        for ii in range(len(self._splits)):
            train_set_ii = train_sets[ii]
            test_set_ii = test_sets[ii]
            self.fold.version = f"assessment{ii}"
            log_versions.append(self.fold.version)

            self.fold.train_and_validate_model(train_set_ii, test_set_ii)

        epochs: MutableSequence[int] = []
        losses: MutableSequence[float] = []
        for log_version in log_versions:
            metrics_path = (
                self.experiment_config.log_path
                / self.experiment_config.experiment_id
                / log_version
                / "metrics.csv"
            )
            series = _load_best_row_from_csv(metrics_path)
            epochs = _append_to_list_from_series(series, epochs, "epoch")
            losses = _append_to_list_from_series(series, losses, "validation_loss")

        self.internal_testing_values = np.asarray(losses).mean()
        self.num_epochs_for_refit = int(np.median(np.asarray(epochs, dtype=int)))

    def _refit(self):
        if not self.num_epochs_for_refit:
            raise AttributeError(
                "Number of epochs not set, run"
                " KFoldCrossValidation(...).assess() before running"
                " KFoldCrossValidation(...).test()"
            )
        # Refit the model
        train_list = [j for i in self._splits for j in i]
        self.fold.min_epochs = self.num_epochs_for_refit
        self.fold.max_epochs = self.num_epochs_for_refit
        self.fold.version = "refit"
        self.fold.train_model(train_list)

        self.best_model_path = (
            self.experiment_config.log_path.joinpath(
                self.experiment_config.experiment_id
            )
            .joinpath("refit")
            .joinpath("refit.pt")
        )

        self.fold.trainer.save_checkpoint(self.best_model_path)

    def test(self):
        self._refit()
        if not self.best_model_path:
            raise AttributeError(
                "Best model path is not set. Probably because the model wasn't properly"
                " refit using the KFoldCrossValidation(...)._refit method"
            )

        testing_values = self.fold.test_model(
            self.best_model_path, self._external_test_data
        )
        self.external_testing_values = testing_values[0]["test_loss"]
