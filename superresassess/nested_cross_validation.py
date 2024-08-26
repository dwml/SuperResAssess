from sre_constants import IN
from typing import MutableSequence

import numpy as np

from superresassess.assessment.results import (
    AssessResult,
    InternalTestResult,
)
from superresassess.kfold_cross_validation import (
    KFoldCrossValidation,
    _load_best_row_from_csv,
    _append_to_list_from_series,
)


class NestedCrossValidation(KFoldCrossValidation):
    def assess(self) -> AssessResult:
        log_versions = []
        epochs_outer = []
        for ii in range(len(self._splits)):
            split_copy = self._splits.copy()
            _ = split_copy.pop(ii)
            inner = split_copy
            inner_log_versions = []
            for jj in range(len(inner)):
                inner_copy = inner.copy()
                val_list = inner_copy.pop(jj)
                train_list = [i for j in inner_copy for i in j]
                self.fold.version = f"assessment{ii}_{jj}"
                inner_log_versions.append(self.fold.version)
                _, _ = self.fold.train_and_validate_model(train_list, val_list)

            epochs_inner: MutableSequence[int] = []
            losses_inner: MutableSequence[float] = []
            for log_version in inner_log_versions:
                metrics_path = (
                    self.experiment_config.log_path
                    / self.experiment_config.experiment_id
                    / log_version
                    / "metrics.csv"
                )
                series = _load_best_row_from_csv(metrics_path)
                epochs_inner = _append_to_list_from_series(
                    series, epochs_inner, "epoch"
                )
                losses_inner = _append_to_list_from_series(
                    series, losses_inner, "validation_loss"
                )

            # add one since counting starts at 1
            epochs_inner_array = np.array(epochs_inner, dtype=int) + 1
            epochs = int(np.median(epochs_inner_array))
            epochs_outer.append(epochs)
            self.fold.min_epochs = epochs
            self.fold.max_epochs = epochs
            self.fold.version = f"assessment{ii}"
            log_versions.append(self.fold.version)
            retrain_data = [i for j in inner for i in j]
            refit_path = (
                self.fold.log_path.joinpath(self.fold.experiment_id)
                .joinpath(self.fold.version)
                .joinpath("refit.ckpt")
            )
            self.fold.train_model(retrain_data, refit_path)

        self.num_epochs_for_refit = int(np.median(np.array(epochs_outer, dtype=int)))
        best_model_path = (
            self.experiment_config.log_path
            / self.experiment_config.experiment_id
            / "refit"
            / "refit.ckpt"
        )
        self.fold.version = "refit"
        self.fold.train_model([i for j in self._splits for i in j], best_model_path)

        return AssessResult(best_model_path=best_model_path, log_versions=log_versions)

    def internal_test(self, log_versions, best_model_path) -> InternalTestResult:
        losses_outer = []
        for ii in range(len(self._splits)):
            split_copy = self._splits.copy()
            outer = split_copy.pop(ii)
            log_version = log_versions[ii]
            refit_path = (
                self.fold.log_path.joinpath(self.fold.experiment_id)
                .joinpath(log_version)
                .joinpath("refit.ckpt")
            )
            self.fold.version = f"internal_test_assessment{ii}"
            losses_outer.append(self.fold.test_model(refit_path, outer))
        self.internal_testing_loss = np.mean(np.array(losses_outer, dtype=float))  # type: ignore
        return InternalTestResult(
            internal_testing_loss=self.internal_testing_loss,
            num_epochs_trained=-99,
        )
