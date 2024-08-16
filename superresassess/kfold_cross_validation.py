from pathlib import Path
from typing import MutableSequence, TypeVar, Optional, Union

from monai.data.dataloader import DataLoader
import pandas as pd
import numpy as np
from lightning import LightningModule, Trainer
from lightning.pytorch.loggers import CSVLogger

from superresassess.assessment.base import AssessmentMethod
from superresassess.data import DataListType, _setup_seeded_dataloader
from superresassess.model import _setup_seeded_model


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


def _run_fold(
    log_path: Path,
    experiment_id: str,
    log_version: str,
    model: LightningModule,
    train_loader: DataLoader,
    train_list: DataListType,
    test_loader: DataLoader,
    test_list: DataListType,
    max_epochs: Optional[int] = None,
    limit_train_batches: Optional[Union[int, float]] = None,
):
    trainer = Trainer(
        logger=CSVLogger(
            log_path,
            experiment_id,
            version=log_version,
        ),
        max_epochs=max_epochs,
        limit_train_batches=limit_train_batches,
    )

    # logger can be of class Logger or None, so pyright complain because log_hyperparams
    # is not a member of None. I know the logger exists, since I assigned it the
    # previous line. Don't know how to fix it, so ignore it for now
    trainer.logger.log_hyperparams(  # type: ignore
        {
            "train_data": [str(datum) for datum in train_list],
            "val_data": [str(datum) for datum in test_list],
        }
    )

    trainer.fit(model, train_loader, test_loader)


class KFoldCrossValidation(AssessmentMethod):
    best_validation_loss = None
    best_model_path = None
    internal_testing_values = None
    external_testing_values = None
    num_epochs_for_refit = None

    def _setup_file_splitting(
        self, train_val_test_ratio: tuple[float, float, float], n_internal_images: int
    ):
        # separate internal and external images
        internal_data = self.dataset[:n_internal_images]
        self._external_test_data = self.dataset[n_internal_images:]

        # train_val_test_ratio is a tuple of floats summing to 1 since in k-fold
        # cross-validation we only use test and training, we can sum the first two
        # fractions
        train_fraction = train_val_test_ratio[0] + train_val_test_ratio[1]
        test_fraction = train_val_test_ratio[2]

        train_folds = train_fraction / test_fraction
        all_folds = train_folds + 1

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
        train_sets, test_sets = _prepare_kfold_sets(self._splits)
        log_versions = [f"assessment{ii}" for ii in range(len(self._splits))]
        for ii in range(len(self._splits)):
            current_log_version = log_versions[ii]
            current_train_set = train_sets[ii]
            current_test_set = test_sets[ii]
            instantiated_model = _setup_seeded_model(
                self.lightning_module_type,
                self.lightning_module_config,
                self.experiment_config.seed,
            )
            train_loader = _setup_seeded_dataloader(
                current_train_set,
                self.experiment_config.seed,
                self.experiment_config.data_config.dict_keys,
                self.experiment_config.data_config.train_batch_size,
                self.experiment_config.data_config.train_workers,
                random_cropping=True,
                cropping_size=self.experiment_config.data_config.train_roi_size,
                samples_per_image=self.experiment_config.data_config.samples_per_image,
            )
            test_loader = _setup_seeded_dataloader(
                current_test_set,
                self.experiment_config.seed,
                self.experiment_config.data_config.dict_keys,
                self.experiment_config.data_config.test_batch_size,
                self.experiment_config.data_config.test_workers,
            )
            _run_fold(
                log_path=self.experiment_config.log_path,
                experiment_id=self.experiment_config.experiment_id,
                log_version=current_log_version,
                model=instantiated_model,
                train_loader=train_loader,
                train_list=current_train_set,
                test_loader=test_loader,
                test_list=current_test_set,
                max_epochs=self.experiment_config.data_config.max_epochs,
                limit_train_batches=self.experiment_config.data_config.limit_train_batches,
            )

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
        train_set = [j for i in self._splits for j in i]
        instantiated_model = _setup_seeded_model(
            self.lightning_module_type,
            self.lightning_module_config,
            self.experiment_config.seed,
        )
        train_loader = _setup_seeded_dataloader(
            train_set,
            self.experiment_config.seed,
            self.experiment_config.data_config.dict_keys,
            self.experiment_config.data_config.train_batch_size,
            self.experiment_config.data_config.train_workers,
            random_cropping=True,
            cropping_size=self.experiment_config.data_config.train_roi_size,
            samples_per_image=self.experiment_config.data_config.samples_per_image,
        )

        refit_trainer = Trainer(
            logger=CSVLogger(
                self.experiment_config.log_path,
                self.experiment_config.experiment_id,
                version="refit",
            ),
            min_epochs=self.num_epochs_for_refit,
            max_epochs=self.num_epochs_for_refit,
            limit_train_batches=self.experiment_config.data_config.limit_train_batches,
        )

        refit_trainer.fit(instantiated_model, train_loader)
        if refit_trainer.is_global_zero:
            self.best_model_path = (
                self.experiment_config.log_path.joinpath(
                    self.experiment_config.experiment_id
                )
                .joinpath("refit")
                .joinpath("refit.pt")
            )

            refit_trainer.save_checkpoint(self.best_model_path)

    def test(self):
        self._refit()
        external_test_dataloader = _setup_seeded_dataloader(
            self._external_test_data,
            seed=self.experiment_config.seed,
            dict_keys=self.experiment_config.data_config.dict_keys,
            batch_size=self.experiment_config.data_config.test_batch_size,
            num_workers=self.experiment_config.data_config.test_workers,
        )

        trainer = Trainer(
            devices=1,
            num_nodes=1,
            logger=CSVLogger(
                self.experiment_config.log_path,
                self.experiment_config.experiment_id,
                version="testing",
            ),
        )

        if not self.best_model_path:
            raise AttributeError(
                "Best model path is not set. Probably because the model wasn't properly"
                " refit using the KFoldCrossValidation(...)._refit method"
            )
        testing_values = trainer.test(
            self.lightning_module_type.load_from_checkpoint(
                self.best_model_path, configuration=self.lightning_module_config
            ),
            external_test_dataloader,
        )
        self.external_testing_values = testing_values[0]["test_loss"]
