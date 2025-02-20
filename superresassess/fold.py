from abc import abstractmethod
from typing import Callable, Protocol, Optional, Union, Any
from pathlib import Path
import gc

import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from monai.data.dataloader import DataLoader

from superresassess.data import DataListType


def check_version_attribute_exists(func: Callable) -> Callable:
    def _return_func(*args, **kwargs):
        if not args[0]._version:
            raise AttributeError("Version not set, please set version first")

        return func(*args, **kwargs)

    return _return_func


class LightningModuleProvider(Protocol):
    @abstractmethod
    def provide(self) -> LightningModule: ...


class DataLoaderProvider(Protocol):
    @abstractmethod
    def provide(self, data: DataListType) -> DataLoader: ...


class Fold:
    """A class that represents a single fold of training.

    This class represents a single fold of training as occurs in for example k-fold
    cross validation and nested cross validation. It can also be used in three-way
    holdout, which could be seen as a single fold of k-fold cross-validation. Since
    we would like the folds to start out the same way, both the model provider, and
    any dataloader provider should be seeded. It is up to the user to correctly setup
    these providers. IMPORTANT: before running any training/validation/testing the
    version has to be set. This is so that a unique logging path will be used every
    fold. The version will be set to None after any training/validation/testing.

    Attributes
    ----------
    seeded_model_provider: LightningModuleProvider
        class that provides the same model everytime its provide method is called
    seeded_training_provider: DataLoaderProvider
        class that provides the same dataloader everytime its provide method is called
    seeded_validation_provider: DataLoaderProvider
        class that provides the same dataloader everytime its provide method is called
    seeded_testing_provider: DataLoaderProvider
        class that provides the same dataloader everytime its provide method is called
    log_path: Path
        path to store the log files
    experiment_id: str
        this will be used to create the subdirectory for the experiment
    max_epochs: int
        how many epochs will be ran
    version: str
        this will be used to create a subdirectory in the experiment_id subdirectory
        (will be set to None after any training/validation/testing)
    limit_train_batches: int | float
        mainly used for testing purposes, to limit the number of train batches and
        thus make testing shorter in time.
    trainer: Trainer
        lightning trainer class.

    Methods
    -------
    train_model(train_data)
    train_and_validate_model(train_data, validation_data)
    test_model(test_data)
    """

    def __init__(
        self,
        seeded_model_provider: LightningModuleProvider,
        seeded_training_provider: DataLoaderProvider,
        seeded_validation_provider: DataLoaderProvider,
        seeded_testing_provider: DataLoaderProvider,
        log_path: Path,
        experiment_id: str,
        max_epochs: Optional[int] = None,
        limit_train_batches: Optional[Union[int, float]] = None,
    ):
        self.seeded_model_provider = seeded_model_provider
        self.seeded_training_provider = seeded_training_provider
        self.seeded_validation_provider = seeded_validation_provider
        self.seeded_testing_provider = seeded_testing_provider
        self.log_path = log_path
        self.experiment_id = experiment_id
        self.max_epochs = max_epochs
        self.limit_train_batches = limit_train_batches
        self.trainer = Trainer()
        self._version = None
        self._logger = None
        self._min_epochs = 0

    @property
    def version(self) -> Union[str, None]:
        return self._version

    @property
    def min_epochs(self) -> int:
        return self._min_epochs

    @min_epochs.setter
    def min_epochs(self, min_epochs: int) -> None:
        self._min_epochs = min_epochs

    @version.setter
    def version(self, version: str) -> None:
        self._version = version
        self._logger = CSVLogger(self.log_path, self.experiment_id, self._version)
        self._checkpoint_callback = ModelCheckpoint(
            self.log_path.joinpath(self.experiment_id).joinpath(self._version),
            monitor="validation_loss",
        )
        self._early_stopping_callback = EarlyStopping(
            monitor="validation_loss", min_delta=0.1, patience=10
        )

    @check_version_attribute_exists
    def train_model(self, train_data: DataListType, checkpoint_path: Path) -> None:
        model = self.seeded_model_provider.provide()
        train_loader = self.seeded_training_provider.provide(train_data)
        self.trainer = Trainer(
            logger=self._logger,
            min_epochs=self.min_epochs,
            max_epochs=self.max_epochs,
            limit_train_batches=self.limit_train_batches,
        )
        self.trainer.logger.log_hyperparams(
            {
                "train_data": [str(datum) for datum in train_data],
            }
        )
        self.trainer.fit(model, train_loader)
        self.trainer.save_checkpoint(checkpoint_path)
        self._version = None

        # Fix memory issues
        self.trainer = None
        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()

    @check_version_attribute_exists
    def train_and_validate_model(
        self, train_data: DataListType, validation_data: DataListType
    ) -> tuple[Any, Any]:
        model = self.seeded_model_provider.provide()
        train_loader = self.seeded_training_provider.provide(train_data)
        validation_loader = self.seeded_validation_provider.provide(validation_data)

        self.trainer = Trainer(
            logger=self._logger,
            callbacks=[self._checkpoint_callback, self._early_stopping_callback],
            min_epochs=self.min_epochs,
            max_epochs=self.max_epochs,
            limit_train_batches=self.limit_train_batches,
        )
        self.trainer.logger.log_hyperparams(
            {
                "train_data": [str(datum) for datum in train_data],
                "val_data": [str(datum) for datum in validation_data],
            }
        )
        self.trainer.fit(model, train_loader, validation_loader)

        self.trainer.strategy.barrier()

        best_validation_loss = self.trainer.checkpoint_callback.best_model_score
        best_model_path = self.trainer.checkpoint_callback.best_model_path
        self._version = None

        # Fix memory issues
        self.trainer = None
        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()

        return (best_validation_loss, best_model_path)

    @check_version_attribute_exists
    def test_model(self, checkpoint_path: Path, test_data: DataListType) -> float:
        model = self.seeded_model_provider.provide()
        test_loader = self.seeded_testing_provider.provide(test_data)
        self.trainer = Trainer(
            devices=1,
            num_nodes=1,
            logger=self._logger,
        )
        self.trainer.logger.log_hyperparams(
            {
                "test_data": [str(datum) for datum in test_data],
            }
        )
        test_result: float = self.trainer.test(
            model=model, dataloaders=test_loader, ckpt_path=checkpoint_path
        )[0]["test_loss"]  # type: ignore
        self._version = None
        # Fix memory issues
        self.trainer = None
        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()
        return test_result
