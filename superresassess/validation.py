from pathlib import Path
from typing import Mapping, Optional
import os

from lightning import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import Logger
from monai.data import DataLoader, Dataset, PatchDataset
from monai.transforms import RandSpatialCropSamplesd
from pydantic import BaseModel

from superresassess.data import get_image_loader
from superresassess.model import ReCNNConfiguration


class ValidationConfig(BaseModel):
    seed: int
    samples_per_image: int
    train_batch_size: int
    val_batch_size: int
    train_workers: int
    val_workers: int
    train_roi_size: tuple[int, int, int]
    max_epochs: int
    learning_rate: float
    dict_keys: tuple[str, str]
    limit_train_batches: Optional[int | float] = 1.0


class Validation:
    def __init__(
        self,
        model: type[LightningModule],
        model_config: ReCNNConfiguration,
        logger: Logger,
        train_data: list[dict[[str], Path]],
        validation_config: ValidationConfig,
        val_data: list[dict[[str], Path]] = None,
    ):
        self.model = model
        self.model_config = model_config
        self.logger = logger
        self.seed = validation_config.seed
        self.train_data = train_data
        self.val_data = val_data
        self.max_epochs = validation_config.max_epochs
        self.validation_config = validation_config
        self.instantiated_model: Optional[LightningModule] = None

        # Log training data

    def _setup(self) -> None:
        seed_everything(self.seed, workers=True)
        self.train_cropper = RandSpatialCropSamplesd(
            self.validation_config.dict_keys,
            self.validation_config.train_roi_size,
            self.validation_config.samples_per_image,
        )
        self.instantiated_model = self.model(self.model_config)
        self.loader = get_image_loader(dict_keys=self.validation_config.dict_keys)
        train_images = Dataset(self.train_data, transform=self.loader)
        train_patches = PatchDataset(
            train_images,
            patch_func=self.train_cropper,
            samples_per_image=self.validation_config.samples_per_image,
        )
        self.train_loader = DataLoader(
            train_patches,
            batch_size=self.validation_config.train_batch_size,
            num_workers=self.validation_config.train_workers,
        )
        if self.val_data:
            val_images = Dataset(
                self.val_data,
                transform=self.loader,
            )
            self.val_loader = DataLoader(
                val_images,
                batch_size=self.validation_config.val_batch_size,
                num_workers=self.validation_config.val_workers,
            )

    def train(self, max_epochs: int) -> None:
        self._setup()
        self.trainer = Trainer(
            max_epochs=max_epochs,
            logger=self.logger,
            deterministic=True,
            limit_train_batches=self.validation_config.limit_train_batches,
        )
        self.trainer.logger.log_hyperparams(
            {"train_data": [str(datum) for datum in self.train_data]}
        )
        self.trainer.fit(self.instantiated_model, self.train_loader)

        self.best_model_path = os.path.join(self.logger.log_dir, "retrained_model.pt")

        self.trainer.save_checkpoint(self.best_model_path)

    def validate(self) -> None:
        self._setup()
        self.trainer = Trainer(
            max_epochs=self.max_epochs,
            logger=self.logger,
            deterministic=True,
            limit_train_batches=self.validation_config.limit_train_batches,
            callbacks=[
                EarlyStopping(monitor="validation_loss", mode="min", patience=15),
                ModelCheckpoint(
                    dirpath=self.logger.log_dir,
                    filename="{epoch}-{validation_loss:0.2f}",
                    monitor="validation_loss",
                    save_top_k=3,
                    mode="min",
                ),
            ],
        )
        self.trainer.logger.log_hyperparams(
            {
                "train_data": [str(datum) for datum in self.train_data],
                "val_data": [str(datum) for datum in self.val_data],
            }
        )
        self.trainer.fit(
            self.instantiated_model,
            train_dataloaders=self.train_loader,
            val_dataloaders=self.val_loader,
        )

        self.best_validation_loss = self.trainer.checkpoint_callback.best_model_score
        self.best_model_path = Path(self.trainer.checkpoint_callback.best_model_path)

    def test(
        self,
        external_test_data: list[dict[[str], Path]],
        batch_size: int,
        num_workers: int,
        internal_test_data: list[dict[[str], Path]] = None,
    ) -> list[Mapping[str, float]]:
        """Test method from the trainer returns a list of mappings between strings and
        floats, here we assume that two dataloaders are used, one internal test set and
        one external test set. The testing value is list of dicts that map the name of
        the loss to its value."""
        external_test_images = Dataset(external_test_data, transform=self.loader)
        external_test_loader = DataLoader(external_test_images, batch_size=batch_size)

        dataloaders = [external_test_loader]
        if internal_test_data:
            internal_test_images = Dataset(internal_test_data, transform=self.loader)
            internal_test_loader = DataLoader(
                internal_test_images, batch_size=batch_size
            )
            dataloaders = [internal_test_loader, external_test_loader]

        testing_values = self.trainer.test(
            model=self.model.load_from_checkpoint(
                self.best_model_path, configuration=self.model_config
            ),
            dataloaders=dataloaders,
        )
        return testing_values
