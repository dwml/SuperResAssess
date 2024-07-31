from pathlib import Path
from lightning.pytorch.loggers import Logger
from lightning import LightningModule, seed_everything, Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from superresassess.model import ReCNNConfiguration
from superresassess.data import get_image_loader
from typing import Optional
from pydantic import BaseModel
from monai.data import DataLoader, PatchDataset, Dataset
from monai.transforms import RandSpatialCropSamplesd


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


class Validation:
    def __init__(
        self,
        model: type[LightningModule],
        model_config: ReCNNConfiguration,
        logger: Logger,
        train_data: list[dict[[str], Path]],
        val_data: list[dict[[str], Path]],
        validation_config: ValidationConfig,
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
        self.logger.log_hyperparams(
            {
                "train_data": [str(datum) for datum in self.train_data],
                "val_data": [str(datum) for datum in self.val_data],
            }
        )
        self.logger.log_metrics({"test": 0.01})
        self.logger.finalize("success")

    def _setup(self) -> None:
        seed_everything(self.seed, workers=True)
        self.train_cropper = RandSpatialCropSamplesd(
            self.validation_config.dict_keys,
            self.validation_config.train_roi_size,
            self.validation_config.samples_per_image,
        )
        self.instantiated_model = self.model(self.model_config)
        loader = get_image_loader(dict_keys=self.validation_config.dict_keys)
        train_images = Dataset(self.train_data, transform=loader)
        train_patches = PatchDataset(
            train_images,
            patch_func=self.train_cropper,
            samples_per_image=self.validation_config.samples_per_image,
        )
        val_images = Dataset(
            self.val_data,
            transform=loader,
        )
        self.train_loader = DataLoader(
            train_patches,
            batch_size=self.validation_config.train_batch_size,
            num_workers=self.validation_config.train_workers,
        )
        self.val_loader = DataLoader(
            val_images,
            batch_size=self.validation_config.val_batch_size,
            num_workers=self.validation_config.val_workers,
        )

    def validate(self) -> None:
        self._setup()
        trainer = Trainer(
            max_epochs=self.max_epochs,
            logger=self.logger,
            deterministic=True,
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
        trainer.fit(
            self.instantiated_model,
            train_dataloaders=self.train_loader,
            val_dataloaders=self.val_loader,
        )

        self.best_validation_loss = trainer.checkpoint_callback.best_model_score
        self.best_model_path = Path(trainer.checkpoint_callback.best_model_path)
