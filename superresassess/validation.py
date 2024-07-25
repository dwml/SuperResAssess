from pathlib import Path
from lightning.pytorch.loggers import Logger
from lightning import LightningModule, seed_everything
from superresassess.model import ReCNNConfiguration
from typing import Optional


class Validation:
    def __init__(
        self,
        model: type[LightningModule],
        config: ReCNNConfiguration,
        logger: Logger,
        seed: int,
        train_data: list[dict[[str], Path]],
        val_data: list[dict[[str], Path]],
    ):
        self.model = model
        self.config = config
        self.logger = logger
        self.seed = seed
        self.train_data = train_data
        self.val_data = val_data
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
        seed_everything(self.seed)
        self.instantiated_model = self.model(self.config)
