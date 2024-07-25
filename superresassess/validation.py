from pathlib import Path
from lightning.pytorch.loggers import Logger


class Validation:
    def __init__(
        self,
        logger: Logger,
        train_data: list[dict[[str], Path]],
        val_data: list[dict[[str], Path]],
    ):
        self.logger = logger
        self.train_data = train_data
        self.val_data = val_data

        # Log training data
        self.logger.log_hyperparams(
            {
                "train_data": [str(datum) for datum in self.train_data],
                "val_data": [str(datum) for datum in self.val_data],
            }
        )
        self.logger.log_metrics({"test": 0.01})
        self.logger.finalize("success")
