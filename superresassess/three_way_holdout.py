from pathlib import Path

from lightning import Trainer
from lightning.pytorch.loggers import CSVLogger

from superresassess.assessment.base import AssessmentMethod
from superresassess.data import _setup_seeded_dataloader
from superresassess.model import _setup_seeded_model


class ThreeWayHoldout(AssessmentMethod):
    best_validation_loss = None
    best_model_path = None
    internal_testing_values = None
    external_testing_values = None

    def _setup_file_splitting(
        self, train_val_test_ratio: tuple[float, float, float], n_internal_images: int
    ) -> None:
        """For now I assume that the dataset is already randomly sampled from a bigger
        dataset, randomizing again thus makes no sense."""
        internal_images = self.dataset[:n_internal_images]
        self._external_test_data = self.dataset[n_internal_images:]

        len_dataset = len(internal_images)
        train_size = int(train_val_test_ratio[0] * len_dataset)
        val_size = int(train_val_test_ratio[1] * len_dataset)
        self._train_data = internal_images[:train_size]
        self._val_data = internal_images[train_size : train_size + val_size]
        self._internal_test_data = internal_images[train_size + val_size :]

    def assess(self) -> None:
        trainer = Trainer(
            logger=CSVLogger(
                self.experiment_config.log_path,
                self.experiment_config.experiment_id,
                version="assessment",
            ),
            max_epochs=self.experiment_config.data_config.max_epochs,
            limit_train_batches=self.experiment_config.data_config.limit_train_batches,
        )
        instantiated_model = _setup_seeded_model(
            self.lightning_module_type,
            self.lightning_module_config,
            self.experiment_config.seed,
        )
        train_loader = _setup_seeded_dataloader(
            self._train_data,
            self.experiment_config.seed,
            self.experiment_config.data_config.dict_keys,
            self.experiment_config.data_config.train_batch_size,
            self.experiment_config.data_config.train_workers,
            random_cropping=True,
            cropping_size=self.experiment_config.data_config.train_roi_size,
            samples_per_image=self.experiment_config.data_config.samples_per_image,
        )
        val_loader = _setup_seeded_dataloader(
            self._val_data,
            self.experiment_config.seed,
            self.experiment_config.data_config.dict_keys,
            self.experiment_config.data_config.val_batch_size,
            self.experiment_config.data_config.val_workers,
        )

        if trainer.logger:
            trainer.logger.log_hyperparams(
                {
                    "train_data": [str(datum) for datum in self._train_data],
                    "val_data": [str(datum) for datum in self._val_data],
                }
            )

        trainer.fit(
            instantiated_model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )

        # Set up a barrier to make sure everything is finished
        trainer.strategy.barrier()

        # Checking for global zero makes sure that all spawned processes have finished
        if trainer.is_global_zero:
            self.best_validation_loss = trainer.checkpoint_callback.best_model_score  # type: ignore
            self.best_model_path = Path(trainer.checkpoint_callback.best_model_path)  # type: ignore

    def test(self) -> None:
        """Make sure to set devices and nodes to 1 in the trainer, since this makes for
        reporducible test data."""
        if not self.best_model_path:
            raise AttributeError(
                "Best model path is needed for testing, but does not exist. Run"
                " ThreeWayHoldout(...).assess() first."
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
        internal_test_dataloader = _setup_seeded_dataloader(
            self._internal_test_data,
            self.experiment_config.seed,
            self.experiment_config.data_config.dict_keys,
            self.experiment_config.data_config.test_batch_size,
            self.experiment_config.data_config.test_workers,
        )
        external_test_dataloader = _setup_seeded_dataloader(
            self._external_test_data,
            self.experiment_config.seed,
            self.experiment_config.data_config.dict_keys,
            self.experiment_config.data_config.test_batch_size,
            self.experiment_config.data_config.test_workers,
        )
        testing_values = trainer.test(
            model=self.lightning_module_type.load_from_checkpoint(
                self.best_model_path, configuration=self.lightning_module_config
            ),
            dataloaders=[internal_test_dataloader, external_test_dataloader],
        )

        # Set up a barrier to make sure everything is finished
        trainer.strategy.barrier()

        # Checking for global zero makes sure that only the main process
        # enters this clause
        if trainer.is_global_zero:
            self.internal_testing_values = testing_values[0][
                "test_loss/dataloader_idx_0"
            ]
            self.external_testing_values = testing_values[1][
                "test_loss/dataloader_idx_1"
            ]
