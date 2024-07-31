from superresassess.validation import Validation, ValidationConfig
from superresassess.model import LitReCNN
from lightning.pytorch.loggers import CSVLogger
from lightning import LightningModule
from pathlib import Path
import yaml
import torch
import pytest


class TestValidation:
    my_seed = 2024
    my_max_epochs = 5
    my_not_so_fast_validation_config = ValidationConfig(
        max_epochs=my_max_epochs,
        seed=my_seed,
        samples_per_image=200,
        train_batch_size=32,
        val_batch_size=1,
        train_workers=1,
        val_workers=1,
        train_roi_size=(32, 32, 32),
        learning_rate=1e-3,
        dict_keys=("img", "lab"),
        limit_train_batches=10,
    )
    my_fast_validation_config = ValidationConfig(
        max_epochs=my_max_epochs,
        seed=my_seed,
        samples_per_image=200,
        train_batch_size=32,
        val_batch_size=1,
        train_workers=1,
        val_workers=1,
        train_roi_size=(32, 32, 32),
        learning_rate=1e-3,
        dict_keys=("img", "lab"),
        limit_train_batches=1,
    )

    @pytest.fixture(scope="function")
    def logger(self, tmp_path):
        return CSVLogger(save_dir=tmp_path, name="test", version="iteration1")

    @pytest.fixture(scope="function")
    def train_val_data(self, mock_data):
        train_val_test_split = (0.6, 0.2, 0.2)
        train_length = int(train_val_test_split[0] * len(mock_data))
        val_length = int(train_val_test_split[1] * len(mock_data))
        return (
            mock_data[:train_length],
            mock_data[train_length : train_length + val_length],
        )

    def test_logging(self, tmp_path, train_val_data, mock_recnn_config, logger):
        """The Validation class should log the filenames to be able to double check
        that the correct files were used."""
        # setup data and logger
        train_data, val_data = train_val_data

        # this should log the file
        validator = Validation(
            model=LitReCNN,
            model_config=mock_recnn_config,
            logger=logger,
            train_data=train_data,
            val_data=val_data,
            validation_config=self.my_fast_validation_config,
        )

        validator.validate()

        # Assert that contents are correct
        yaml_path = Path(logger.log_dir).joinpath("hparams.yaml")
        contents = yaml.safe_load(yaml_path.read_text())
        assert contents["train_data"] == [str(datum) for datum in train_data]
        assert contents["val_data"] == [str(datum) for datum in val_data]

    @pytest.mark.parametrize("seed", [2024, 2025, 2026])
    def test_model_initialization_is_deterministic(
        self, tmp_path, train_val_data, mock_recnn_config, seed, logger
    ):
        # setup data and logger
        train_data, val_data = train_val_data

        # Seting up the validators with the same seed should yield
        # models with the same initialized weights
        validator = Validation(
            model=LitReCNN,
            model_config=mock_recnn_config,
            logger=logger,
            train_data=train_data,
            val_data=val_data,
            validation_config=self.my_fast_validation_config,
        )
        validator._setup()
        validator2 = Validation(
            model=LitReCNN,
            model_config=mock_recnn_config,
            logger=logger,
            train_data=train_data,
            val_data=val_data,
            validation_config=self.my_fast_validation_config,
        )
        validator2._setup()

        # compare weights
        weights = validator.instantiated_model.state_dict()
        weights2 = validator.instantiated_model.state_dict()
        equality = [
            torch.equal(w1, w2) for w1, w2 in zip(weights.values(), weights2.values())
        ]
        assert all(equality)

    @pytest.mark.slow
    def test_validate_creates_new_model_instance(
        self, train_val_data, mock_recnn_config, logger
    ):
        # setup data, logger and validator
        train_data, val_data = train_val_data
        validator = Validation(
            model=LitReCNN,
            model_config=mock_recnn_config,
            logger=logger,
            train_data=train_data,
            val_data=val_data,
            validation_config=self.my_fast_validation_config,
        )

        # validate model
        validator.validate()

        # assert instantiated model is not None and is a LightningModule
        assert validator.instantiated_model is not None
        assert isinstance(validator.instantiated_model, LightningModule)

    @pytest.mark.slow
    def test_validate_gives_correct_validation_loss(
        self,
        train_val_data,
        mock_recnn_config,
        log_dir="logs/",
    ):
        logger = CSVLogger(save_dir=log_dir)
        # setup data, logger and validator
        train_data, val_data = train_val_data
        validator = Validation(
            model=LitReCNN,
            model_config=mock_recnn_config,
            logger=logger,
            train_data=train_data,
            val_data=val_data,
            validation_config=self.my_not_so_fast_validation_config,
        )

        # validate model
        validator.validate()

        # assert best_validation_loss is not None
        assert validator.best_validation_loss == pytest.approx(0.50, abs=1e-2)
        assert validator.best_model_path.name == "epoch=4-validation_loss=0.50.ckpt"
