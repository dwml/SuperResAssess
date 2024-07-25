from superresassess.validation import Validation
from superresassess.model import LitReCNN
from lightning.pytorch.loggers import CSVLogger
import yaml
import torch


class TestValidation:
    seed = 2024

    def test_logging(self, tmp_path, mock_data, mock_recnn_config):
        """The Validation class should log the filenames to be able to double check
        that the correct files were used."""
        # setup data and logger
        train_val_test_split = (0.6, 0.2, 0.2)
        train_length = int(train_val_test_split[0] * len(mock_data))
        val_length = int(train_val_test_split[1] * len(mock_data))
        logger = CSVLogger(save_dir=tmp_path, name="test", version="iteration1")

        # this should log the file
        _ = Validation(
            model=LitReCNN,
            config=mock_recnn_config,
            logger=logger,
            seed=self.seed,
            train_data=mock_data[:train_length],
            val_data=mock_data[train_length : train_length + val_length],
        )

        # Assert that contents are correct
        yaml_path = (
            tmp_path.joinpath("test").joinpath("iteration1").joinpath("hparams.yaml")
        )
        contents = yaml.safe_load(yaml_path.read_text())
        assert contents["train_data"] == [
            str(datum) for datum in mock_data[:train_length]
        ]
        assert contents["val_data"] == [
            str(datum) for datum in mock_data[train_length : train_length + val_length]
        ]

    def test_model_initialization_is_deterministic(
        self, tmp_path, mock_data, mock_recnn_config
    ):
        # setup data and logger
        train_val_test_split = (0.6, 0.2, 0.2)
        train_length = int(train_val_test_split[0] * len(mock_data))
        val_length = int(train_val_test_split[1] * len(mock_data))
        logger = CSVLogger(save_dir=tmp_path, name="test", version="iteration1")

        # Seting up the validators with the same seed should yield
        # models with the same initialized weights
        validator = Validation(
            model=LitReCNN,
            config=mock_recnn_config,
            logger=logger,
            seed=self.seed,
            train_data=mock_data[:train_length],
            val_data=mock_data[train_length : train_length + val_length],
        )
        validator._setup()
        validator2 = Validation(
            model=LitReCNN,
            config=mock_recnn_config,
            logger=logger,
            seed=self.seed,
            train_data=mock_data[:train_length],
            val_data=mock_data[train_length : train_length + val_length],
        )
        validator2._setup()

        # compare weights
        weights = validator.instantiated_model.state_dict()
        weights2 = validator.instantiated_model.state_dict()
        equality = [
            torch.equal(w1, w2) for w1, w2 in zip(weights.values(), weights2.values())
        ]
        assert all(equality)
