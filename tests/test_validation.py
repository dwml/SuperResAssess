from superresassess.validation import Validation
from lightning.pytorch.loggers import CSVLogger
import yaml


class TestValidation:
    def test_logging(self, tmp_path, mock_data):
        """The Validation class should log the filenames to be able to double check
        that the correct files were used."""
        # setup data and logger
        train_val_test_split = (0.6, 0.2, 0.2)
        train_length = int(train_val_test_split[0] * len(mock_data))
        val_length = int(train_val_test_split[1] * len(mock_data))
        logger = CSVLogger(save_dir=tmp_path, name="test", version="iteration1")

        # this should log the file
        _ = Validation(
            logger=logger,
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
