import pytest
from lightning import Trainer

from superresassess.fold import Fold
from superresassess.model import LitReCNN, ReCNNConfiguration
from superresassess.seeded import (
    SeededModelProvider,
    SeededCroppedDataLoaderProvider,
    SeededDataLoaderProvider,
)
from .conftest import SEED


class TestFold:
    my_model_config = ReCNNConfiguration(
        n_layers=10,
        spatial_dims=3,
        in_channels=1,
        intermediate_channels=10,
        out_channels=1,
        kernel_size=3,
        stride=1,
    )

    @pytest.fixture
    def seeded_model_provider(self) -> SeededModelProvider:
        return SeededModelProvider(LitReCNN, self.my_model_config, seed=SEED)

    @pytest.fixture
    def training_provider(
        self, cropped_dataloader_config
    ) -> SeededCroppedDataLoaderProvider:
        return SeededCroppedDataLoaderProvider(cropped_dataloader_config)

    @pytest.fixture
    def validation_provider(self, dataloader_config) -> SeededDataLoaderProvider:
        return SeededDataLoaderProvider(dataloader_config)

    @pytest.fixture
    def testing_provider(self, dataloader_config) -> SeededDataLoaderProvider:
        return SeededDataLoaderProvider(dataloader_config)

    @pytest.fixture
    def fold(
        self,
        seeded_model_provider,
        training_provider,
        validation_provider,
        testing_provider,
        tmp_path,
    ) -> Fold:
        return Fold(
            seeded_model_provider=seeded_model_provider,
            seeded_training_provider=training_provider,
            seeded_validation_provider=validation_provider,
            seeded_testing_provider=testing_provider,
            log_path=tmp_path,
            max_epochs=2,
            experiment_id="00001",
        )

    @pytest.fixture
    def patch_provide_model(self, monkeypatch):
        def mock_provide_model(self):
            return "test_model"

        monkeypatch.setattr(SeededModelProvider, "provide", mock_provide_model)

    @pytest.fixture
    def patch_provide_train_dataloader(self, fold, monkeypatch):
        def mock_provide_train_dataloader(train_data):
            return "test_train_dataloader"

        monkeypatch.setattr(
            fold.seeded_training_provider, "provide", mock_provide_train_dataloader
        )

    @pytest.fixture
    def patch_provide_validation_dataloader(self, fold, monkeypatch):
        def mock_provide_val_dataloader(val_data):
            return "test_validation_dataloader"

        monkeypatch.setattr(
            fold.seeded_validation_provider, "provide", mock_provide_val_dataloader
        )

    @pytest.fixture
    def patch_provide_testing_dataloader(self, fold, monkeypatch):
        def mock_provide_testing_dataloader(testing_data):
            return "test_testing_dataloader"

        monkeypatch.setattr(
            fold.seeded_testing_provider, "provide", mock_provide_testing_dataloader
        )

    def test_unset_version(self, fold) -> None:
        with pytest.raises(AttributeError):
            fold.train_model()

    def test_train_model(
        self, patch_provide_model, patch_provide_train_dataloader, monkeypatch, fold
    ) -> None:
        patch_params = None

        def mock_fit(self, model, dataloader):
            nonlocal patch_params
            patch_params = [model, dataloader]

        monkeypatch.setattr(Trainer, "fit", mock_fit)

        fold.version = "test_version"
        fold.train_model("test_train_data")

        assert patch_params == ["test_model", "test_train_dataloader"]

    def test_fold_unsets_version_after_training(
        self, patch_provide_model, patch_provide_train_dataloader, monkeypatch, fold
    ) -> None:
        patch_params = None

        def mock_fit(self, model, dataloader):
            nonlocal patch_params
            patch_params = [model, dataloader]

        monkeypatch.setattr(Trainer, "fit", mock_fit)

        fold.version = "test_version"
        fold.train_model("test_train_data")

        with pytest.raises(AttributeError):
            fold.train_model("test_train_data")

    def test_train_and_validate_model(
        self,
        patch_provide_model,
        patch_provide_train_dataloader,
        patch_provide_validation_dataloader,
        monkeypatch,
        fold,
    ) -> None:
        patch_params = None

        def mock_fit(self, model, train_dataloader, val_dataloader):
            nonlocal patch_params
            patch_params = [model, train_dataloader, val_dataloader]

        monkeypatch.setattr(Trainer, "fit", mock_fit)

        fold.version = "test_version"
        fold.train_and_validate_model("test_train_data", "test_validation_data")

        assert patch_params == [
            "test_model",
            "test_train_dataloader",
            "test_validation_dataloader",
        ]

    def test_fold_unsets_version_after_training_and_validating(
        self,
        patch_provide_model,
        patch_provide_train_dataloader,
        patch_provide_validation_dataloader,
        monkeypatch,
        fold,
    ) -> None:
        patch_params = None

        def mock_fit(self, model, train_dataloader, val_dataloader):
            nonlocal patch_params
            patch_params = [model, train_dataloader, val_dataloader]

        monkeypatch.setattr(Trainer, "fit", mock_fit)

        fold.version = "test_version"
        fold.train_and_validate_model("test_train_data", "test_validation_data")

        with pytest.raises(AttributeError):
            fold.train_model("test_train_data")

    def test_test_model(
        self,
        patch_provide_model,
        patch_provide_testing_dataloader,
        monkeypatch,
        fold,
    ) -> None:
        patch_params = None

        def mock_test(self, model, dataloaders, ckpt_path):
            nonlocal patch_params
            patch_params = [model, dataloaders, ckpt_path]
            return [{"test_loss": 0.1}]

        monkeypatch.setattr(Trainer, "test", mock_test)

        fold.version = "test_version"

        fold.test_model("test_path", "test_testing_data")

        assert patch_params == ["test_model", "test_testing_dataloader", "test_path"]

    def test_fold_unsets_version_after_testing(
        self,
        patch_provide_model,
        patch_provide_testing_dataloader,
        monkeypatch,
        fold,
    ) -> None:
        patch_params = None

        def mock_test(self, model, dataloaders, ckpt_path):
            nonlocal patch_params
            patch_params = [model, dataloaders, ckpt_path]
            return [{"test_loss": 0.1}]

        monkeypatch.setattr(Trainer, "test", mock_test)

        fold.version = "test_version"

        fold.test_model("test_path", "test_testing_data")

        with pytest.raises(AttributeError):
            fold.train_model("test_train_data")
