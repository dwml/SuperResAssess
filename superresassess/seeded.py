from lightning import LightningModule, seed_everything

from monai.data.dataloader import DataLoader
from monai.transforms.croppad.dictionary import RandSpatialCropSamplesd
from monai.data.grid_dataset import PatchDataset
from monai.data.dataset import Dataset

from superresassess.data import (
    DataListType,
    get_image_loader,
    DataLoaderConfig,
    CroppedDataLoaderConfig,
)
from superresassess.model import ReCNNConfiguration


class SeededModelProvider:
    def __init__(
        self,
        lightning_module_type: type[LightningModule],
        lightning_module_config: ReCNNConfiguration,
        seed: int,
    ):
        self.lightning_module_type = lightning_module_type
        self.lightning_module_config = lightning_module_config
        self.seed = seed

    def provide(self) -> LightningModule:
        seed_everything(self.seed)
        return self.lightning_module_type(self.lightning_module_config)


class SeededDataLoaderProvider:
    def __init__(self, config: DataLoaderConfig):
        self.seed = config.seed
        self.dict_keys = config.dict_keys
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers

    def provide(self, data: DataListType) -> DataLoader:
        seed_everything(self.seed, workers=True)
        loader = get_image_loader(dict_keys=self.dict_keys)
        images = Dataset(data, transform=loader)
        return DataLoader(
            images, batch_size=self.batch_size, num_workers=self.num_workers
        )


class SeededCroppedDataLoaderProvider(SeededDataLoaderProvider):
    def __init__(self, config: CroppedDataLoaderConfig) -> None:
        super().__init__(config)
        self.roi_size = config.roi_size
        self.samples_per_image = config.samples_per_image

    def provide(self, data: DataListType) -> DataLoader:
        seed_everything(self.seed, workers=True)
        loader = get_image_loader(dict_keys=self.dict_keys)
        images = Dataset(data, transform=loader)

        train_cropper = RandSpatialCropSamplesd(
            self.dict_keys,
            self.roi_size,
            self.samples_per_image,
        )
        images = PatchDataset(
            images,  # type: ignore
            patch_func=train_cropper,
            samples_per_image=self.samples_per_image,
        )

        return DataLoader(
            images, batch_size=self.batch_size, num_workers=self.num_workers
        )
