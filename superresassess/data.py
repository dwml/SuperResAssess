from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Union, Sequence, Mapping
import boto3

from monai.transforms.transform import Transform
from monai.transforms.compose import Compose
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.utility.dictionary import EnsureChannelFirstd, ToTensord
from monai.data.dataset import Dataset


DataListType = Sequence[Mapping[str, Path]]


class HCP_T2W:
    """This class contains functionality to obtain T2w images from the Human
    Connectome Project.

    More specifically the images that are used are the T2w brain-extracted
    images that are registered to the native T1w patient space.

    Args:
        root (str or ``pathlib.Path``): Root directory where dataset lives or
            will be saved if download is set to True.
        download (bool, optional): If true, downloads the dataset from the
            internet and puts it in root directory. If dataset is already
            downloaded, it is not downloaded again. Mind that if you would
            like this class to download the dataset, the end user has to setup
            the AWS S3 command line tool properly. For more information, see:
    """

    bucket = "hcp-openaccess"
    image_file = Path("../configurations/images.txt")

    def __init__(
        self,
        root: Union[str, Path],
        file_list: List[str],
        download: Optional[bool] = False,
    ):
        self.root = Path(root)
        self.download = download
        self.file_list = file_list

        if self.download:
            self.s3 = boto3.client("s3")
            for file in self.file_list:
                file_path = Path(file)
                file_name = file_path.name
                file_parents = file_path.parents[0]
                full_target_path = self.root.joinpath(file_parents)
                full_target_path.mkdir(parents=True, exist_ok=True)
                self.s3.download_file(
                    self.bucket,
                    str(file_path),  # endpoint must be converted to str
                    full_target_path.joinpath(file_name),
                )


def get_image_loader(dict_keys: tuple[str, ...]) -> Transform:
    return Compose(
        [LoadImaged(dict_keys), EnsureChannelFirstd(dict_keys), ToTensord(dict_keys)]
    )


class ImageDatasetd(Dataset):
    """Modification of a dataset that reads all images from disk before usage."""

    def __init__(
        self, img_paths: list[dict[str, str]], img_transform: Transform
    ) -> None:
        super().__init__(data=img_transform(img_paths))


@dataclass
class DataLoaderConfig:
    seed: int
    dict_keys: tuple[str, str]
    batch_size: int
    num_workers: int


@dataclass
class CroppedDataLoaderConfig(DataLoaderConfig):
    roi_size: tuple[int, int, int]
    samples_per_image: int
    limit_train_batches: Optional[int | float] = 1.0
