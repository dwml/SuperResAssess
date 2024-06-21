from pathlib import Path
from typing import Optional, List
from abc import ABC, abstractmethod
import logging


class AbstractClient(ABC):

    @abstractmethod
    def download_file(bucket: str, object_name: str, file_name: str) -> None:
        pass


def _download_file(
    client: AbstractClient, bucket: str, object_name: str, file_name: str
) -> None:
    try:
        client.download_file(bucket, object_name, file_name)
    except Exception as e:
        logging.log(e)


class HCP:
    def __init__(
        self, data_dir: Path, file_list: List[str], download: Optional[bool] = False
    ):
        self.data_dir = data_dir
        self.download = download
        self.file_list = file_list
