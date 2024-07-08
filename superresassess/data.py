from pathlib import Path
from typing import Optional, List, Union
import boto3


class HCP_T2W:
    """This class contains functionality to obtain T2w images from the Human Connectome
    Project.

    More specifically the images that are used are the T2w brain-extracted images that
    are registered to the native T1w patient space.

    Args:
        root (str or ``pathlib.Path``): Root directory where dataset lives or will be
            saved if download is set to True.
        download (bool, optional): If true, downloads the dataset from the internet
            and puts it in root directory. If dataset is already donwloaded, it is not
            downloaded again. Mind that if you would like this class to download the
            dataset, the end user has to setup the AWS S3 command line tool properly.
            For more information, see:
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

        self.s3 = boto3.client("s3")

        if self.download:
            for file in self.file_list:
                file_path = Path(file)
                file_name = file_path.name
                file_parents = file_path.parents[0]
                full_target_path = self.root.joinpath(file_parents)
                full_target_path.mkdir(parents=True, exist_ok=True)
                self.s3.download_file(
                    self.bucket,
                    str(file_path),  # endpoint cant be path, must be converted to str
                    full_target_path.joinpath(file_name),
                )
