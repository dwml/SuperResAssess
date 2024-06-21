from botocore.exceptions import ClientError
from pathlib import Path
import shutil
import pytest

from superresassess.data import HCP, _download_file


IMAGE_LIST = [
    "HCP_1200/100206/T1w/T2w_acpc_dc_restore_brain.nii.gz",
]

NON_EXISTING_IMAGE_LIST = [
    "HCP_1200/non_sense.nii.gz",
]

TEST_NIFTI = Path("./tests/test_data/test.nii.gz")


# For now I don't want to access aws S3 in testing, so this will suffice
class MockS3:
    def __init__(self, service: str):
        _ = service

    def download_file(self, bucket, object_name: str, file_name: str):
        if not bucket == "hcp-openaccess":
            raise ClientError(
                {"Error": {"Code": "404", "Message": "Not Found"}}, "download_file"
            )

        test_file = Path(object_name)

        if not test_file.is_file():
            raise ClientError(
                {"Error": {"Code": "404", "Message": "Not Found"}}, "download_file"
            )

        target_file = Path(file_name)
        shutil.copy(test_file, target_file)


def test_hcp_setup(tmpdir):
    hcp = HCP(tmpdir, IMAGE_LIST, download=True)
    assert hcp.data_dir == tmpdir
    assert hcp.file_list == IMAGE_LIST
    assert hcp.download


def test_existing_file(tmp_path):
    s3 = MockS3("s3")
    bucket = "hcp-openaccess"
    file_name = str(tmp_path / "test.nii.gz")

    _download_file(s3, bucket, TEST_NIFTI, file_name)

    assert TEST_NIFTI.is_file()
    assert Path(file_name).is_file()


def test_non_existent_bucket(tmp_path):
    s3 = MockS3("s3")
    file_name = str(tmp_path / "test.nii.gz")

    with pytest.raises(ClientError):
        _download_file(s3, "non_existent", TEST_NIFTI, file_name)
