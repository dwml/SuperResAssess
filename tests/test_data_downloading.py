from pathlib import Path
import pytest
from typing import List

from superresassess.data import HCP_T2W

IMAGE_LIST_FILE = "./configurations/images.txt"

IMAGE_FIRST = "HCP_1200/100206/T1w/T2w_acpc_dc_restore_brain.nii.gz"
IMAGE_500 = "HCP_1200/199655/T1w/T2w_acpc_dc_restore_brain.nii.gz"
IMAGE_LAST = "HCP_1200/996782/T1w/T2w_acpc_dc_restore_brain.nii.gz"

NON_EXISTING_IMAGE_LIST = [
    "HCP_1200/non_sense.nii.gz",
]

TEST_NIFTI = Path("./tests/test_data/test.nii.gz")


@pytest.fixture
def image_list() -> List[str]:
    with open(IMAGE_LIST_FILE, "r") as f:
        img_lst = f.read().splitlines()
    return img_lst


@pytest.fixture
def first_image_list(image_list) -> List[str]:
    return image_list[:1]


def test_hcp_setup(tmpdir, image_list):
    hcp = HCP_T2W(tmpdir, image_list, download=False)
    assert hcp.root == tmpdir
    assert hcp.file_list == image_list
    assert not hcp.download


def test_image_list(tmpdir, image_list):
    hcp = HCP_T2W(tmpdir, image_list, download=False)
    assert hcp.file_list[0] == IMAGE_FIRST
    assert hcp.file_list[499] == IMAGE_500
    assert hcp.file_list[-1] == IMAGE_LAST


def test_download_one_image(tmpdir, first_image_list):
    _ = HCP_T2W(tmpdir, first_image_list, download=True)
    assert tmpdir.join(IMAGE_FIRST).exists()


def test_download_list_images(tmpdir):
    _ = HCP_T2W(tmpdir, [IMAGE_500, IMAGE_LAST], download=True)
    assert tmpdir.join(IMAGE_500).exists()
    assert tmpdir.join(IMAGE_LAST).exists()
