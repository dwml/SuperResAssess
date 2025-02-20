from pathlib import Path

from superresassess.data import DataListType


def read_image_label_file(image_label_file: Path) -> DataListType:
    with open(image_label_file, "r") as file_handle:
        image_label_lines = file_handle.readlines()
    image_mapping_list = [
        {
            "img": Path(line.strip().split(" ")[0]),
            "lab": Path(line.strip().split(" ")[1]),
        }
        for line in image_label_lines
    ]
    return image_mapping_list
