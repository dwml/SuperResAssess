from pathlib import Path
from pydantic import BaseModel
import yaml


def read_yaml_as_configuration(yaml_file: Path, configuration: BaseModel) -> BaseModel:
    with open(yaml_file, "r") as stream:
        config = yaml.safe_load(stream)
    return configuration(**config)


def write_configuration_as_yaml(configuration: BaseModel, yaml_file: Path) -> None:
    with open(yaml_file, "w") as stream:
        yaml.safe_dump(configuration.model_dump(), stream)
