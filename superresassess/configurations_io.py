from pathlib import Path
from typing import TypeVar
from pydantic import BaseModel
import yaml

Config = TypeVar(name="Config", bound=BaseModel)


def read_yaml_as_configuration(yaml_path: Path, configuration: type[Config]) -> Config:
    with open(yaml_path, "r") as stream:
        config = yaml.safe_load(stream)
    return configuration(**config)


def write_configuration_as_yaml(configuration: BaseModel, yaml_path: Path) -> None:
    with open(yaml_path, "w") as stream:
        yaml.safe_dump(configuration.model_dump(), stream)
