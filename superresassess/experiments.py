import yaml
from pathlib import Path

from pydantic import BaseModel


class Experiment(BaseModel):
    initial_seed: int


def _read_yaml_as_experiment(yaml_file: Path) -> dict:
    with open(yaml_file, "r") as stream:
        config = yaml.safe_load(stream)
    return Experiment(**config).model_dump()


def setup_experiments(yaml_file: Path, destination_path: Path):
    _ = _read_yaml_as_experiment(yaml_file)["initial_seed"]
