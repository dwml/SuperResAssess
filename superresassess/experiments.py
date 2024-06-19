import yaml
from pathlib import Path

from pydantic import BaseModel


class ExperimentConfiguration(BaseModel):
    """The data that is needed to configure all individual experiments."""

    initial_seed: int


class IndividualExperiment(BaseModel):
    """The data that is needed to run an individual experiment."""

    seed: int


def _read_yaml_as_experiment(yaml_file: Path) -> dict:
    with open(yaml_file, "r") as stream:
        config = yaml.safe_load(stream)
    return ExperimentConfiguration(**config).model_dump()


def _write_individual_experiment_as_yaml(
    experiment: IndividualExperiment, yaml_file: Path
) -> None:
    with open(yaml_file, "w") as stream:
        yaml.safe_dump(experiment.model_dump(), stream)


def setup_experiments(yaml_file: Path, destination_path: Path):
    """Given a configuration file, create the configuration files necessary to
    run the individual experiments.

    Args:
        yaml_file (Path): configuration file
        destination_path (Path): directory where individual configuration
            should be placed
    """
    experiment_config = _read_yaml_as_experiment(yaml_file)
    seed = experiment_config["initial_seed"]
    individual_experiment = IndividualExperiment(seed=seed)
    _write_individual_experiment_as_yaml(
        individual_experiment, destination_path / "00001.yaml"
    )
