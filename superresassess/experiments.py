import yaml
from typing import Optional
from pathlib import Path

from pydantic import BaseModel


class ExperimentConfiguration(BaseModel):
    """The data that is needed to configure all individual experiments."""

    initial_seed: int


class IndividualExperiment(BaseModel):
    """The data that is needed to run an individual experiment."""

    seed: int


def _read_yaml_as_experiment_configuration(yaml_file: Path) -> dict:
    with open(yaml_file, "r") as stream:
        config = yaml.safe_load(stream)
    return ExperimentConfiguration(**config).model_dump()


def _read_yaml_as_individual_experiment(yaml_file: Path) -> dict:
    with open(yaml_file, "r") as stream:
        config = yaml.safe_load(stream)
    return IndividualExperiment(**config).model_dump()


def _write_individual_experiment_as_yaml(
    experiment: IndividualExperiment, yaml_file: Path
) -> None:
    with open(yaml_file, "w") as stream:
        yaml.safe_dump(experiment.model_dump(), stream)


def setup_experiments(
    yaml_file: Path, destination_path: Path, iterations: Optional[int] = None
):
    """Given a configuration file, create the configuration files necessary to
    run the individual experiments.

    Args:
        yaml_file (Path): configuration file
        destination_path (Path): directory where individual configuration
            should be placed
        iterations (int): if we would like to repeat the experiment with
            different seed
    """
    experiment_config = _read_yaml_as_experiment_configuration(yaml_file)

    if iterations is None:
        seed = experiment_config["initial_seed"]
        individual_experiment = IndividualExperiment(seed=seed)
        _write_individual_experiment_as_yaml(
            individual_experiment, destination_path / "00001.yaml"
        )
        return

    # If we would like to repeat an experiment with a different seed we can do so:
    for i in range(iterations):
        seed = experiment_config["initial_seed"] + i
        individual_experiment = IndividualExperiment(seed=seed)
        _write_individual_experiment_as_yaml(
            individual_experiment, destination_path / f"{1 + i:>05}.yaml"
        )
