import yaml
from typing import Optional, List
from pathlib import Path
from pydantic import BaseModel, field_serializer

from superresassess.assessment_methods import AssessmentEnum


class ExperimentConfiguration(BaseModel):
    """The data that is needed to configure all individual experiments."""

    initial_seed: int
    iterations: Optional[int] = None
    assessment_methods: List[AssessmentEnum]


class IndividualExperiment(BaseModel):
    """The data that is needed to run an individual experiment."""

    seed: int
    assessment_method: AssessmentEnum

    @field_serializer("assessment_method")
    def serialize_assessment_method(self, assessment_method: AssessmentEnum, _info):
        return assessment_method.value


def _read_yaml_as_experiment_configuration(yaml_file: Path) -> ExperimentConfiguration:
    with open(yaml_file, "r") as stream:
        config = yaml.safe_load(stream)
    return ExperimentConfiguration(**config)


def read_yaml_as_individual_experiment(yaml_file: Path) -> IndividualExperiment:
    with open(yaml_file, "r") as stream:
        config = yaml.safe_load(stream)
    return IndividualExperiment(**config)


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
        iterations (int): if we would like to repeat the experiment with
            different seed
    """
    experiment_config = _read_yaml_as_experiment_configuration(yaml_file)
    initial_seed = experiment_config.initial_seed
    iterations = experiment_config.iterations

    if iterations is None:
        for i, mthd in enumerate(experiment_config.assessment_methods):
            individual_experiment = IndividualExperiment(
                seed=initial_seed, assessment_method=mthd
            )
            _write_individual_experiment_as_yaml(
                individual_experiment, destination_path / f"{1 + i:>05}.yaml"
            )
        return

    # If we would like to repeat an experiment with a different seed we can do
    for i, mthd in enumerate(experiment_config.assessment_methods):
        for j in range(iterations):
            seed = initial_seed + j
            individual_experiment = IndividualExperiment(
                seed=seed, assessment_method=mthd
            )
            _write_individual_experiment_as_yaml(
                individual_experiment,
                destination_path / f"{1 + i*iterations + j:>05}.yaml",
            )
