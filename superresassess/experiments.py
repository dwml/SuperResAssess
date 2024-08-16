from typing import Optional, List
from pathlib import Path
from pydantic import BaseModel, field_serializer

from superresassess.configurations_io import (
    read_yaml_as_configuration,
    write_configuration_as_yaml,
)
from superresassess.data import DataConfig
from superresassess.assessment.enums import AssessmentEnum

HoldoutProportion = tuple[float, float, float]

DEFAULT_EXPERIMENT_SETTINGS = {
    "train_val_test_ratio": (0.6, 0.2, 0.2),
    "log_path": "logs/",
    "data_config": {
        "train_batch_size": 6,
        "val_batch_size": 1,
        "test_batch_size": 1,
        "train_workers": 16,
        "val_workers": 16,
        "test_workers": 16,
        "train_roi_size": (64, 64, 64),
        "max_epochs": 50,
        "learning_rate": 1e-4,
        "dict_keys": ("img", "lab"),
        "samples_per_image": 200,
    },
    "n_internal_images": 20,
}


class StudyConfiguration(BaseModel):
    """The data that is needed to configure all individual experiments."""

    initial_seed: int
    iterations: Optional[int] = None
    assessment_methods: List[AssessmentEnum]


class ExperimentConfiguration(BaseModel):
    """The data that is needed to run an individual experiment."""

    assessment_method: AssessmentEnum
    train_val_test_ratio: HoldoutProportion
    log_path: Path
    data_config: DataConfig
    n_internal_images: int
    experiment_id: str
    seed: int

    @field_serializer("assessment_method")
    def serialize_assessment_method(self, assessment_method: AssessmentEnum, _info):
        return assessment_method.value

    @field_serializer("log_path")
    def serialize_log_path(self, log_path: Path, _info):
        return str(log_path)


def setup_experiments(
    yaml_file: Path,
    destination_path: Path,
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
    experiment_config = read_yaml_as_configuration(yaml_file, StudyConfiguration)
    initial_seed = experiment_config.initial_seed
    iterations = experiment_config.iterations

    if iterations is None:
        for i, mthd in enumerate(experiment_config.assessment_methods):
            experiment_id = f"{1 + i:>05}"
            data_config = DataConfig(**DEFAULT_EXPERIMENT_SETTINGS["data_config"])
            individual_experiment = ExperimentConfiguration(
                assessment_method=mthd,
                train_val_test_ratio=DEFAULT_EXPERIMENT_SETTINGS[
                    "train_val_test_ratio"
                ],
                log_path=DEFAULT_EXPERIMENT_SETTINGS["log_path"],
                data_config=data_config,
                n_internal_images=DEFAULT_EXPERIMENT_SETTINGS["n_internal_images"],
                experiment_id=experiment_id,
                seed=initial_seed,
            )
            write_configuration_as_yaml(
                individual_experiment, destination_path / (experiment_id + ".yaml")
            )
        return

    # If we would like to repeat an experiment with a different seed we can do
    for i, mthd in enumerate(experiment_config.assessment_methods):
        for j in range(iterations):
            seed = initial_seed + j
            experiment_id = f"{1 + i*iterations + j:>05}"
            data_config = DataConfig(**DEFAULT_EXPERIMENT_SETTINGS["data_config"])
            individual_experiment = ExperimentConfiguration(
                assessment_method=mthd,
                train_val_test_ratio=DEFAULT_EXPERIMENT_SETTINGS[
                    "train_val_test_ratio"
                ],
                log_path=DEFAULT_EXPERIMENT_SETTINGS["log_path"],
                data_config=data_config,
                n_internal_images=DEFAULT_EXPERIMENT_SETTINGS["n_internal_images"],
                experiment_id=experiment_id,
                seed=seed,
            )
            write_configuration_as_yaml(
                individual_experiment,
                destination_path / (experiment_id + ".yaml"),
            )
