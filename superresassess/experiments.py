from typing import Optional, List
from pathlib import Path
from pydantic import BaseModel, field_serializer

from superresassess.configurations_io import (
    read_yaml_as_configuration,
    write_configuration_as_yaml,
)
from superresassess.data import CroppedDataLoaderConfig, DataLoaderConfig
from superresassess.assessment.enums import AssessmentEnum

HoldoutProportion = tuple[float, float, float]

DEFAULT_EXPERIMENT_SETTINGS = {
    "train_val_test_ratio": (0.6, 0.2, 0.2),
    "log_path": "logs/",
    "max_epochs": 200,
    "learning_rate": 1e-3,
    "training_dataloader_config": {
        "dict_keys": ("img", "lab"),
        "batch_size": 2,
        "num_workers": 16,
        "roi_size": (64, 64, 64),
        "samples_per_image": 200,
    },
    "validation_dataloader_config": {
        "dict_keys": ("img", "lab"),
        "batch_size": 1,
        "num_workers": 16,
    },
    "testing_dataloader_config": {
        "dict_keys": ("img", "lab"),
        "batch_size": 1,
        "num_workers": 16,
    },
    "n_internal_images": 19,
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
    learning_rate: float
    max_epochs: int
    training_dataloader_config: CroppedDataLoaderConfig
    validation_dataloader_config: DataLoaderConfig
    testing_dataloader_config: DataLoaderConfig
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
            training_dataloader_config = CroppedDataLoaderConfig(
                seed=initial_seed,
                **DEFAULT_EXPERIMENT_SETTINGS["training_dataloader_config"],
            )
            validation_dataloader_config = DataLoaderConfig(
                seed=initial_seed,
                **DEFAULT_EXPERIMENT_SETTINGS["validation_dataloader_config"],
            )
            testing_dataloader_config = DataLoaderConfig(
                seed=initial_seed,
                **DEFAULT_EXPERIMENT_SETTINGS["testing_dataloader_config"],
            )
            individual_experiment = ExperimentConfiguration(
                assessment_method=mthd,
                train_val_test_ratio=DEFAULT_EXPERIMENT_SETTINGS[
                    "train_val_test_ratio"
                ],
                log_path=DEFAULT_EXPERIMENT_SETTINGS["log_path"],
                learning_rate=DEFAULT_EXPERIMENT_SETTINGS["learning_rate"],
                max_epochs=DEFAULT_EXPERIMENT_SETTINGS["max_epochs"],
                training_dataloader_config=training_dataloader_config,
                validation_dataloader_config=validation_dataloader_config,
                testing_dataloader_config=testing_dataloader_config,
                n_internal_images=DEFAULT_EXPERIMENT_SETTINGS["n_internal_images"],
                experiment_id=experiment_id,
                seed=initial_seed,
            )
            write_configuration_as_yaml(
                individual_experiment, destination_path / (experiment_id + ".yaml")
            )
        return

    # If we would like to repeat an experiment with a different seed we can do
    for i in range(iterations):
        for j, mthd in enumerate(experiment_config.assessment_methods):
            seed = initial_seed + i
            experiment_id = f"{1 + j + i*len(experiment_config.assessment_methods):>05}"
            training_dataloader_config = CroppedDataLoaderConfig(
                seed=seed,
                **DEFAULT_EXPERIMENT_SETTINGS["training_dataloader_config"],
            )
            validation_dataloader_config = DataLoaderConfig(
                seed=seed,
                **DEFAULT_EXPERIMENT_SETTINGS["validation_dataloader_config"],
            )
            testing_dataloader_config = DataLoaderConfig(
                seed=seed,
                **DEFAULT_EXPERIMENT_SETTINGS["testing_dataloader_config"],
            )
            individual_experiment = ExperimentConfiguration(
                assessment_method=mthd,
                train_val_test_ratio=DEFAULT_EXPERIMENT_SETTINGS[
                    "train_val_test_ratio"
                ],
                log_path=DEFAULT_EXPERIMENT_SETTINGS["log_path"],
                learning_rate=DEFAULT_EXPERIMENT_SETTINGS["learning_rate"],
                max_epochs=DEFAULT_EXPERIMENT_SETTINGS["max_epochs"],
                training_dataloader_config=training_dataloader_config,
                validation_dataloader_config=validation_dataloader_config,
                testing_dataloader_config=testing_dataloader_config,
                n_internal_images=DEFAULT_EXPERIMENT_SETTINGS["n_internal_images"],
                experiment_id=experiment_id,
                seed=seed,
            )
            write_configuration_as_yaml(
                individual_experiment,
                destination_path / (experiment_id + ".yaml"),
            )
