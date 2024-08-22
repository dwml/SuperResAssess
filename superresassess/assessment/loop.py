from pathlib import Path

from lightning import LightningModule
from pydantic import BaseModel, field_serializer

from superresassess.experiments import ExperimentConfiguration
from superresassess.assessment.mapper import assessment_mapper
from superresassess.configurations_io import (
    read_yaml_as_configuration,
    write_configuration_as_yaml,
)
from superresassess.model import ReCNNConfiguration
from superresassess.utils import read_image_label_file


class LoopResult(BaseModel):
    best_model_path: Path
    internal_testing_loss: float
    external_testing_loss: float
    difference_internal_external_loss: float

    @field_serializer("best_model_path")
    def serialize_best_model_path(self, best_model_path: Path, _info):
        return str(best_model_path)


class Loop:
    """class that represents the loop of an experiment

    This means it sets up the experiment, runs it, logs it and cleans everything up.
    """

    def __init__(self, experiment_configuration_path: Path, dataset_path: Path):
        self._experiment_configuration_path = experiment_configuration_path
        self._experiment_configuration = read_yaml_as_configuration(
            self._experiment_configuration_path, ExperimentConfiguration
        )
        self._dataset_path = dataset_path

    def setup_experiment(
        self,
        lightning_module_type: type[LightningModule],
        lightning_module_config_path: Path,
    ) -> None:
        lightning_module_config = read_yaml_as_configuration(
            lightning_module_config_path, ReCNNConfiguration
        )
        dataset = read_image_label_file(self._dataset_path)
        assessment_method_type = assessment_mapper[
            self._experiment_configuration.assessment_method
        ]
        self._assessment_method = assessment_method_type(
            lightning_module_type,
            lightning_module_config,
            dataset,
            self._experiment_configuration,
        )

    def run_experiment(self) -> None:
        self._assessment_method.assess()
        self._assessment_method.test()

    def log_experiment(self) -> None:
        if not self._assessment_method.internal_testing_loss:
            raise AttributeError(
                "No internal testing loss set. Was assessment method tested properly?"
            )
        if not self._assessment_method.external_testing_loss:
            raise AttributeError(
                "No external testing loss set. Was assessment method tested properly?"
            )
        if not self._assessment_method.best_model_path:
            raise AttributeError(
                "No best model path set loss. Was assessment method tested properly?"
            )

        loop_result = LoopResult(
            best_model_path=self._assessment_method.best_model_path,
            internal_testing_loss=self._assessment_method.internal_testing_loss,
            external_testing_loss=self._assessment_method.external_testing_loss,
            difference_internal_external_loss=(
                self._assessment_method.internal_testing_loss
                - self._assessment_method.external_testing_loss
            ),
        )
        write_configuration_as_yaml(
            loop_result,
            self._experiment_configuration.log_path.joinpath("loop_result.yaml"),
        )

    def clean_up_experiment(self, completed_experiment_directory: Path) -> None:
        experiment_name = self._experiment_configuration_path.name
        self._experiment_configuration_path.rename(
            completed_experiment_directory.joinpath(experiment_name)
        )
