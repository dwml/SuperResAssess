import sys
from pathlib import Path
import random

from superresassess.assessment.results import AssessResult
from superresassess.configurations_io import (
    read_yaml_as_configuration,
    write_configuration_as_yaml,
)
from superresassess.experiments import ExperimentConfiguration
from superresassess.model import LitReCNN, ReCNNConfiguration
from superresassess.utils import read_image_label_file
from superresassess.assessment.mapper import assessment_mapper


def main(experiment_configuration_path, dataset_path, lightning_module_config_path):
    experiment_configuration = read_yaml_as_configuration(
        experiment_configuration_path, ExperimentConfiguration
    )
    lightning_module_config = read_yaml_as_configuration(
        lightning_module_config_path, ReCNNConfiguration
    )
    dataset = read_image_label_file(dataset_path)
    random.seed(experiment_configuration.seed)
    random.shuffle(dataset)
    assessment_method_type = assessment_mapper[
        experiment_configuration.assessment_method
    ]
    assessment_method = assessment_method_type(
        LitReCNN, lightning_module_config, dataset, experiment_configuration
    )

    experiment_log_dir = experiment_configuration.log_path.joinpath(
        experiment_configuration.experiment_id
    )
    assess_result = read_yaml_as_configuration(
        experiment_log_dir.joinpath("assess_result.yaml"), AssessResult
    )
    test_result = assessment_method.external_test(assess_result.best_model_path)
    write_configuration_as_yaml(
        test_result, experiment_log_dir.joinpath("external_test_result.yaml")
    )


if __name__ == "__main__":
    experiment_configuration_path = Path(sys.argv[1])
    dataset_path = Path(sys.argv[2])
    lightning_module_config_path = Path(sys.argv[3])
    main(experiment_configuration_path, dataset_path, lightning_module_config_path)
