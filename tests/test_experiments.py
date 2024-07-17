import pytest

from superresassess.experiments import (
    setup_experiments,
    ExperimentConfiguration,
    IndividualExperiment,
)

from superresassess.configurations_io import (
    read_yaml_as_configuration,
)

CONTENT = """
---
  initial_seed: 2024
  iterations: null
  assessment_methods:
    - three_way_holdout
    - nested_cross_validation
    - k_fold_cross_validation
"""

CONTENT_ITERATIONS = """
---
  initial_seed: 2024
  iterations: 3
  assessment_methods:
    - three_way_holdout
    - nested_cross_validation
    - k_fold_cross_validation
"""


@pytest.fixture()
def temp_yaml(tmp_path):
    d = tmp_path / "test.yaml"
    d.write_text(CONTENT)
    yield d


@pytest.fixture()
def temp_yaml_iterations(tmp_path):
    d = tmp_path / "test.yaml"
    d.write_text(CONTENT_ITERATIONS)
    yield d


@pytest.fixture()
def temp_experiment_directory(tmp_path):
    d = tmp_path / "experiments"
    d.mkdir()
    yield d


def test_yaml_reading(temp_yaml):
    experiment_dict = read_yaml_as_configuration(temp_yaml, ExperimentConfiguration)
    assert experiment_dict.initial_seed == 2024


def test_file_creation(temp_yaml, temp_experiment_directory):
    """Test whether the function creates an individual experiment based on the
    configuration file."""
    setup_experiments(temp_yaml, temp_experiment_directory)
    assert (temp_experiment_directory / "00001.yaml").exists()


def test_iterations(temp_yaml_iterations, temp_experiment_directory):
    """Test whether the function creates repeated experiments."""
    setup_experiments(temp_yaml_iterations, temp_experiment_directory)
    # this assertion should be equal to 9 since it is 3 iterations with 3
    # assessment methods
    assert len(list(temp_experiment_directory.glob("*.yaml"))) == 9


def test_iterations_seeding(temp_yaml_iterations, temp_experiment_directory):
    """Test whether the function creates repeated experiments with different
    seeds."""
    setup_experiments(temp_yaml_iterations, temp_experiment_directory)
    experiment_3_dict = read_yaml_as_configuration(
        temp_experiment_directory / "00003.yaml", IndividualExperiment
    )
    assert experiment_3_dict.seed == 2026  # 2024 + 2


def test_assessment_methods(temp_yaml_iterations, temp_experiment_directory):
    """Test whether all assessment methods are where they need to be."""
    setup_experiments(temp_yaml_iterations, temp_experiment_directory)
    experiment_1_dict = read_yaml_as_configuration(
        temp_experiment_directory / "00001.yaml", IndividualExperiment
    )
    experiment_5_dict = read_yaml_as_configuration(
        temp_experiment_directory / "00005.yaml", IndividualExperiment
    )
    experiment_7_dict = read_yaml_as_configuration(
        temp_experiment_directory / "00007.yaml", IndividualExperiment
    )

    # the first one should be three way holdout
    assert experiment_1_dict.seed == 2024
    assert experiment_1_dict.assessment_method == "three_way_holdout"

    # the third one should be nested cross validation
    assert experiment_5_dict.seed == 2025
    assert experiment_5_dict.assessment_method == "nested_cross_validation"

    # the fifth one should be k fold cross validation
    assert experiment_7_dict.seed == 2024
    assert experiment_7_dict.assessment_method == "k_fold_cross_validation"


def test_multiple_methods_without_iterations(temp_yaml, temp_experiment_directory):
    setup_experiments(temp_yaml, temp_experiment_directory)

    assert (temp_experiment_directory / "00001.yaml").exists()
    assert (temp_experiment_directory / "00002.yaml").exists()
    assert (temp_experiment_directory / "00003.yaml").exists()
