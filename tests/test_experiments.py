import pytest

from superresassess.experiments import (
    setup_experiments,
    _read_yaml_as_experiment_configuration,
    _read_yaml_as_individual_experiment,
)


# TODO move iterations to config file
CONTENT = """
---
  initial_seed: 2024
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
def temp_experiment_directory(tmp_path):
    d = tmp_path / "experiments"
    d.mkdir()
    yield d


def test_yaml_reading(temp_yaml):
    experiment_dict = _read_yaml_as_experiment_configuration(temp_yaml)
    assert experiment_dict["initial_seed"] == 2024


def test_file_creation(temp_yaml, temp_experiment_directory):
    """Test whether the function creates an individual experiment based on the
    configuration file."""
    setup_experiments(temp_yaml, temp_experiment_directory)
    assert (temp_experiment_directory / "00001.yaml").exists()


def test_iterations(temp_yaml, temp_experiment_directory):
    """Test whether the function creates repeated experiments."""
    setup_experiments(temp_yaml, temp_experiment_directory, iterations=3)

    # mind that this assert needs to equal 9 since we have 3 iterations and
    # 3 assessment methods
    assert len(list(temp_experiment_directory.glob("*.yaml"))) == 9


def test_iterations_seeding(temp_yaml, temp_experiment_directory):
    """Test whether the function creates repeated experiments with different
    seeds."""
    setup_experiments(temp_yaml, temp_experiment_directory, iterations=3)
    experiment_3_dict = _read_yaml_as_individual_experiment(
        temp_experiment_directory / "00003.yaml"
    )
    assert experiment_3_dict["seed"] == 2026  # 2024 + 2


def test_assessment_methods(temp_yaml, temp_experiment_directory):
    """Test whether all assessment methods are where they need to be."""
    setup_experiments(temp_yaml, temp_experiment_directory, iterations=2)
    experiment_1_dict = _read_yaml_as_individual_experiment(
        temp_experiment_directory / "00001.yaml"
    )
    experiment_3_dict = _read_yaml_as_individual_experiment(
        temp_experiment_directory / "00003.yaml"
    )
    experiment_5_dict = _read_yaml_as_individual_experiment(
        temp_experiment_directory / "00005.yaml"
    )

    # the first one should be three way holdout
    assert experiment_1_dict["assessment_method"] == "three_way_holdout"

    # the third one should be nested cross validation
    assert experiment_3_dict["assessment_method"] == "nested_cross_validation"

    # the fifth one should be k fold cross validation
    assert experiment_5_dict["assessment_method"] == "k_fold_cross_validation"
