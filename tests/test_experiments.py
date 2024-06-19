import pytest

from superresassess.experiments import setup_experiments, _read_yaml_as_experiment

CONTENT = """
---
  initial_seed: 2024
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
    experiment_dict = _read_yaml_as_experiment(temp_yaml)
    assert experiment_dict["initial_seed"] == 2024


def test_file_creation(temp_yaml, temp_experiment_directory):
    """Test whether the function creates an individual experiment based on the
    configuration file."""
    setup_experiments(temp_yaml, temp_experiment_directory)
    assert (temp_experiment_directory / "00001.yaml").exists()
