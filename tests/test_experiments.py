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


def test_yaml_reading(temp_yaml):
    experiment_dict = _read_yaml_as_experiment(temp_yaml)
    assert experiment_dict["initial_seed"] == 2024


def test_file_creation(temp_yaml, tmp_path):
    """Test whether the function creates an individual experiment based on the
    configuration file."""
    experiment_folder = tmp_path / "experiments"
    experiment_folder.mkdir()
    setup_experiments(temp_yaml, experiment_folder)
    assert (experiment_folder / "00001.yaml").exists()
