import subprocess

from pathlib import Path


class Loop:
    """class that represents the loop of an experiment

    This means it sets up the experiment, runs it, logs it and cleans everything up.
    """

    def __init__(
        self,
        experiment_configuration_path: Path,
        dataset_path: Path,
        model_configuration_file: Path,
    ):
        self._experiment_configuration_path = experiment_configuration_path
        self._dataset_path = dataset_path
        self._model_configuration_path = model_configuration_file

    def run_experiment(self) -> None:
        subprocess.call(
            [
                "python",
                "superresassess/assess_script.py",
                str(self._experiment_configuration_path),
                str(self._dataset_path),
                str(self._model_configuration_path),
            ]
        )
        subprocess.call(
            [
                "python",
                "superresassess/internal_test_script.py",
                str(self._experiment_configuration_path),
                str(self._dataset_path),
                str(self._model_configuration_path),
            ]
        )
        subprocess.call(
            [
                "python",
                "superresassess/external_test_script.py",
                str(self._experiment_configuration_path),
                str(self._dataset_path),
                str(self._model_configuration_path),
            ]
        )

    def clean_up_experiment(self, completed_experiment_directory: Path) -> None:
        experiment_name = self._experiment_configuration_path.name
        self._experiment_configuration_path.rename(
            completed_experiment_directory.joinpath(experiment_name)
        )
