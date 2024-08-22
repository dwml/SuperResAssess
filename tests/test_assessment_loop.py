import pytest
from pathlib import Path

from superresassess.assessment.enums import AssessmentEnum
from superresassess.assessment.loop import Loop, LoopResult
from superresassess.configurations_io import (
    read_yaml_as_configuration,
    write_configuration_as_yaml,
)
from superresassess.experiments import ExperimentConfiguration
from superresassess.model import LitReCNN
from superresassess.three_way_holdout import ThreeWayHoldout

from .conftest import _mock_data


# TODO: add nested_cross_validation
@pytest.fixture
def experiment_config_three_way_holdout(
    tmp_path,
    cropped_dataloader_config,
    dataloader_config,
) -> ExperimentConfiguration:
    return ExperimentConfiguration(
        assessment_method=AssessmentEnum("three_way_holdout"),
        train_val_test_ratio=(0.6, 0.2, 0.2),
        log_path=tmp_path,
        learning_rate=1e-3,
        max_epochs=200,
        training_dataloader_config=cropped_dataloader_config,
        validation_dataloader_config=dataloader_config,
        testing_dataloader_config=dataloader_config,
        n_internal_images=20,
        experiment_id="00001",
        seed=2024,
    )


@pytest.fixture
def experiment_config_k_fold_cross_validation(
    tmp_path,
    cropped_dataloader_config,
    dataloader_config,
) -> ExperimentConfiguration:
    return ExperimentConfiguration(
        assessment_method=AssessmentEnum("three_way_holdout"),
        train_val_test_ratio=(0.6, 0.2, 0.2),
        log_path=tmp_path,
        learning_rate=1e-3,
        max_epochs=200,
        training_dataloader_config=cropped_dataloader_config,
        validation_dataloader_config=dataloader_config,
        testing_dataloader_config=dataloader_config,
        n_internal_images=20,
        experiment_id="00001",
        seed=2024,
    )


@pytest.fixture
def experiment_config_path_three_way_holdout(
    experiment_config_three_way_holdout: ExperimentConfiguration, tmp_path
) -> Path:
    experiment_directory = tmp_path.joinpath("experiments")
    experiment_directory.mkdir(exist_ok=True)
    experiment_path = experiment_directory.joinpath("00001.yaml")
    write_configuration_as_yaml(
        experiment_config_three_way_holdout,
        experiment_path,
    )
    return experiment_path


@pytest.fixture
def experiment_config_path_k_fold_cross_validation(
    experiment_config_k_fold_cross_validation: ExperimentConfiguration, tmp_path
) -> Path:
    experiment_directory = tmp_path.joinpath("experiments")
    experiment_directory.mkdir(exist_ok=True)
    experiment_path = experiment_directory.joinpath("00001.yaml")
    write_configuration_as_yaml(
        experiment_config_k_fold_cross_validation,
        experiment_path,
    )
    return experiment_path


@pytest.fixture
def configuration_directory(tmp_path) -> Path:
    configuration_directory = tmp_path.joinpath("configurations")
    configuration_directory.mkdir(exist_ok=True)
    return configuration_directory


@pytest.fixture
def mock_data_path(tmpdir_factory, configuration_directory) -> Path:
    """Create 40 hr/lr pairs"""
    # Setup path
    save_dir = Path(tmpdir_factory.mktemp("data"))

    data_list = _mock_data(40, save_dir)

    data_text = [",".join(map(str, dictionary.values())) for dictionary in data_list]

    image_file_path = configuration_directory.joinpath("image_files.txt")
    with open(image_file_path, "w+") as fh:
        fh.writelines(data_text)

    return image_file_path


@pytest.fixture
def mock_experiment_path(tmp_path, mock_recnn_config):
    configuration_directory = tmp_path.joinpath("configurations")
    configuration_directory.mkdir(exist_ok=True)
    recnn_path = configuration_directory.joinpath("reccn_config.yaml")
    write_configuration_as_yaml(mock_recnn_config, recnn_path)
    return recnn_path


def test_reading_three_way_holdout_setup(
    experiment_config_path_three_way_holdout, mock_data_path, mock_experiment_path
):
    loop = Loop(
        experiment_configuration_path=experiment_config_path_three_way_holdout,
        dataset_path=mock_data_path,
    )
    loop.setup_experiment(LitReCNN, mock_experiment_path)
    assert isinstance(loop._assessment_method, ThreeWayHoldout)


def test_reading_k_fold_cross_validation_setup(
    experiment_config_path_k_fold_cross_validation, mock_data_path, mock_experiment_path
):
    loop = Loop(
        experiment_configuration_path=experiment_config_path_k_fold_cross_validation,
        dataset_path=mock_data_path,
    )
    loop.setup_experiment(LitReCNN, mock_experiment_path)
    assert isinstance(loop._assessment_method, ThreeWayHoldout)


def test_run_experiment(
    monkeypatch,
    experiment_config_path_k_fold_cross_validation,
    mock_data_path,
    mock_experiment_path,
):
    loop = Loop(
        experiment_configuration_path=experiment_config_path_k_fold_cross_validation,
        dataset_path=mock_data_path,
    )
    loop.setup_experiment(LitReCNN, mock_experiment_path)

    assess_is_called = False

    def mock_assess() -> None:
        nonlocal assess_is_called
        assess_is_called = True

    test_is_called = False

    def mock_test() -> None:
        nonlocal test_is_called
        test_is_called = True

    monkeypatch.setattr(loop._assessment_method, "assess", mock_assess)
    monkeypatch.setattr(loop._assessment_method, "test", mock_test)

    loop.run_experiment()

    assert assess_is_called
    assert test_is_called


def test_log_experiment_fails_without_running_experiment(
    experiment_config_path_k_fold_cross_validation,
    mock_data_path,
    mock_experiment_path,
) -> None:
    loop = Loop(
        experiment_configuration_path=experiment_config_path_k_fold_cross_validation,
        dataset_path=mock_data_path,
    )
    loop.setup_experiment(LitReCNN, mock_experiment_path)
    with pytest.raises(AttributeError):
        loop.log_experiment()


def test_log_experiment(
    monkeypatch,
    experiment_config_path_k_fold_cross_validation,
    mock_data_path,
    mock_experiment_path,
    tmp_path,
) -> None:
    loop = Loop(
        experiment_configuration_path=experiment_config_path_k_fold_cross_validation,
        dataset_path=mock_data_path,
    )
    loop.setup_experiment(LitReCNN, mock_experiment_path)
    monkeypatch.setattr(loop._assessment_method, "internal_testing_loss", 0.16)
    monkeypatch.setattr(loop._assessment_method, "external_testing_loss", 0.18)
    monkeypatch.setattr(
        loop._assessment_method, "best_model_path", Path("./best_model_path")
    )

    loop.log_experiment()

    logged_result = read_yaml_as_configuration(
        tmp_path.joinpath("loop_result.yaml"), LoopResult
    )
    assert logged_result.best_model_path == Path("best_model_path")
    assert logged_result.internal_testing_loss == pytest.approx(0.16)
    assert logged_result.external_testing_loss == pytest.approx(0.18)
    assert logged_result.difference_internal_external_loss == pytest.approx(-0.02)
