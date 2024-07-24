from superresassess.model import ReCNN, ReCNNConfiguration
from tests.utils import try_script_save
import pytest
import torch
from monai.networks import eval_mode


CASES_3D = [
    [
        {
            "spatial_dims": 3,
            "n_layers": 10,
            "in_channels": 1,
            "intermediate_channels": 64,
            "out_channels": 2,
            "kernel_size": 3,
            "stride": 1,
            "padding": "same",
        },
        (3, 1, 64, 64, 64),
        (3, 2, 64, 64, 64),
    ]
]


class TestReCNNConfig:
    correct_config = {
        "spatial_dims": 3,
        "n_layers": 10,
        "in_channels": 1,
        "intermediate_channels": 64,
        "out_channels": 2,
        "kernel_size": 3,
        "stride": 1,
        "padding": "same",
    }
    incorrect_config = {
        "spatial_dims": 3,
        "n_layers": 10,
        "in_channels": 1,
        "intermediate_channels": 64,
        "kernel_size": 3,
        "stride": 1,
        "padding": "same",
    }

    def test_config(self) -> None:
        config_object = ReCNNConfiguration(**self.correct_config)
        assert dict(config_object) == self.correct_config

    def test_model_initialization(self) -> None:
        config_object = ReCNNConfiguration(**self.correct_config)
        _ = ReCNN(config_object)

    def test_incorrect_config(self) -> None:
        with pytest.raises(Exception):
            _ = ReCNNConfiguration(**self.incorrect_config)


@pytest.mark.slow
class TestReCNN:
    @pytest.mark.parametrize("params, input_shape, expected_shape", CASES_3D)
    def test_shape(self, params, input_shape, expected_shape):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        config = ReCNNConfiguration(**params)
        net = ReCNN(config).to(device)
        with eval_mode(net):
            result = net(torch.randn(input_shape).to(device))
        assert result.shape == expected_shape

    def test_script(self):
        config = ReCNNConfiguration(
            n_layers=10,
            spatial_dims=3,
            in_channels=2,
            intermediate_channels=4,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding="same",
        )
        net = ReCNN(config)
        test_data = torch.randn(16, 2, 32, 32, 32)
        try_script_save(net, test_data)


class TestLightningWrapper:
    """Testing is difficult since:

    There is a logging in the training step and validation step that needs a lightning
    module to be registered to a trainer. The trainer needs a train dataloader
    implemented. I couldn't figure out how to register the lighting module to a
    trainer without the data loader so I couldn't test these steps.
    """

    def __init__(self): ...
