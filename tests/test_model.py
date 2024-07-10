from superresassess.model import ReCNN
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


class TestReCNN:

    @pytest.mark.parametrize("params, input_shape, expected_shape", CASES_3D)
    def test_shape(self, params, input_shape, expected_shape):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        net = ReCNN(**params).to(device)
        with eval_mode(net):
            result = net(torch.randn(input_shape).to(device))
        assert all([a == b for a, b in zip(result.shape, expected_shape)])

    def test_script(self):
        net = ReCNN(
            n_layers=10,
            spatial_dims=3,
            in_channels=2,
            intermediate_channels=4,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding="same",
        )
        test_data = torch.randn(16, 2, 32, 32, 32)
        try_script_save(net, test_data)
