from typing import Callable

import torch
import torch.nn as nn

from monai.networks.layers.factories import Conv


class ReCNN(nn.Module):
    def __init__(
        self,
        n_layers: int,
        spatial_dims: int,
        in_channels: int,
        intermediate_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: str = "same",
    ):
        nn.Module.__init__(self)
        conv_type: Callable = Conv[Conv.CONV, spatial_dims]

        self.conv1 = conv_type(
            in_channels=in_channels,
            out_channels=intermediate_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.relu1 = nn.ReLU()

        intermediate_layers_definition = []
        # Add n_layers of conv and relu
        for _ in range(2, n_layers):
            intermediate_layers_definition.append(
                conv_type(
                    in_channels=intermediate_channels,
                    out_channels=intermediate_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
            intermediate_layers_definition.append(nn.ReLU())
        self.intermediate_layers = nn.Sequential(*intermediate_layers_definition)

        self.final_conv = conv_type(
            in_channels=intermediate_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.intermediate_layers(x)
        x = self.final_conv(x)
        return x + x_in
