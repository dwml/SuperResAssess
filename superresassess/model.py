from typing import Callable

import torch
import torch.nn as nn

from monai.networks.layers.factories import Conv
import lightning as L


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


class LitReCNN(L.LightningModule):
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
        super().__init__()
        self.model = ReCNN(
            n_layers=n_layers,
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            intermediate_channels=intermediate_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.loss = torch.nn.MSELoss()

    def training_step(self, batch, batch_idx):
        # batch_idx is needed for lighting
        # name it here to avoid linting errors
        _ = batch_idx
        x, y = batch["img"], batch["lab"]
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # batch_idx is needed for lighting
        # name it here to avoid linting errors
        _ = batch_idx
        x, y = batch["img"], batch["lab"]
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log("validation_loss", loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
