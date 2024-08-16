from typing import Callable

from pydantic import BaseModel

import torch
import torch.nn as nn

from monai.networks.layers.factories import Conv
import lightning as L


class ReCNNConfiguration(BaseModel):
    n_layers: int
    spatial_dims: int
    in_channels: int
    intermediate_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    padding: str = "same"


class ReCNN(nn.Module):
    def __init__(
        self,
        configuration: ReCNNConfiguration,
    ):
        nn.Module.__init__(self)
        conv_type: Callable = Conv[Conv.CONV, configuration.spatial_dims]

        self.conv1 = conv_type(
            in_channels=configuration.in_channels,
            out_channels=configuration.intermediate_channels,
            kernel_size=configuration.kernel_size,
            stride=configuration.stride,
            padding=configuration.padding,
        )
        self.relu1 = nn.ReLU()

        intermediate_layers_definition = []
        # Add n_layers of conv and relu
        for _ in range(2, configuration.n_layers):
            intermediate_layers_definition.append(
                conv_type(
                    in_channels=configuration.intermediate_channels,
                    out_channels=configuration.intermediate_channels,
                    kernel_size=configuration.kernel_size,
                    stride=configuration.stride,
                    padding=configuration.padding,
                )
            )
            intermediate_layers_definition.append(nn.ReLU())
        self.intermediate_layers = nn.Sequential(*intermediate_layers_definition)

        self.final_conv = conv_type(
            in_channels=configuration.intermediate_channels,
            out_channels=configuration.out_channels,
            kernel_size=configuration.kernel_size,
            stride=configuration.stride,
            padding=configuration.padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.intermediate_layers(x)
        x = self.final_conv(x)
        return x + x_in


class LitReCNN(L.LightningModule):
    def __init__(self, configuration: ReCNNConfiguration):
        super().__init__()
        self.model = ReCNN(configuration)
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

    def _test_step(self, batch, batch_idx):
        # batch_idx is needed for lighting
        # name it here to avoid linting errors
        _ = batch_idx
        x, y = batch["img"], batch["lab"]
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self._test_step(batch, batch_idx)
        self.log("test_loss", loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)  # type: ignore
        return optimizer


def _setup_seeded_model(
    model_type: type[L.LightningModule], model_config: ReCNNConfiguration, seed: int
) -> L.LightningModule:
    L.seed_everything(seed, workers=True)
    return model_type(model_config)
