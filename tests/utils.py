from monai.networks import convert_to_torchscript
from torch.utils.data import Dataset
import torch
import tempfile
import os


def try_script_save(net, *inputs, device=None, rtol=1e-4, atol=0.0):
    """
    Test the ability to save `net` as a Torchscript object, reload it, and
    apply inference. The value `inputs` is forward-passed through the original
    and loaded copy of the network and their results returned. The forward
    pass for both is done without gradient accumulation.

    The test will be performed with CUDA if available, else CPU.

    source: monai test utils
    """
    device = "cpu"
    with tempfile.TemporaryDirectory() as tempdir:
        convert_to_torchscript(
            model=net,
            filename_or_obj=os.path.join(tempdir, "model.ts"),
            verify=True,
            inputs=inputs,
            device=device,
            rtol=rtol,
            atol=atol,
        )


class MockDataset(Dataset):
    def __init__(self, size: tuple[int, ...], length: int):
        if not isinstance(size, tuple):
            raise TypeError(
                f"Size should be of type tuple but was of type: {type(size)}"
            )
        self.len = length
        self.data = torch.randn((length,) + size)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.data[index]

    def __len__(self) -> int:
        return self.len
