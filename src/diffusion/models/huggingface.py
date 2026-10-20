"""Adapter for pretrained unconditional Diffusers UNet2DModel checkpoints."""

import torch
from torch import nn

from diffusion.models.base import DiffusionModel


class HuggingFaceUNetModel(DiffusionModel):
    """Expose a pretrained Diffusers UNet2DModel through DiffusionModel."""

    def __init__(self, pretrained_model_id: str, revision: str | None = None,
                 subfolder: str | None = None):
        super().__init__()
        try:
            from diffusers import UNet2DModel
        except ImportError as exc:
            raise ImportError(
                "Pretrained Hugging Face models require the optional dependencies; "
                "install with `python -m pip install -e '.[pretrained]'`."
            ) from exc
        self.pretrained_model_id = pretrained_model_id
        self.revision = revision
        self.subfolder = subfolder
        self.unet = UNet2DModel.from_pretrained(pretrained_model_id, revision=revision,
                                                subfolder=subfolder)
        config = self.unet.config
        size = config.sample_size
        if isinstance(size, int):
            spatial_size = (size, size)
        else:
            spatial_size = tuple(size)
        self.sample_shape = (config.in_channels, *spatial_size)

    def forward(self, noisy_images: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return self.unet(noisy_images, timesteps).sample
