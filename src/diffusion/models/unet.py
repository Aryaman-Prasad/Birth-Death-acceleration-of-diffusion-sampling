"""A compact time-conditioned U-Net that predicts DDPM noise."""

import torch
from torch import nn
import torch.nn.functional as F

from diffusion.models.base import DiffusionModel
from diffusion.models.time_embedding import SinusoidalTimeEmbedding


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_projection = nn.Linear(time_channels, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_projection(F.silu(time_embedding))[:, :, None, None]
        return self.conv2(F.silu(self.norm2(h))) + self.skip(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class SimpleUNet(DiffusionModel):
    """Small two-resolution U-Net suitable for low-resolution experiments."""

    def __init__(self, image_channels: int = 3, base_channels: int = 64, time_channels: int = 256):
        super().__init__()
        if base_channels % 8:
            raise ValueError("base_channels must be divisible by 8 for GroupNorm")
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels), nn.Linear(base_channels, time_channels), nn.SiLU(),
            nn.Linear(time_channels, time_channels),
        )
        self.input = nn.Conv2d(image_channels, base_channels, 3, padding=1)
        self.down_block1 = ResidualBlock(base_channels, base_channels, time_channels)
        self.downsample = nn.Conv2d(base_channels, base_channels, 3, stride=2, padding=1)
        self.down_block2 = ResidualBlock(base_channels, base_channels * 2, time_channels)
        self.middle1 = ResidualBlock(base_channels * 2, base_channels * 2, time_channels)
        self.middle2 = ResidualBlock(base_channels * 2, base_channels * 2, time_channels)
        self.upsample = Upsample(base_channels * 2)
        self.up_block = ResidualBlock(base_channels * 3, base_channels, time_channels)
        self.output_norm = nn.GroupNorm(8, base_channels)
        self.output = nn.Conv2d(base_channels, image_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        t = self.time_mlp(timesteps)
        skip = self.down_block1(self.input(x), t)
        h = self.down_block2(self.downsample(skip), t)
        h = self.middle2(self.middle1(h, t), t)
        h = self.upsample(h)
        if h.shape[-2:] != skip.shape[-2:]:
            h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
        h = self.up_block(torch.cat((h, skip), dim=1), t)
        return self.output(F.silu(self.output_norm(h)))
