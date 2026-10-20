"""Sinusoidal embeddings for integer diffusion timesteps."""

import math

import torch
from torch import nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        scale = math.log(10000) / max(half - 1, 1)
        frequencies = torch.exp(torch.arange(half, device=timesteps.device) * -scale)
        angles = timesteps.float()[:, None] * frequencies[None, :]
        embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        return F.pad(embedding, (0, 1)) if self.dimension % 2 else embedding
