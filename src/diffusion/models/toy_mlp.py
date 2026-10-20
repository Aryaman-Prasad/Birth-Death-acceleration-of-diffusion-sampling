"""A small MLP denoiser for vector-valued toy distributions."""

import torch
from torch import nn

from diffusion.models.base import DiffusionModel
from diffusion.models.time_embedding import SinusoidalTimeEmbedding


class ToyMLP(DiffusionModel):
    def __init__(self, input_dim: int = 2, hidden_channels: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_channels = hidden_channels
        self.time_embedding = SinusoidalTimeEmbedding(hidden_channels)
        self.network = nn.Sequential(
            nn.Linear(input_dim + hidden_channels, hidden_channels), nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels), nn.SiLU(),
            nn.Linear(hidden_channels, input_dim),
        )

    def forward(self, noisy_samples: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if noisy_samples.ndim != 2 or noisy_samples.shape[-1] != self.input_dim:
            raise ValueError(f"ToyMLP expects samples shaped (batch, {self.input_dim})")
        time = self.time_embedding(timesteps)
        return self.network(torch.cat((noisy_samples, time), dim=-1))
