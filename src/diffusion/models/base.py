"""Common interface for models that predict the diffusion noise."""

from abc import ABC, abstractmethod

import torch
from torch import nn


class DiffusionModel(nn.Module, ABC):
    """A denoiser maps a noisy batch and integer times to model predictions."""

    @abstractmethod
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Return a tensor matching ``x`` (epsilon for this baseline)."""
