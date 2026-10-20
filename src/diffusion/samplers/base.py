"""Interface for reverse-process update rules."""

from abc import ABC, abstractmethod

import torch
from torch import nn

from diffusion.schedulers.base import NoiseScheduler


class Sampler(ABC):
    @abstractmethod
    def inference_timesteps(self, scheduler: NoiseScheduler, num_inference_steps: int | None,
                            device: torch.device) -> torch.Tensor:
        """Return descending training-time indices visited by this sampler."""

    @abstractmethod
    @torch.no_grad()
    def step(self, model: nn.Module, noisy_images: torch.Tensor, timestep: int,
             previous_timestep: int, scheduler: NoiseScheduler,
             generator: torch.Generator | None = None) -> torch.Tensor:
        """Apply one reverse-process update and return the next noisy batch."""
