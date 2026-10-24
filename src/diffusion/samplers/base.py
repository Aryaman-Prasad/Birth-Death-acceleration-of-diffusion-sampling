"""Interface for reverse-process update rules."""

from abc import ABC, abstractmethod

import torch
from torch import nn

from diffusion.schedulers.base import NoiseScheduler


class Sampler(ABC):
    def initial_samples(self, shape: tuple[int, ...], device: torch.device,
                        scheduler: NoiseScheduler, timesteps: torch.Tensor,
                        generator: torch.Generator | None = None) -> torch.Tensor:
        """Create the state from which this sampler begins its trajectory."""
        return torch.randn(shape, device=device, generator=generator)

    def sampling_metadata(self, timestep: int, previous_timestep: int,
                          scheduler: NoiseScheduler) -> dict:
        """Describe this update to modifiers; ordinary reverse samplers need none."""
        return {}

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
