"""Interface for target distributions used by particle samplers."""

from abc import ABC, abstractmethod

import torch


class TargetDistribution(ABC):
    """A target density and score on a declared sampler state space.

    Targets in this project currently use normalized noisy coordinates
    ``z = x_t / sqrt(alpha_bar_t)``. A target defined in another coordinate
    system should override this class attribute; sampling modifiers can then
    convert the sampler state before evaluating its density.
    """

    state_coordinates = "normalized_x_t"

    @abstractmethod
    def log_prob(self, samples: torch.Tensor, timestep: int,
                 scheduler) -> torch.Tensor:
        """Return one unnormalized or normalized log density per sample."""

    @abstractmethod
    def score(self, samples: torch.Tensor, timestep: int,
              scheduler) -> torch.Tensor:
        """Return the gradient of log density, matching ``samples`` shape."""
