"""Interface for target distributions used by particle samplers."""

from abc import ABC, abstractmethod

import torch


class TargetDistribution(ABC):
    """A target density and score on the sampler's current state space."""

    @abstractmethod
    def log_prob(self, samples: torch.Tensor, timestep: int,
                 scheduler) -> torch.Tensor:
        """Return one unnormalized or normalized log density per sample."""

    @abstractmethod
    def score(self, samples: torch.Tensor, timestep: int,
              scheduler) -> torch.Tensor:
        """Return the gradient of log density, matching ``samples`` shape."""
