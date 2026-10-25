"""Common interface for optional sampling-time interventions."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch

from diffusion.sampling import SamplingContext


@dataclass
class SamplingState:
    """Current population plus its post-update noise level and coordinates."""

    samples: torch.Tensor
    timestep: int
    previous_timestep: int
    metadata: dict[str, Any] = field(default_factory=dict)
    context: SamplingContext | None = None


class SamplingModifier(ABC):
    """Optional hook applied after a sampler update during generation.

    A modifier may return a different number of population members, which lets
    a future birth-death implementation represent particle birth/death directly.
    It must return a state whose samples can be consumed by the next sampler step.
    """

    @abstractmethod
    def apply(self, state: SamplingState, *, model, scheduler, generator) -> SamplingState:
        """Modify a sampling state; this hook is never used during training."""
