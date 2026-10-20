"""Common interface for optional sampling-time interventions."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class SamplingState:
    """Current sample population and time indices between reverse updates."""

    samples: torch.Tensor
    timestep: int
    previous_timestep: int
    metadata: dict[str, Any] = field(default_factory=dict)


class SamplingModifier(ABC):
    """Optional hook applied after a sampler update during generation.

    A modifier may return a different number of population members, which lets
    a future birth-death implementation represent particle birth/death directly.
    It must return a state whose samples can be consumed by the next sampler step.
    """

    @abstractmethod
    def apply(self, state: SamplingState, *, model, scheduler, generator) -> SamplingState:
        """Modify a sampling state; this hook is never used during training."""
