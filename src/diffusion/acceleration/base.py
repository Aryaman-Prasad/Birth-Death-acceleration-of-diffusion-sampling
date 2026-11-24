"""Common interface for optional sampling-time interventions."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch

from diffusion.sampling import SamplingContext


def coordinate_noise_std(samples: torch.Tensor, timestep: int | None, scheduler,
                         coordinates: str) -> torch.Tensor:
    """Return the forward-noise standard deviation in the state's coordinates.

    Clone jitter is optional and should match the coordinate system in which
    the particle tensor is stored. In particular, normalized ``z=x_t/sqrt(a_bar)``
    states have noise scale ``sqrt((1-a_bar)/a_bar)``, while raw ``x_t`` states
    have scale ``sqrt(1-a_bar)``.
    """
    if timestep is None or not hasattr(scheduler, "alpha_bars"):
        raise ValueError("coordinate-aware clone jitter requires a scheduler timestep")
    if not 0 <= int(timestep) < len(scheduler.alpha_bars):
        raise ValueError("clone-jitter timestep is outside the scheduler range")
    alpha_bar = scheduler.alpha_bars[int(timestep)].to(samples)
    if coordinates == "x_t":
        variance = 1.0 - alpha_bar
    elif coordinates == "normalized_x_t":
        variance = (1.0 - alpha_bar) / alpha_bar
    else:
        raise ValueError(f"clone jitter is undefined for coordinates {coordinates!r}")
    return variance.clamp_min(0).sqrt()


@dataclass
class SamplingState:
    """Post-update population and the source state used for its transition."""

    samples: torch.Tensor
    timestep: int
    previous_timestep: int
    metadata: dict[str, Any] = field(default_factory=dict)
    context: SamplingContext | None = None
    source_samples: torch.Tensor | None = None


class SamplingModifier(ABC):
    """Optional hook applied after a sampler update during generation.

    A modifier may return a different number of population members, which lets
    a future birth-death implementation represent particle birth/death directly.
    It must return a state whose samples can be consumed by the next sampler step.
    """

    @abstractmethod
    def apply(self, state: SamplingState, *, model, scheduler, generator) -> SamplingState:
        """Modify a sampling state; this hook is never used during training."""
