"""Shared state descriptions passed between samplers and modifiers."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SamplingContext:
    """Coordinates and noise level of the samples returned by one sampler step.

    ``state_timestep`` is ``None`` when the update returned a clean terminal
    sample rather than a state associated with a scheduler noise level.
    """

    state_timestep: int | None
    state_coordinates: Literal["x_t", "normalized_x_t", "clean_x0"]
    is_clean_sample: bool = False
