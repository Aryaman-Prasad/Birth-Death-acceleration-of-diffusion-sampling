"""Shared state descriptions passed between samplers and modifiers."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SamplingContext:
    """Coordinates and noise level of the samples returned by one sampler step.

    ``state_timestep`` is ``None`` when the update returned a clean terminal
    sample rather than a state associated with a scheduler noise level. The
    pipeline fills in the zero-based trajectory position and endpoint flag
    after asking the sampler to describe its state semantics.
    """

    state_timestep: int | None
    state_coordinates: Literal["x_t", "normalized_x_t", "clean_x0"]
    is_clean_sample: bool = False
    # Filled by SamplingPipeline so modifiers can schedule work consistently
    # over the complete reverse trajectory.
    step_index: int = 0
    num_steps: int = 1
    has_clean_endpoint: bool = False
