"""Extension point for birth-death sampling-time acceleration."""

from diffusion.acceleration.base import SamplingModifier, SamplingState


class BirthDeathAcceleration(SamplingModifier):
    """Placeholder for future particle birth/death dynamics.

    The baseline does not implement the method. A future implementation belongs
    in this modifier and can be attached to a SamplingPipeline without changing
    DDPM/DDIM update rules or the training loop.
    """

    def apply(self, state: SamplingState, *, model, scheduler, generator) -> SamplingState:
        raise NotImplementedError("Birth-death acceleration is a later project deliverable")
