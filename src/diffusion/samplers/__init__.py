"""Reverse-process update rules."""

from diffusion.samplers.base import Sampler, SamplingContext
from diffusion.samplers.ddim import DDIMSampler
from diffusion.samplers.ddpm import DDPMSampler
from diffusion.samplers.langevin import AnnealedLangevinSampler, LangevinSampler


def make_sampler(name: str, eta: float = 0.0, *, langevin_step_size: float | None = None,
                 langevin_steps_per_level: int = 1, langevin_timestep: int = 0,
                 langevin_initial_scale: float = 1.0) -> Sampler:
    if name == "ddpm":
        return DDPMSampler()
    if name == "ddim":
        return DDIMSampler(eta=eta)
    if name in {"annealed-langevin", "ald"}:
        return AnnealedLangevinSampler(
            step_size=1e-7 if langevin_step_size is None else langevin_step_size,
            steps_per_level=langevin_steps_per_level,
        )
    if name in {"langevin", "ula"}:
        return LangevinSampler(
            timestep=langevin_timestep,
            step_size=1e-3 if langevin_step_size is None else langevin_step_size,
            initial_scale=langevin_initial_scale,
        )
    raise ValueError(f"Unknown sampler: {name}")


__all__ = ["Sampler", "SamplingContext", "DDPMSampler", "DDIMSampler", "LangevinSampler",
           "AnnealedLangevinSampler", "make_sampler"]
