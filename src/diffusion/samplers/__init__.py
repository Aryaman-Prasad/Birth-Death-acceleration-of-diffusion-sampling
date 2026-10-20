"""Reverse-process update rules."""

from diffusion.samplers.base import Sampler
from diffusion.samplers.ddim import DDIMSampler
from diffusion.samplers.ddpm import DDPMSampler


def make_sampler(name: str, eta: float = 0.0) -> Sampler:
    if name == "ddpm":
        return DDPMSampler()
    if name == "ddim":
        return DDIMSampler(eta=eta)
    raise ValueError(f"Unknown sampler: {name}")


__all__ = ["Sampler", "DDPMSampler", "DDIMSampler", "make_sampler"]
