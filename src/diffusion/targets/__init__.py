"""Known or user-supplied target densities for sampling experiments."""

from diffusion.targets.base import TargetDistribution
from diffusion.targets.toy_mixture import ToyGaussianMixtureTarget

__all__ = ["TargetDistribution", "ToyGaussianMixtureTarget"]
