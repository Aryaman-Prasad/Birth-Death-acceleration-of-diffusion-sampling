"""A low-dimensional Gaussian mixture for fast sampler experiments."""

import math

import torch
from torch.utils.data import TensorDataset

from diffusion.datasets.base import DiffusionDataModule


class ToyGaussianMixtureDataModule(DiffusionDataModule):
    """Eight Gaussian modes arranged on a ring in two dimensions."""

    def __init__(self, count: int = 2048, seed: int = 0, radius: float = 2.0,
                 standard_deviation: float = 0.15):
        if count < 1 or radius <= 0 or standard_deviation <= 0:
            raise ValueError("count, radius, and standard_deviation must be positive")
        self.count = count
        self.seed = seed
        self.radius = radius
        self.standard_deviation = standard_deviation

    @property
    def sample_shape(self) -> tuple[int, ...]:
        return (2,)

    def dataset(self):
        generator = torch.Generator().manual_seed(self.seed)
        angles = torch.arange(8, dtype=torch.float32) * (2 * math.pi / 8)
        centers = self.radius * torch.stack((angles.cos(), angles.sin()), dim=1)
        labels = torch.randint(0, len(centers), (self.count,), generator=generator)
        points = centers[labels] + self.standard_deviation * torch.randn(
            (self.count, 2), generator=generator
        )
        return TensorDataset(points, labels)
