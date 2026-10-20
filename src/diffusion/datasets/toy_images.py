"""Tiny deterministic RGB pattern data for quick end-to-end image runs."""

import torch
from torch.utils.data import Dataset

from diffusion.datasets.base import DiffusionDataModule


class _PatternDataset(Dataset):
    def __init__(self, size: int, count: int, seed: int):
        generator = torch.Generator().manual_seed(seed)
        coordinates = torch.linspace(-1, 1, size)
        yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
        colors = torch.tensor([
            [0.9, -0.5, -0.5], [-0.5, 0.9, -0.5], [-0.5, -0.5, 0.9],
            [0.9, 0.8, -0.5], [0.8, -0.5, 0.8], [-0.5, 0.8, 0.8],
        ])
        images = []
        for index in range(count):
            kind = index % 4
            cx = torch.rand((), generator=generator) * 0.7 - 0.35
            cy = torch.rand((), generator=generator) * 0.7 - 0.35
            radius = 0.3 + torch.rand((), generator=generator) * 0.18
            if kind == 0:
                mask = (xx - cx).square() + (yy - cy).square() < radius.square()
            elif kind == 1:
                mask = ((xx - cx).abs() < radius * 0.45) & ((yy - cy).abs() < radius)
            elif kind == 2:
                mask = ((xx - cx).abs() < radius) & ((yy - cy).abs() < radius * 0.45)
            else:
                mask = (xx - yy - cx + cy).abs() < radius * 0.35
            image = torch.full((3, size, size), -1.0)
            color = colors[index % len(colors), :, None, None]
            image = torch.where(mask[None, :, :], color, image)
            images.append(image)
        self.images = torch.stack(images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], index % 4


class ToyImageDataModule(DiffusionDataModule):
    """Procedural circles, bars, and diagonals, generated once in memory."""

    def __init__(self, image_size: int = 16, count: int = 256, seed: int = 0):
        if image_size < 4 or count < 1:
            raise ValueError("toy images require image_size >= 4 and count >= 1")
        self.image_size = image_size
        self.count = count
        self.seed = seed

    @property
    def sample_shape(self) -> tuple[int, ...]:
        return (3, self.image_size, self.image_size)

    def dataset(self):
        return _PatternDataset(self.image_size, self.count, self.seed)
