"""
Dataset loaders and synthetic distribution generators.
Supports:
1. 2D Multimodal Toy Distributions: 8-Gaussians, 25-Gaussians, Swiss Roll, Two Moons.
2. Standard Image Benchmarks: MNIST, Fashion-MNIST, CIFAR-10.
"""

import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple
from torchvision import datasets, transforms


class Synthetic2DDataset(Dataset):
    """
    Synthetic 2D distribution generator for multimodal diffusion and birth-death benchmarking.
    """
    def __init__(self, name: str = "8gaussians", num_samples: int = 100000, seed: int = 42):
        super().__init__()
        self.name = name.lower()
        self.num_samples = num_samples
        np.random.seed(seed)

        self.data = self._generate_data()

    def _generate_data(self) -> torch.Tensor:
        n = self.num_samples
        if self.name == "8gaussians":
            scale = 4.0
            centers = [
                (scale * math.cos(2 * math.pi * i / 8), scale * math.sin(2 * math.pi * i / 8))
                for i in range(8)
            ]
            centers = np.array(centers, dtype=np.float32)
            indices = np.random.choice(len(centers), size=n)
            noise = np.random.randn(n, 2).astype(np.float32) * 0.4
            data = centers[indices] + noise
            # Normalize to roughly [-2, 2]
            data = data / 2.0

        elif self.name == "25gaussians":
            range_val = 4.0
            coords = np.linspace(-range_val, range_val, 5)
            centers = np.array([[x, y] for x in coords for y in coords], dtype=np.float32)
            indices = np.random.choice(len(centers), size=n)
            noise = np.random.randn(n, 2).astype(np.float32) * 0.25
            data = (centers[indices] + noise) / 2.0

        elif self.name == "swissroll":
            # Pure NumPy Swiss Roll generation
            phi = np.sqrt(np.random.uniform(0, 1, n)) * 3 * math.pi + 1.5 * math.pi
            noise = np.random.randn(n) * 0.3
            x = (phi + noise) * np.cos(phi)
            y = (phi + noise) * np.sin(phi)
            data = np.stack([x, y], axis=1).astype(np.float32)
            data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) * 1.5)

        elif self.name == "moons":
            # Pure NumPy Two Moons generation
            n_samples_out = n // 2
            n_samples_in = n - n_samples_out
            outer_circ_x = np.cos(np.linspace(0, math.pi, n_samples_out))
            outer_circ_y = np.sin(np.linspace(0, math.pi, n_samples_out))
            inner_circ_x = 1.0 - np.cos(np.linspace(0, math.pi, n_samples_in))
            inner_circ_y = 1.0 - np.sin(np.linspace(0, math.pi, n_samples_in)) - 0.5
            
            x = np.append(outer_circ_x, inner_circ_x) + np.random.randn(n) * 0.08
            y = np.append(outer_circ_y, inner_circ_y) + np.random.randn(n) * 0.08
            data = np.stack([x, y], axis=1).astype(np.float32)
            data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) * 1.5)

        else:
            raise ValueError(f"Unknown synthetic distribution: {self.name}. Choose from '8gaussians', '25gaussians', 'swissroll', 'moons'.")

        return torch.tensor(data, dtype=torch.float32)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


def get_2d_dataloader(
    name: str = "8gaussians",
    batch_size: int = 256,
    num_samples: int = 100000,
    seed: int = 42,
    num_workers: int = 0
) -> DataLoader:
    """
    Returns DataLoader for 2D synthetic dataset.
    """
    dataset = Synthetic2DDataset(name=name, num_samples=num_samples, seed=seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


def get_image_dataloader(
    dataset_name: str = "mnist",
    data_dir: str = "./data",
    batch_size: int = 128,
    image_size: int = 28,
    num_workers: int = 2,
    train: bool = True
) -> DataLoader:
    """
    Returns DataLoader for image datasets (MNIST, FashionMNIST, CIFAR10).
    Images normalized to [-1, 1] range as standard in diffusion models.
    """
    dataset_name = dataset_name.lower()
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) if dataset_name != "cifar10" else transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    if dataset_name == "mnist":
        dataset = datasets.MNIST(root=data_dir, train=train, download=True, transform=transform)
    elif dataset_name == "fashionmnist":
        dataset = datasets.FashionMNIST(root=data_dir, train=train, download=True, transform=transform)
    elif dataset_name == "cifar10":
        dataset = datasets.CIFAR10(root=data_dir, train=train, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported image dataset: {dataset_name}. Choose 'mnist', 'fashionmnist', or 'cifar10'.")

    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers, drop_last=True)
