"""CIFAR-10 dataset adapter."""

from pathlib import Path

from torchvision import datasets, transforms

from diffusion.datasets.base import DiffusionDataModule


class CIFAR10DataModule(DiffusionDataModule):
    def __init__(self, data_dir: str, image_size: int, download: bool = False):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.download = download
        self.transform = transforms.Compose([
            transforms.Resize(image_size), transforms.CenterCrop(image_size), transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    @property
    def sample_shape(self) -> tuple[int, ...]:
        return (3, self.image_size, self.image_size)

    def dataset(self):
        return datasets.CIFAR10(self.data_dir, train=True, download=self.download, transform=self.transform)
