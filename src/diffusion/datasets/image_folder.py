"""ImageFolder adapter; expects one subdirectory per image class."""

from pathlib import Path

from torchvision import datasets, transforms

from diffusion.datasets.base import DiffusionDataModule


class ImageFolderDataModule(DiffusionDataModule):
    def __init__(self, data_dir: str, image_size: int):
        self.data_dir = Path(data_dir)
        self.transform = transforms.Compose([
            transforms.Resize(image_size), transforms.CenterCrop(image_size), transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    @property
    def sample_shape(self) -> tuple[int, ...]:
        return (3, self.image_size, self.image_size)

    def dataset(self):
        return datasets.ImageFolder(self.data_dir, transform=self.transform)
