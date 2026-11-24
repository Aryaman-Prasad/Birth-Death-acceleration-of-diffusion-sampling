"""CIFAR-10 dataset adapter."""

from pathlib import Path

from torchvision import datasets, transforms

# Fast CDN mirror to bypass slow cs.toronto.edu server throttling (~60 kB/s -> ~50 MB/s)
datasets.CIFAR10.url = (
    "https://github.com/Digital-Media/cv_data/releases/download/cifar-10/cifar-10-python.tar.gz"
)

from diffusion.datasets.base import DiffusionDataModule


class CIFAR10DataModule(DiffusionDataModule):
    def __init__(self, data_dir: str, image_size: int, download: bool = False,
                 train: bool = True):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.download = download
        self.train = train
        self.transform = transforms.Compose([
            transforms.Resize(image_size), transforms.CenterCrop(image_size), transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    @property
    def sample_shape(self) -> tuple[int, ...]:
        return (3, self.image_size, self.image_size)

    def dataset(self):
        return datasets.CIFAR10(self.data_dir, train=self.train, download=self.download,
                                transform=self.transform)
