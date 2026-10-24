"""Dataset adapters used by the training loop."""

from diffusion.datasets.base import DiffusionDataModule
from diffusion.datasets.cifar10 import CIFAR10DataModule
from diffusion.datasets.image_folder import ImageFolderDataModule
from diffusion.datasets.toy_images import ToyImageDataModule
from diffusion.datasets.toy_mixture import ToyGaussianMixtureDataModule


def make_data_module(name: str, data_dir: str, image_size: int, download: bool = False,
                     toy_size: int = 512, seed: int = 0, train: bool = True):
    if name.lower() == "cifar10":
        return CIFAR10DataModule(data_dir, image_size, download, train)
    if name.lower() in {"imagefolder", "image-folder"}:
        return ImageFolderDataModule(data_dir, image_size)
    if name.lower() == "toy-images":
        return ToyImageDataModule(image_size, toy_size, seed)
    if name.lower() in {"toy-mixture", "toy-gaussian-mixture"}:
        return ToyGaussianMixtureDataModule(toy_size, seed)
    raise ValueError(f"Unknown dataset: {name}")


__all__ = ["DiffusionDataModule", "CIFAR10DataModule", "ImageFolderDataModule",
           "ToyImageDataModule", "ToyGaussianMixtureDataModule", "make_data_module"]
