"""ImageFolder adapter; supports both subdirectories and flat image folders."""

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from diffusion.datasets.base import DiffusionDataModule


class FlatImageDataset(Dataset):
    """Fallback dataset for flat image directories without class subfolders."""

    def __init__(self, root: Path, transform=None):
        self.root = Path(root)
        self.transform = transform
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.image_paths = sorted([
            p for p in self.root.iterdir()
            if p.is_file() and p.suffix.lower() in valid_exts
        ])
        if not self.image_paths:
            self.image_paths = sorted([
                p for p in self.root.rglob("*")
                if p.is_file() and p.suffix.lower() in valid_exts
            ])
        if not self.image_paths:
            raise FileNotFoundError(f"No image files found in {root}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, 0


class ImageFolderDataModule(DiffusionDataModule):
    def __init__(self, data_dir: str, image_size: int):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize(image_size), transforms.CenterCrop(image_size), transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    @property
    def sample_shape(self) -> tuple[int, ...]:
        return (3, self.image_size, self.image_size)

    def dataset(self):
        try:
            ds = datasets.ImageFolder(self.data_dir, transform=self.transform)
            if len(ds) > 0:
                return ds
        except Exception:
            pass
        return FlatImageDataset(self.data_dir, transform=self.transform)
