"""Dataset adapter interface."""

from abc import ABC, abstractmethod

from torch.utils.data import DataLoader, Dataset


class DiffusionDataModule(ABC):
    """Builds a PyTorch dataset and a shuffled training data loader."""

    @property
    @abstractmethod
    def sample_shape(self) -> tuple[int, ...]:
        """Shape of one data example, excluding its batch dimension."""

    @abstractmethod
    def dataset(self) -> Dataset:
        """Return a dataset yielding image tensors or (image, label) pairs."""

    def make_dataloader(self, batch_size: int, num_workers: int = 4) -> DataLoader:
        data = self.dataset()
        if len(data) == 0:
            raise ValueError("Dataset contains no examples")
        return DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                          pin_memory=True, drop_last=False)
