"""Thin Lightning adapters around raw datasets and caller-owned transforms."""

from collections.abc import Callable
from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset

from data.datasets import Caltech101Dataset, EuroSATDataset, TinyBENDataset


class TransformedDataset(Dataset):
    def __init__(self, dataset: Dataset, transform: Callable | None) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        return (self.transform(image) if self.transform else image), label


class ClassificationDataModule(L.LightningDataModule):
    dataset_class: type[Dataset]

    def __init__(
        self,
        dataset_path: str | Path,
        batch_size: int = 32,
        num_workers: int = 4,
        transform: Callable | None = None,
        *,
        train_transform: Callable | None = None,
        eval_transform: Callable | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.dataset_path = Path(dataset_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.train_transform = train_transform if train_transform is not None else transform
        self.eval_transform = eval_transform if eval_transform is not None else transform
        self.task = self.dataset_class.task
        self.num_classes = self.dataset_class.num_classes
        self.num_channels = self.dataset_class.num_channels
        self._raw_datasets: dict[str, Dataset] = {}

    def setup(self, stage: str | None = None) -> None:
        if not self._raw_datasets:
            kwargs = {} if self.dataset_class is TinyBENDataset else {"seed": self.seed}
            self._raw_datasets = {
                split: self.dataset_class(self.dataset_path, split, **kwargs)
                for split in ("train", "validation", "test")
            }
            self.raw_train_dataset = self._raw_datasets["train"]
            self.class_names = self.raw_train_dataset.class_names
        self.train_dataset = TransformedDataset(self.raw_train_dataset, self.train_transform)
        self.val_dataset = TransformedDataset(self._raw_datasets["validation"], self.eval_transform)
        self.test_dataset = TransformedDataset(self._raw_datasets["test"], self.eval_transform)

    def _loader(self, dataset: Dataset, *, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            generator=torch.Generator().manual_seed(self.seed),
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_dataset)

    def feature_dataloader(self) -> DataLoader:
        """Training samples in stable order using only evaluation preprocessing."""
        return self._loader(TransformedDataset(self.raw_train_dataset, self.eval_transform))


class TinyBENDataModule(ClassificationDataModule):
    dataset_class = TinyBENDataset


class EuroSATDataModule(ClassificationDataModule):
    dataset_class = EuroSATDataset


class Caltech101DataModule(ClassificationDataModule):
    dataset_class = Caltech101Dataset
