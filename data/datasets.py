"""Local datasets. Samples remain raw until a DataModule applies a transform."""

import csv
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from torch.utils.data import Dataset

SELECTED_BANDS = ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12")
SPLIT_FILES = {"train": "train.csv", "validation": "val.csv", "test": "test.csv"}


def read_multispectral(path: str | Path) -> np.ndarray:
    """Read uint16 CHW imagery by band name, never by assumed file ordering."""
    with rasterio.open(path) as source:
        descriptions = tuple((name or "").strip().upper() for name in source.descriptions)
        missing = [band for band in SELECTED_BANDS if descriptions.count(band) != 1]
        if missing:
            raise ValueError(f"{path}: missing or duplicate band descriptions: {missing}")
        indexes = [descriptions.index(band) + 1 for band in SELECTED_BANDS]
        if any(source.dtypes[index - 1] != "uint16" for index in indexes):
            raise ValueError(f"{path}: expected raw uint16 GeoTIFF bands")
        return source.read(indexes)


def _validate_split(split: str) -> None:
    if split not in SPLIT_FILES:
        raise ValueError(f"Unknown split {split!r}; choose train, validation, or test")


class TinyBENDataset(Dataset):
    """tiny-BEN CSV splits with a shared, alphabetical 19-class vocabulary."""

    task = "multilabel"
    num_channels = len(SELECTED_BANDS)
    num_classes = 19

    def __init__(self, dataset_path: str | Path, split: str = "train") -> None:
        _validate_split(split)
        self.dataset_path = Path(dataset_path)
        self.samples = []
        seen_paths: set[Path] = set()
        seen_ids: set[str] = set()
        vocabulary: set[str] = set()
        for name, filename in SPLIT_FILES.items():
            with (self.dataset_path / filename).open(newline="", encoding="utf-8-sig") as handle:
                for row in csv.DictReader(handle):
                    path = (self.dataset_path / row["image_path"]).resolve()
                    patch_id = row["patch_id"].strip()
                    if path in seen_paths or patch_id in seen_ids:
                        raise ValueError(f"Duplicate sample across CSV rows/splits: {patch_id}")
                    seen_paths.add(path)
                    seen_ids.add(patch_id)
                    labels = tuple(
                        label.strip() for label in row["labels"].split(";") if label.strip()
                    )
                    vocabulary.update(labels)
                    if name == split:
                        self.samples.append((path, labels))
        if not self.samples:
            raise ValueError(f"{SPLIT_FILES[split]} contains no samples")
        self.class_names = tuple(sorted(vocabulary))
        if len(self.class_names) != self.num_classes:
            raise ValueError(
                f"tiny-BEN needs 19 classes across all splits, found {len(vocabulary)}"
            )
        self.class_to_idx = {name: index for index, name in enumerate(self.class_names)}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        path, names = self.samples[index]
        labels = np.zeros(self.num_classes, dtype=np.float32)
        labels[[self.class_to_idx[name] for name in names]] = 1
        return read_multispectral(path), labels


def _folder_samples(root: Path, suffixes: set[str], expected_classes: int, split: str, seed: int):
    """Seeded per-class 70/10/20 split; fractional leftovers belong to test."""
    _validate_split(split)
    folders = sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and not path.name.startswith(".") and path.name != "BACKGROUND_Google"
    )
    class_names = tuple(path.name for path in folders)
    if len(folders) != expected_classes:
        raise ValueError(f"{root}: expected {expected_classes} class folders, found {len(folders)}")
    generator = np.random.default_rng(seed)
    samples = []
    for label, folder in enumerate(folders):
        paths = sorted(path for path in folder.iterdir() if path.suffix.lower() in suffixes)
        if len(paths) < 10:
            raise ValueError(f"{folder}: need at least 10 images for a nonempty 70/10/20 split")
        order = generator.permutation(len(paths))
        train_end = 7 * len(paths) // 10
        validation_end = train_end + len(paths) // 10
        indexes = {
            "train": order[:train_end],
            "validation": order[train_end:validation_end],
            "test": order[validation_end:],
        }[split]
        samples.extend((paths[index], label) for index in indexes)
    return class_names, samples


class EuroSATDataset(Dataset):
    """EuroSAT GeoTIFFs with a reproducible, stratified 70/10/20 split."""

    task = "multiclass"
    num_channels = len(SELECTED_BANDS)
    num_classes = 10

    def __init__(self, dataset_path: str | Path, split: str = "train", seed: int = 42) -> None:
        self.dataset_path = Path(dataset_path)
        self.class_names, self.samples = _folder_samples(
            self.dataset_path, {".tif", ".tiff"}, self.num_classes, split, seed
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
        path, label = self.samples[index]
        return read_multispectral(path), label


class Caltech101Dataset(Dataset):
    """Local RGB fallback using 70/10/20; replace with course splits when available."""

    task = "multiclass"
    num_channels = 3
    num_classes = 101

    def __init__(self, dataset_path: str | Path, split: str = "train", seed: int = 42) -> None:
        root = Path(dataset_path)
        if (root / "101_ObjectCategories").is_dir():
            root = root / "101_ObjectCategories"
        self.dataset_path = root
        self.class_names, self.samples = _folder_samples(
            root, {".jpg", ".jpeg", ".png"}, self.num_classes, split, seed
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Image.Image, int]:
        path, label = self.samples[index]
        with Image.open(path) as image:
            return image.convert("RGB"), label
