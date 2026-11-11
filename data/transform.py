"""Training-set statistics and augmentations for RGB or multispectral images."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torchvision import transforms as T
from torchvision.transforms import functional as F


@dataclass(frozen=True)
class ChannelStatistics:
    percentiles: tuple[float, ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> ChannelStatistics:
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**{name: tuple(channels) for name, channels in values.items()})


def fit_statistics(raw_training_dataset, *, rgb: bool = False) -> ChannelStatistics:
    """Fit exact p99 and clipped/scaled moments using bounded channel histograms.

    Only pass the raw training split. RGB uses division by 255 instead of p99.
    Zero percentiles and constant-channel standard deviations use denominator 1.
    """
    if len(raw_training_dataset) == 0:
        raise ValueError("Cannot fit statistics on an empty training dataset.")
    bins = 256 if rgb else 65536
    histograms = None
    for index in range(len(raw_training_dataset)):
        image = np.asarray(raw_training_dataset[index][0])
        if image.ndim != 3 or image.size == 0 or image.dtype != (np.uint8 if rgb else np.uint16):
            raise ValueError("Expected raw uint8 RGB or uint16 multispectral images.")
        if rgb:
            image = image.transpose(2, 0, 1)
        if histograms is None:
            histograms = np.zeros((len(image), bins), dtype=np.int64)
        for counts, channel in zip(histograms, image, strict=True):
            counts += np.bincount(channel.ravel(), minlength=bins)

    percentiles, means, deviations = [], [], []
    levels = np.arange(bins, dtype=np.float64)
    for counts in histograms:
        count = int(counts.sum())
        percentile = 255.0
        if not rgb:
            # Linear interpolation matches numpy.percentile(..., 99).
            rank = (count - 1) * 0.99
            lower, upper = int(np.floor(rank)), int(np.ceil(rank))
            low, high = np.searchsorted(counts.cumsum(), [lower + 1, upper + 1])
            percentile = float(low + (rank - lower) * (high - low))
        values = np.minimum(levels / (percentile or 1.0), 1.0)
        mean = float(np.dot(counts, values) / count)
        std = float(np.sqrt(np.dot(counts, (values - mean) ** 2) / count))
        percentiles.append(percentile)
        means.append(mean)
        deviations.append(std if std > 1e-8 else 1.0)
    return ChannelStatistics(tuple(percentiles), tuple(means), tuple(deviations))


def fit_rgb_statistics(raw_training_dataset) -> ChannelStatistics:
    return fit_statistics(raw_training_dataset, rgb=True)


class PercentileScale:
    def __init__(self, percentiles: tuple[float, ...]) -> None:
        scale = torch.tensor(percentiles, dtype=torch.float32)
        self.scale = torch.where(scale == 0, 1.0, scale)[:, None, None]

    def __call__(self, image: np.ndarray | Tensor) -> Tensor:
        image = torch.as_tensor(image, dtype=torch.float32)
        return (image / self.scale.to(image.device)).clamp(0, 1)


@dataclass
class Brightness:
    """Multiply every band by the same random factor."""

    strength: float = 0.2
    p: float = 0.5

    def __call__(self, image: Tensor) -> Tensor:
        if torch.rand(()) >= self.p:
            return image
        factor = torch.empty(()).uniform_(1 - self.strength, 1 + self.strength).item()
        return (image * factor).clamp(0, 1)


@dataclass
class Contrast:
    """Adjust each band's contrast about its spatial mean with one shared factor."""

    strength: float = 0.2
    p: float = 0.5

    def __call__(self, image: Tensor) -> Tensor:
        if torch.rand(()) >= self.p:
            return image
        factor = torch.empty(()).uniform_(1 - self.strength, 1 + self.strength).item()
        mean = image.mean(dim=(-2, -1), keepdim=True)
        return (mean + factor * (image - mean)).clamp(0, 1)


@dataclass
class GrayScale:
    """Replace each band with the spectral mean, retaining the channel count."""

    p: float = 0.5

    def __call__(self, image: Tensor) -> Tensor:
        if torch.rand(()) >= self.p:
            return image
        return image.mean(dim=-3, keepdim=True).expand_as(image)


def _random_resized_crop(image: Tensor) -> Tensor:
    """Crop and resize back to the input resolution."""
    top, left, height, width = T.RandomResizedCrop.get_params(
        image, scale=(0.7, 1.0), ratio=(0.75, 4 / 3)
    )
    return F.resized_crop(image, top, left, height, width, list(image.shape[-2:]), antialias=True)


AUGMENTATIONS = ("flip", "crop", "cutout", "brightness", "contrast", "grayscale")


def build_transforms(
    statistics: ChannelStatistics,
    *,
    train: bool,
    image_size: int | None = None,
    augmentation: str = "flip",
    rgb: bool = False,
) -> T.Compose:
    """Scale, resize, optionally augment, then normalize with training statistics.

    Training always includes horizontal/vertical flips with p=0.5. RGB callers
    should set image_size for batching variable-sized photographs.
    """
    operations = [T.ToTensor() if rgb else PercentileScale(statistics.percentiles)]
    if image_size is not None:
        operations.append(T.Resize((image_size, image_size), antialias=True))
    if train:
        operations.extend([T.RandomHorizontalFlip(0.5), T.RandomVerticalFlip(0.5)])
        extra = {
            "crop": _random_resized_crop,
            "cutout": T.RandomErasing(p=0.5, scale=(0.02, 0.2), value=0),
            "brightness": Brightness(),
            "contrast": Contrast(),
            "grayscale": GrayScale(),
        }
        if augmentation != "flip":
            operations.append(extra[augmentation])
    operations.append(T.Normalize(statistics.mean, statistics.std))
    return T.Compose(operations)
