"""Image classifiers, independent of datasets and training orchestration."""

from __future__ import annotations

import timm
import torch
from torch import nn


def _dropout_probability(dropout: bool | float) -> float:
    probability = 0.5 if dropout is True else float(dropout)
    if not 0 <= probability < 1:
        raise ValueError("dropout must be False, True (0.5), or a probability in [0, 1).")
    return probability


class ResidualBlock(nn.Module):
    """Downsample with two convolutions and a projected residual connection."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.convolutions = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.convolutions(images) + self.shortcut(images))


class ResNet6(nn.Module):
    """The assignment's five main convolutions followed by one linear layer."""

    def __init__(self, num_channels: int, num_classes: int, dropout: bool | float = False) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(num_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.blocks = nn.Sequential(ResidualBlock(64, 128), ResidualBlock(128, 256))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(_dropout_probability(dropout))
        self.fc = nn.Linear(256, num_classes)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.pool(self.blocks(self.stem(images))).flatten(1)
        return self.fc(self.dropout(features))

    def get_classifier(self) -> nn.Linear:
        return self.fc


def get_network(
    arch_name: str,
    num_channels: int,
    num_classes: int,
    pretrained: bool = False,
    dropout: bool | float = False,
) -> nn.Module:
    """timm adapts spectral inputs and resamples pretrained ViT patches/positions."""
    name = arch_name.lower().replace("-", "").replace("_", "")
    probability = _dropout_probability(dropout)
    if name == "resnet6":
        if pretrained:
            raise ValueError("ResNet-6 has no pretrained weights; use pretrained=False.")
        return ResNet6(num_channels, num_classes, probability)
    kwargs = dict(
        pretrained=pretrained,
        in_chans=num_channels,
        num_classes=num_classes,
        drop_rate=probability,
    )
    if name == "resnet18":
        return timm.create_model("resnet18", **kwargs)
    if name == "vittiny":
        return timm.create_model("vit_tiny_patch16_224", img_size=112, patch_size=8, **kwargs)
    raise ValueError(f"Unknown architecture {arch_name!r}; choose ResNet18, ResNet-6, or ViT-Tiny.")
