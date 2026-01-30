from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Type

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    """
    Basic residual block (as in the ResNet.md example).

    Notes:
    - This is a "basic block" style residual unit: 3x3 conv -> 3x3 conv.
    - If (stride != 1) or (in_ch != out_ch), we use a 1x1 conv shortcut.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(
            out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.skip_connection = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.skip_connection(x)
        out = F.relu(out)
        return out


class CustomResNet(nn.Module):
    """
    A simple ResNet-like model (CIFAR-style stem) based on ResNet.md example.

    - Stem: 3x3 conv (stride=1) + BN + ReLU
    - Stages: [64,128,256,512] with configurable blocks per stage
    - Head: AdaptiveAvgPool2d(1) + Linear(512 -> num_classes)
    """

    def __init__(
        self,
        block: Type[nn.Module],
        layers: Sequence[int],
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        if len(layers) != 4:
            raise ValueError("layers must have 4 elements, e.g. [2,2,2,2] for ResNet-18")

        self.initial_channels = 64

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._create_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._create_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._create_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._create_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _create_layer(
        self, block: Type[nn.Module], out_ch: int, num_layers: int, stride: int
    ) -> nn.Sequential:
        if num_layers < 1:
            raise ValueError("Each stage must have at least 1 block")

        layers: List[nn.Module] = []
        layers.append(block(self.initial_channels, out_ch, stride))
        self.initial_channels = out_ch
        for _ in range(1, num_layers):
            layers.append(block(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def custom_resnet18(num_classes: int = 10) -> CustomResNet:
    """Factory: Custom ResNet-18 = [2,2,2,2]."""

    return CustomResNet(Block, [2, 2, 2, 2], num_classes=num_classes)

