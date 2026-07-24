"""Small FP vs binary classifiers on spectrogram features."""

from __future__ import annotations

import torch
import torch.nn as nn

from bnn.layers import BinaryConv2d, BinaryLinear, BiRealBlock
from bnn.ste import clip_weights_


class FP32AudioCNN(nn.Module):
    def __init__(self, n_classes: int = 8, channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(channels, channels * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels * 2, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(channels * 2, n_classes)

    def forward(self, x):
        return self.head(self.net(x).flatten(1))


class BinaryAudioCNN(nn.Module):
    """FP stem/head, binary middle — audio-feature analogue of Bi-Real."""

    def __init__(self, n_classes: int = 8, channels: int = 32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels, momentum=0.9),
        )
        self.b1 = BiRealBlock(channels)
        self.pool = nn.MaxPool2d(2)
        self.down = nn.Sequential(
            BinaryConv2d(channels, channels * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels * 2, momentum=0.9),
        )
        self.skip = nn.Conv2d(channels, channels * 2, 1, bias=False)
        self.b2 = BiRealBlock(channels * 2)
        self.pool2 = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(channels * 2, n_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.b1(x)
        x = self.pool(x)
        x = self.down(x) + self.skip(x)
        x = self.b2(x)
        return self.head(self.pool2(x).flatten(1))

    def clip_weights(self):
        clip_weights_(self)


class BinaryAudioMLP(nn.Module):
    def __init__(self, n_mels: int = 40, frames: int = 32, hidden: int = 128, n_classes: int = 8):
        super().__init__()
        d = n_mels * frames
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(d, hidden)
        self.bn1 = nn.BatchNorm1d(hidden, momentum=0.9)
        self.fc2 = BinaryLinear(hidden, hidden)
        self.bn2 = nn.BatchNorm1d(hidden, momentum=0.9)
        self.fc3 = nn.Linear(hidden, n_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.bn1(self.fc1(x))
        x = self.bn2(self.fc2(x))
        return self.fc3(x)

    def clip_weights(self):
        clip_weights_(self)


def build_audio_model(name: str, n_classes: int = 8, channels: int = 32) -> nn.Module:
    name = name.lower()
    if name in ("fp32", "fp32_cnn"):
        return FP32AudioCNN(n_classes=n_classes, channels=channels)
    if name in ("binary", "binary_cnn"):
        return BinaryAudioCNN(n_classes=n_classes, channels=channels)
    if name in ("binary_mlp",):
        return BinaryAudioMLP(n_classes=n_classes)
    raise ValueError(name)
