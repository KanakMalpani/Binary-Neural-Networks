"""Model zoo: FP32 baselines vs Bi-Real-style binary networks."""

from __future__ import annotations

import torch
import torch.nn as nn

from .layers import BinaryConv2d, BinaryLinear, BiRealBlock, TernaryLinear
from .ste import clip_weights_


class FP32MLP(nn.Module):
    def __init__(self, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, hidden),
            nn.BatchNorm1d(hidden, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 10),
        )

    def forward(self, x):
        return self.net(x)


class BinaryMLP(nn.Module):
    """FP stem/head, binary hidden layers (standard BNN practice)."""

    def __init__(self, hidden: int = 512):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, hidden)  # FP first
        self.bn1 = nn.BatchNorm1d(hidden, momentum=0.9)
        self.fc2 = BinaryLinear(hidden, hidden)
        self.bn2 = nn.BatchNorm1d(hidden, momentum=0.9)
        self.fc3 = BinaryLinear(hidden, hidden)
        self.bn3 = nn.BatchNorm1d(hidden, momentum=0.9)
        self.fc4 = nn.Linear(hidden, 10)  # FP last

    def forward(self, x):
        x = self.flatten(x)
        x = self.bn1(self.fc1(x))
        x = self.bn2(self.fc2(x))
        x = self.bn3(self.fc3(x))
        return self.fc4(x)

    def clip_weights(self):
        clip_weights_(self)


class TernaryMLP(nn.Module):
    """BitNet-style ternary hidden weights, FP activations."""

    def __init__(self, hidden: int = 512):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, hidden)
        self.bn1 = nn.BatchNorm1d(hidden, momentum=0.9)
        self.act1 = nn.ReLU(inplace=True)
        self.fc2 = TernaryLinear(hidden, hidden)
        self.bn2 = nn.BatchNorm1d(hidden, momentum=0.9)
        self.act2 = nn.ReLU(inplace=True)
        self.fc3 = TernaryLinear(hidden, hidden)
        self.bn3 = nn.BatchNorm1d(hidden, momentum=0.9)
        self.act3 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(hidden, 10)

    def forward(self, x):
        x = self.act1(self.bn1(self.fc1(self.flatten(x))))
        x = self.act2(self.bn2(self.fc2(x)))
        x = self.act3(self.bn3(self.fc3(x)))
        return self.fc4(x)

    def clip_weights(self):
        clip_weights_(self)


class FP32CNN(nn.Module):
    def __init__(self, channels: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(channels, channels * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels * 2, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(channels * 2, 10)

    def forward(self, x):
        x = self.features(x)
        return self.head(x.flatten(1))


class BinaryCNN(nn.Module):
    """Bi-Real-style: FP stem, binary blocks with FP residuals, FP head."""

    def __init__(self, channels: int = 32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels, momentum=0.9),
        )
        self.block1 = BiRealBlock(channels)
        self.block2 = BiRealBlock(channels)
        self.pool = nn.MaxPool2d(2)
        self.down = nn.Sequential(
            BinaryConv2d(channels, channels * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels * 2, momentum=0.9),
        )
        self.skip_down = nn.Conv2d(channels, channels * 2, 1, bias=False)
        self.block3 = BiRealBlock(channels * 2)
        self.pool2 = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(channels * 2, 10)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)
        identity = self.skip_down(x)
        x = self.down(x) + identity
        x = self.block3(x)
        x = self.pool2(x).flatten(1)
        return self.head(x)

    def clip_weights(self):
        clip_weights_(self)


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    binaryish = 0
    for name, module in model.named_modules():
        if isinstance(module, (BinaryLinear, BinaryConv2d, TernaryLinear)):
            binaryish += module.weight.numel()
    return {
        "total_params": total,
        "binary_or_ternary_weight_params": binaryish,
        "fp_params": total - binaryish,
        "theoretical_weight_bytes_if_packed": (total - binaryish) * 4
        + max(binaryish // 8, 0),
        "fp32_weight_bytes": total * 4,
    }


def build_model(name: str, hidden: int = 512, channels: int = 32) -> nn.Module:
    name = name.lower()
    table = {
        "fp32_mlp": lambda: FP32MLP(hidden),
        "binary_mlp": lambda: BinaryMLP(hidden),
        "ternary_mlp": lambda: TernaryMLP(hidden),
        "fp32_cnn": lambda: FP32CNN(channels),
        "binary_cnn": lambda: BinaryCNN(channels),
    }
    if name not in table:
        raise ValueError(f"Unknown model {name}. Choose from {list(table)}")
    return table[name]()
