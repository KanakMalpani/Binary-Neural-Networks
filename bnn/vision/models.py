"""Vision models: CIFAR Bi-Real CNN + tiny binary ViT sketch."""

from __future__ import annotations

import torch
import torch.nn as nn

from bnn.layers import BinaryConv2d, BinaryLinear, BiRealBlock
from bnn.ste import clip_weights_


class FP32CIFARCNN(nn.Module):
    def __init__(self, channels: int = 64, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(channels, channels * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels * 2, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels * 2, channels * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels * 2, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(channels * 2, num_classes)

    def forward(self, x):
        return self.head(self.features(x).flatten(1))


class BinaryCIFARCNN(nn.Module):
    """Bi-Real-style CIFAR CNN: FP stem/head, binary blocks + FP residuals."""

    def __init__(self, channels: int = 64, num_classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels, momentum=0.9),
        )
        self.b1 = BiRealBlock(channels)
        self.b2 = BiRealBlock(channels)
        self.pool = nn.MaxPool2d(2)
        self.down = nn.Sequential(
            BinaryConv2d(channels, channels * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels * 2, momentum=0.9),
        )
        self.skip = nn.Conv2d(channels, channels * 2, 1, bias=False)
        self.b3 = BiRealBlock(channels * 2)
        self.b4 = BiRealBlock(channels * 2)
        self.pool2 = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(channels * 2, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.pool(x)
        x = self.down(x) + self.skip(x)
        x = self.b3(x)
        x = self.b4(x)
        return self.head(self.pool2(x).flatten(1))

    def clip_weights(self):
        clip_weights_(self)


class TinyBinaryViT(nn.Module):
    """Tiny ViT-ish sketch: FP patch embed + binary MLP blocks (attn stays FP Linear).

    Pedagogical — not ImageNet SOTA. Shows hybrid FFN-binary pattern on vision tokens.
    """

    def __init__(
        self,
        img_size: int = 32,
        patch: int = 8,
        dim: int = 64,
        depth: int = 2,
        num_classes: int = 10,
        binary_ffn: bool = True,
    ):
        super().__init__()
        assert img_size % patch == 0
        self.n_patches = (img_size // patch) ** 2
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch, stride=patch, bias=False)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches + 1, dim))
        blocks = []
        for _ in range(depth):
            blocks.append(_ViTBlock(dim, binary_ffn=binary_ffn))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        b = x.size(0)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # B, N, D
        cls = self.cls.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x[:, 0]))

    def clip_weights(self):
        clip_weights_(self)


class _ViTBlock(nn.Module):
    def __init__(self, dim: int, binary_ffn: bool = True):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, dim * 3)
        self.attn_proj = nn.Linear(dim, dim)
        self.n2 = nn.LayerNorm(dim)
        # Either binary or FP depending on binary_ffn; declare the union.
        self.ff1: BinaryLinear | nn.Linear
        self.ff2: BinaryLinear | nn.Linear
        if binary_ffn:
            self.ff1 = BinaryLinear(dim, dim * 2)
            self.ff2 = BinaryLinear(dim * 2, dim)
        else:
            self.ff1 = nn.Linear(dim, dim * 2)
            self.ff2 = nn.Linear(dim * 2, dim)

    def forward(self, x):
        # Single-head attention (FP)
        h = self.n1(x)
        qkv = self.attn_qkv(h).chunk(3, dim=-1)
        q, k, v = qkv
        scale = q.size(-1) ** -0.5
        attn = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
        x = x + self.attn_proj(attn @ v)
        h = self.n2(x)
        x = x + self.ff2(torch.relu(self.ff1(h)))
        return x


def build_vision_model(name: str, channels: int = 64, **kwargs) -> nn.Module:
    name = name.lower()
    if name in ("fp32_cifar", "fp32_cnn"):
        return FP32CIFARCNN(channels=channels, **{k: v for k, v in kwargs.items() if k == "num_classes"})
    if name in ("binary_cifar", "binary_bireal", "binary_cnn"):
        return BinaryCIFARCNN(channels=channels, **{k: v for k, v in kwargs.items() if k == "num_classes"})
    if name in ("tiny_vit_binary", "binary_vit"):
        return TinyBinaryViT(binary_ffn=True, **{k: kwargs[k] for k in kwargs if k in ("dim", "depth", "num_classes", "img_size", "patch")})
    if name in ("tiny_vit_fp", "fp_vit"):
        return TinyBinaryViT(binary_ffn=False, **{k: kwargs[k] for k in kwargs if k in ("dim", "depth", "num_classes", "img_size", "patch")})
    raise ValueError(name)
