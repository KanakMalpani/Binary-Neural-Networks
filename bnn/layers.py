"""Binary / ternary layers (simulation mode — correct for training)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .ste import get_binary_sign_fn, ternary_weight


class BinaryLinear(nn.Module):
    """Dense layer with binary weights & binary activations (+ channel scale).

    Forward (sim): y = (alpha * sign(W)) @ sign(x)^T ... via F.linear
    Does NOT accelerate; use packed kernels for inference speed.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.weight._bnn_clip = True  # type: ignore[attr-defined]
        self.alpha = nn.Parameter(torch.ones(out_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        with torch.no_grad():
            self.weight.clamp_(-1, 1)
            self.alpha.fill_(self.weight.abs().mean().clamp(min=1e-4).item())

    def forward(self, x: Tensor) -> Tensor:
        sign = get_binary_sign_fn()
        x_b = sign(x)
        w_b = sign(self.weight)
        y = F.linear(x_b, w_b, None)
        y = y * self.alpha
        if self.bias is not None:
            y = y + self.bias
        return y


class TernaryLinear(nn.Module):
    """BitNet-style ternary weights, full-precision activations (training sim)."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.weight._bnn_clip = True  # type: ignore[attr-defined]
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: Tensor) -> Tensor:
        w_t = ternary_weight(self.weight)
        scale = self.weight.abs().mean().clamp(min=1e-8)
        return F.linear(x, w_t * scale, self.bias)


class BinaryConv2d(nn.Module):
    """3x3 binary conv with binary activations and per-out-channel scale."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.weight._bnn_clip = True  # type: ignore[attr-defined]
        self.alpha = nn.Parameter(torch.ones(out_channels, 1, 1))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)
        nn.init.xavier_uniform_(self.weight)
        with torch.no_grad():
            self.weight.clamp_(-1, 1)

    def forward(self, x: Tensor) -> Tensor:
        sign = get_binary_sign_fn()
        x_b = sign(x)
        w_b = sign(self.weight)
        y = F.conv2d(x_b, w_b, None, stride=self.stride, padding=self.padding)
        y = y * self.alpha
        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1)
        return y


class BiRealBlock(nn.Module):
    """Binary conv + BN + FP residual (Bi-Real Net idea)."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = BinaryConv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels, momentum=0.9)

    def forward(self, x: Tensor) -> Tensor:
        # x is full-precision residual stream
        out = self.bn(self.conv(x))
        return x + out
