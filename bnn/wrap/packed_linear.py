"""Packed Linear / Conv inference modules — pack once, cache on module."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..kernels.packed import (
    binary_gemm_native_prepacked,
    binary_gemm_native_scaled,
    binary_gemm_numpy_or_blas,
    native_kernel_available,
    pack_binary_pm1,
)
from .calibrate import CalibConfig, calibrate_linear_scales


def absmean_ternary(w: Tensor) -> tuple[Tensor, Tensor]:
    scale = w.detach().abs().mean().clamp(min=1e-8)
    q = (w.detach() / scale).round().clamp(-1, 1).to(torch.int8)
    return q, scale


def absmean_ternary_per_channel(w: Tensor) -> tuple[Tensor, Tensor]:
    """Per-out-channel absmean ternary (more accurate PTQ)."""
    w = w.detach().float()
    scale = w.abs().mean(dim=1).clamp(min=1e-8)  # (out,)
    q = (w / scale.unsqueeze(1)).round().clamp(-1, 1).to(torch.int8)
    return q, scale


def sign_pm1(w: Tensor) -> Tensor:
    # w >= 0 -> +1, else -1 (ties to +1, matching the packed encoding once
    # zeros are removed). Single allocation; see bnn/ste.py for the rationale.
    return w.ge(0).to(w.dtype).mul_(2).sub_(1)


def _pack_activations_fast(x2: np.ndarray, n: int) -> np.ndarray:
    """Pack (B, n) float activations → uint64 words.

    Delegates to :func:`pack_binary_pm1`, which uses ``np.packbits``. The
    previous shift-multiply-sum expanded the batch to a (B, words, 64) uint64
    temporary — ~6.5x slower for identical output, and once the native GEMM got
    faster this packing dominated the whole forward pass.
    """
    xp, _ = pack_binary_pm1(x2, axis=1)
    return xp


class PackedBinaryXNORLinear(nn.Module):
    """Inference Linear: packed ±1 weights + signed activations → XNOR GEMM.

    Weights are packed **once** at construction and cached on the module.
    """

    # nn.Module.__getattr__ is typed as returning Tensor | Module, so buffers
    # must be declared for a type checker to see them as tensors.
    weight_packed_i64: Tensor
    alpha: Tensor
    bias: Tensor | None
    _wp_np: np.ndarray
    _alpha_np: np.ndarray
    _bias_np: np.ndarray | None

    def __init__(
        self,
        weight: Tensor,
        bias: Tensor | None = None,
        *,
        alpha: Tensor | None = None,
        calib: CalibConfig | None = None,
    ):
        super().__init__()
        out_f, in_f = weight.shape
        self.in_features = in_f
        self.out_features = out_f
        w_pm1 = sign_pm1(weight.detach().float().cpu()).numpy().astype(np.float32)
        packed, n = pack_binary_pm1(w_pm1, axis=1)
        assert n == in_f
        # Persist packed words in state_dict (int64 view of uint64 bits)
        wp_i64 = torch.from_numpy(np.ascontiguousarray(packed).view(np.int64).copy())
        self.register_buffer("weight_packed_i64", wp_i64)
        self._n = in_f
        self._packed_once = True
        if alpha is None:
            alpha_t = calibrate_linear_scales(weight, cfg=calib or CalibConfig(per_channel=True))
            if alpha_t.ndim == 0:
                alpha_t = alpha_t.expand(out_f)
            alpha_t = alpha_t.float().cpu()
        else:
            alpha_t = alpha.detach().float().reshape(-1).cpu()
            if alpha_t.numel() == 1:
                alpha_t = alpha_t.expand(out_f)
        self.register_buffer("alpha", alpha_t.contiguous().clone())
        if bias is not None:
            self.register_buffer("bias", bias.detach().float().cpu().contiguous().clone())
        else:
            self.bias = None
        self.uses_native = native_kernel_available()
        self._sync_numpy_views()

    def _sync_numpy_views(self) -> None:
        """Rebuild fast NumPy views after init / load_state_dict."""
        wp = self.weight_packed_i64.detach().cpu().numpy()
        self._wp_np = np.ascontiguousarray(wp.view(np.uint64))
        self._alpha_np = np.ascontiguousarray(
            self.alpha.detach().cpu().numpy(), dtype=np.float32
        )
        self._bias_np = (
            None
            if self.bias is None
            else np.ascontiguousarray(self.bias.detach().cpu().numpy(), dtype=np.float32)
        )

    def _load_from_state_dict(self, *args, **kwargs) -> None:
        super()._load_from_state_dict(*args, **kwargs)
        self._sync_numpy_views()
        self.uses_native = native_kernel_available()
        self._packed_once = True

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"native={self.uses_native}, mode=binary_xnor, packed_once={self._packed_once}"
        )

    def forward(self, x: Tensor) -> Tensor:
        orig = x.shape
        x_cpu = x.detach().to(dtype=torch.float32, device="cpu").contiguous()
        x2 = x_cpu.reshape(-1, self.in_features).numpy()
        xp = _pack_activations_fast(x2, self._n)
        y = None
        if self.uses_native:
            # Preferred: alpha/bias folded into the kernel, so the (B, M)
            # output is written once instead of re-read twice by NumPy.
            y = binary_gemm_native_scaled(
                xp, self._wp_np, self._n, self._alpha_np, self._bias_np
            )
        if y is None:
            if self.uses_native:
                y = binary_gemm_native_prepacked(xp, self._wp_np, self._n)
                assert y is not None
            else:
                y = binary_gemm_numpy_or_blas(xp, self._wp_np, self._n)
            # Unfused fallback: scale (+ bias) in-place on numpy
            y *= self._alpha_np
            if self._bias_np is not None:
                y += self._bias_np
        out = torch.from_numpy(np.ascontiguousarray(y))
        if x.device.type != "cpu":
            out = out.to(x.device)
        return out.reshape(*orig[:-1], self.out_features)

    def packed_weight_bytes(self) -> int:
        return int(self._wp_np.nbytes)

    def gemm_only(self, x_pm1: np.ndarray) -> np.ndarray:
        """Microbench: x already ±1 float (B, N); uses cached packed weights."""
        xp, _ = pack_binary_pm1(x_pm1, axis=1)
        if self.uses_native:
            y = binary_gemm_native_prepacked(xp, self._wp_np, self._n)
            assert y is not None
        else:
            y = binary_gemm_numpy_or_blas(xp, self._wp_np, self._n, x_pm1=x_pm1)
        return y * self._alpha_np


class TernaryWeightOnlyLinear(nn.Module):
    """Accurate-first weight-only ternary (FP activations, FP GEMM after dequant)."""

    # Buffers declared for the type checker: nn.Module.__getattr__ is
    # typed as Tensor | Module.
    weight_q: Tensor
    scale: Tensor
    bias: Tensor | None

    def __init__(
        self,
        weight: Tensor,
        bias: Tensor | None = None,
        *,
        per_channel: bool = True,
        calib: CalibConfig | None = None,
    ):
        super().__init__()
        out_f, in_f = weight.shape
        self.in_features = in_f
        self.out_features = out_f
        cfg = calib or CalibConfig(per_channel=per_channel)
        if cfg.per_channel:
            q, scale = absmean_ternary_per_channel(weight.float())
        else:
            q, scale = absmean_ternary(weight.float())
        self.register_buffer("weight_q", q.cpu())
        if scale.ndim == 0:
            self.register_buffer("scale", scale.cpu().reshape(()))
            self._per_channel = False
        else:
            self.register_buffer("scale", scale.cpu().contiguous())
            self._per_channel = True
        if bias is not None:
            self.register_buffer("bias", bias.detach().float().cpu().clone())
        else:
            self.bias = None

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"mode=ternary_weight_only, per_channel={self._per_channel}"
        )

    def forward(self, x: Tensor) -> Tensor:
        if self._per_channel:
            w = self.weight_q.float() * self.scale.unsqueeze(1)
        else:
            w = self.weight_q.float() * self.scale
        return F.linear(
            x.float(),
            w.to(x.device),
            None if self.bias is None else self.bias.to(x.device),
        )

    def packed_weight_bytes(self) -> int:
        # Theoretical 2-bit pack size (actual buffer is int8 — report theoretical)
        return max(self.weight_q.numel() * 2 // 8, 1)

    @property
    def compression_kind(self) -> str:
        return "theoretical_2bit"


class BinaryWeightOnlyDequantLinear(nn.Module):
    # Buffers declared for the type checker: nn.Module.__getattr__ is
    # typed as Tensor | Module.
    weight_pm1: Tensor
    alpha: Tensor
    bias: Tensor | None
    _wp_np: np.ndarray

    def __init__(self, weight: Tensor, bias: Tensor | None = None):
        super().__init__()
        out_f, in_f = weight.shape
        self.in_features = in_f
        self.out_features = out_f
        w_pm1 = sign_pm1(weight.detach().float().cpu()).numpy().astype(np.float32)
        packed, _ = pack_binary_pm1(w_pm1, axis=1)
        self._wp_np = packed
        self.register_buffer("weight_pm1", torch.from_numpy(w_pm1))
        alpha = weight.detach().abs().mean().clamp(min=1e-4)
        self.register_buffer("alpha", alpha.cpu().reshape(()))
        if bias is not None:
            self.register_buffer("bias", bias.detach().float().cpu().clone())
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight_pm1.to(x.device) * self.alpha.to(x.device)
        return F.linear(
            x.float(),
            w,
            None if self.bias is None else self.bias.to(x.device),
        )

    def packed_weight_bytes(self) -> int:
        return int(self._wp_np.nbytes)


class PackedBinaryConv2d(nn.Module):
    """Packed ±1 Conv2d weights (size win). Forward = dequant + F.conv2d.

    Thesis: this is a **size** path (uint64 pack of ±1 kernels), not an XNOR
    popcount Conv claim. Packed words live in ``weight_packed_i64`` for
    ``.bnnpack`` / state_dict round-trips (W5.T09).
    """

    # Buffers declared for the type checker: nn.Module.__getattr__ is
    # typed as Tensor | Module.
    weight_pm1: Tensor
    weight_packed_i64: Tensor
    alpha: Tensor
    bias: Tensor | None
    _wp_np: np.ndarray

    def __init__(
        self,
        weight: Tensor,
        bias: Tensor | None = None,
        *,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        alpha: Tensor | None = None,
    ):
        super().__init__()
        if groups != 1:
            raise ValueError("PackedBinaryConv2d supports groups=1 only")
        if dilation != 1:
            raise ValueError("PackedBinaryConv2d supports dilation=1 only")
        out_c, in_c, kh, kw = weight.shape
        self.in_channels = in_c
        self.out_channels = out_c
        self.kernel_size = (kh, kw)
        self.stride = int(stride)
        self.padding = int(padding)
        self.dilation = int(dilation)
        self.groups = int(groups)
        w_pm1 = sign_pm1(weight.detach().float().cpu()).numpy().astype(np.float32)
        flat = w_pm1.reshape(out_c, -1)
        packed, n = pack_binary_pm1(flat, axis=1)
        self._n = n
        wp_i64 = torch.from_numpy(np.ascontiguousarray(packed).view(np.int64).copy())
        self.register_buffer("weight_packed_i64", wp_i64)
        self.register_buffer("weight_pm1", torch.from_numpy(w_pm1))
        self._sync_numpy_views()
        if alpha is None:
            a = weight.detach().abs().mean(dim=(1, 2, 3)).clamp(min=1e-4).float().cpu()
        else:
            a = alpha.detach().float().reshape(-1).cpu()
            if a.numel() == 1:
                a = a.expand(out_c)
        self.register_buffer("alpha", a.contiguous().clone())
        if bias is not None:
            self.register_buffer("bias", bias.detach().float().cpu().clone())
        else:
            self.bias = None

    def _sync_numpy_views(self) -> None:
        wp = self.weight_packed_i64.detach().cpu().numpy()
        self._wp_np = np.ascontiguousarray(wp.view(np.uint64))

    def _load_from_state_dict(self, *args, **kwargs) -> None:
        super()._load_from_state_dict(*args, **kwargs)
        self._sync_numpy_views()

    def extra_repr(self) -> str:
        return (
            f"in={self.in_channels}, out={self.out_channels}, "
            f"k={self.kernel_size}, mode=binary_conv_packed_dequant, "
            f"packed_once=True"
        )

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight_pm1.to(x.device) * self.alpha.view(-1, 1, 1, 1).to(x.device)
        # Out-of-place ±1 activations — do not mutate caller tensors.
        x_b = x.gt(0).to(x.dtype).mul(2).sub(1)
        return F.conv2d(
            x_b,
            w,
            None if self.bias is None else self.bias.to(x.device),
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def packed_weight_bytes(self) -> int:
        return int(self._wp_np.nbytes)


# The packed replacement modules all expose packed_weight_bytes(); naming the
# union keeps that visible to type checkers, which a bare `nn.Module`
# annotation would erase.
PackedLinearLike = (
    PackedBinaryXNORLinear | TernaryWeightOnlyLinear | BinaryWeightOnlyDequantLinear
)
