"""Drop-in wrappers that replace nn.Linear for CPU packed inference.

Honest modes
------------
1. ``binary_xnor`` — packed ±1 weights + signed activations → native XNOR GEMM.
2. ``ternary_weight_only`` — absmean ternary weights, FP activations (size; FP GEMM).
3. ``binary_weight_only_dequant`` — packed store, dequant GEMM (anti-pattern for speed).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .kernels.packed import (
    binary_gemm_native_prepacked,
    binary_gemm_numpy_prepacked,
    native_kernel_available,
    pack_binary_pm1,
)

WrapMode = Literal["binary_xnor", "ternary_weight_only", "binary_weight_only_dequant"]
WrapPolicy = Literal["default", "hybrid_ffn", "all_large_linear"]

HYBRID_FFN_SKIP = (
    "embed",
    "attn",
    "attention",
    "lm_head",
    "classifier",
    "stem",
    "head",
    "norm",
    "qkv",
    "query",
    "key",
    "value",
    "pooler",
)
# When policy=hybrid_ffn, only wrap modules matching these substrings (FFN/MLP).
HYBRID_FFN_ALLOW = ("ffn", "intermediate", "mlp", "fc1", "fc2", "dense_h_to_4h", "dense_4h_to_h")

DEFAULT_SKIP = (
    "embed",
    "lm_head",
    "classifier",
    "fc1",
    "stem",
    "head",
)


def absmean_ternary(w: Tensor) -> tuple[Tensor, Tensor]:
    scale = w.detach().abs().mean().clamp(min=1e-8)
    q = (w.detach() / scale).round().clamp(-1, 1).to(torch.int8)
    return q, scale


def sign_pm1(w: Tensor) -> Tensor:
    return torch.where(w >= 0, torch.ones_like(w), -torch.ones_like(w))


class PackedBinaryXNORLinear(nn.Module):
    """Inference Linear: packed ±1 weights + signed activations → XNOR GEMM."""

    def __init__(
        self,
        weight: Tensor,
        bias: Tensor | None = None,
        *,
        alpha: Tensor | None = None,
    ):
        super().__init__()
        out_f, in_f = weight.shape
        self.in_features = in_f
        self.out_features = out_f
        w_pm1 = sign_pm1(weight.detach().float().cpu()).numpy().astype(np.float32)
        packed, n = pack_binary_pm1(w_pm1, axis=1)
        assert n == in_f
        self._wp_np = np.ascontiguousarray(packed)
        self._n = in_f
        if alpha is None:
            alpha_t = (
                weight.detach().abs().mean().clamp(min=1e-4).expand(out_f).float().cpu()
            )
        else:
            alpha_t = alpha.detach().float().reshape(-1).cpu()
            if alpha_t.numel() == 1:
                alpha_t = alpha_t.expand(out_f)
        self.register_buffer("alpha", alpha_t.contiguous().clone())
        if bias is not None:
            self.register_buffer("bias", bias.detach().float().cpu().contiguous().clone())
        else:
            self.bias = None  # type: ignore[assignment]
        self.uses_native = native_kernel_available()
        # Preallocate alpha numpy for fast scale
        self._alpha_np = self.alpha.numpy()
        self._bias_np = None if self.bias is None else self.bias.numpy()

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"native={self.uses_native}, mode=binary_xnor"
        )

    def forward(self, x: Tensor) -> Tensor:
        orig = x.shape
        # Contiguous float32 CPU view for packing
        x_cpu = x.detach().to(dtype=torch.float32, device="cpu").contiguous()
        x2 = x_cpu.reshape(-1, self.in_features)
        # Sign without full float copy of ±1: pack from threshold
        bits = (x2.numpy() <= 0).astype(np.uint8)
        n = self._n
        pad = (-n) % 64
        if pad:
            bits = np.pad(bits, ((0, 0), (0, pad)), constant_values=0)
        B = bits.shape[0]
        bits = bits.reshape(B, -1, 64)
        weights = np.uint64(1) << np.arange(64, dtype=np.uint64)
        xp = np.ascontiguousarray((bits.astype(np.uint64) * weights).sum(axis=-1))
        if self.uses_native:
            y = binary_gemm_native_prepacked(xp, self._wp_np, self._n)
            assert y is not None
        else:
            y = binary_gemm_numpy_prepacked(xp, self._wp_np, self._n)
        y *= self._alpha_np
        if self._bias_np is not None:
            y += self._bias_np
        out = torch.from_numpy(np.ascontiguousarray(y))
        if x.device.type != "cpu":
            out = out.to(x.device)
        return out.reshape(*orig[:-1], self.out_features)

    def packed_weight_bytes(self) -> int:
        return int(self._wp_np.nbytes)

    def gemm_only(
        self, x_pm1: np.ndarray
    ) -> np.ndarray:
        """Microbench helper: x already ±1 float (B, N)."""
        xp, _ = pack_binary_pm1(x_pm1, axis=1)
        if self.uses_native:
            y = binary_gemm_native_prepacked(xp, self._wp_np, self._n)
            assert y is not None
        else:
            y = binary_gemm_numpy_prepacked(xp, self._wp_np, self._n)
        return y * self._alpha_np


class TernaryWeightOnlyLinear(nn.Module):
    def __init__(self, weight: Tensor, bias: Tensor | None = None):
        super().__init__()
        out_f, in_f = weight.shape
        self.in_features = in_f
        self.out_features = out_f
        q, scale = absmean_ternary(weight.float())
        # Store only int8 ternary + scale (not a float copy — that would erase the size win)
        self.register_buffer("weight_q", q.cpu())
        self.register_buffer("scale", scale.cpu().reshape(()))
        if bias is not None:
            self.register_buffer("bias", bias.detach().float().cpu().clone())
        else:
            self.bias = None  # type: ignore[assignment]

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, mode=ternary_weight_only"

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight_q.float() * self.scale
        return F.linear(
            x.float(),
            w.to(x.device),
            None if self.bias is None else self.bias.to(x.device),
        )

    def packed_weight_bytes(self) -> int:
        return max(self.weight_q.numel() * 2 // 8, 1)


class BinaryWeightOnlyDequantLinear(nn.Module):
    def __init__(self, weight: Tensor, bias: Tensor | None = None):
        super().__init__()
        out_f, in_f = weight.shape
        self.in_features = in_f
        self.out_features = out_f
        w_pm1 = sign_pm1(weight.detach().float().cpu()).numpy().astype(np.float32)
        packed, _ = pack_binary_pm1(w_pm1, axis=1)
        self._wp_np = packed
        # Keep packed only; dequant on the fly (demonstrates size vs speed tradeoff)
        self.register_buffer("weight_pm1", torch.from_numpy(w_pm1))
        alpha = weight.detach().abs().mean().clamp(min=1e-4)
        self.register_buffer("alpha", alpha.cpu().reshape(()))
        if bias is not None:
            self.register_buffer("bias", bias.detach().float().cpu().clone())
        else:
            self.bias = None  # type: ignore[assignment]

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight_pm1.to(x.device) * self.alpha.to(x.device)
        return F.linear(
            x.float(),
            w,
            None if self.bias is None else self.bias.to(x.device),
        )

    def packed_weight_bytes(self) -> int:
        return int(self._wp_np.nbytes)


@dataclass
class WrapReport:
    mode: WrapMode
    replaced: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    fp32_weight_bytes_replaced: int = 0
    packed_weight_bytes: int = 0
    native_kernel: bool = False

    @property
    def compression(self) -> float:
        if self.packed_weight_bytes <= 0:
            return 0.0
        return self.fp32_weight_bytes_replaced / self.packed_weight_bytes


def _should_skip(name: str, skip_substr: Iterable[str]) -> bool:
    lname = name.lower()
    return any(s.lower() in lname for s in skip_substr)


def resolve_skip_list(
    policy: WrapPolicy = "default",
    skip_name_substr: Iterable[str] | None = None,
) -> tuple[str, ...]:
    if skip_name_substr is not None:
        return tuple(skip_name_substr)
    if policy == "hybrid_ffn":
        return HYBRID_FFN_SKIP
    if policy == "all_large_linear":
        return ()
    return DEFAULT_SKIP


def wrap_model(
    model: nn.Module,
    mode: WrapMode = "binary_xnor",
    *,
    policy: WrapPolicy = "default",
    skip_name_substr: Iterable[str] | None = None,
    min_in_features: int = 64,
    inplace: bool = True,
) -> tuple[nn.Module, WrapReport]:
    """Product wrap API with named policies (hybrid FFN / all-large / default)."""
    if policy == "hybrid_ffn" and skip_name_substr is None:
        # Allowlist FFN-like names; skip everything else
        report = WrapReport(mode=mode, native_kernel=native_kernel_available())
        to_replace: list[tuple[str, nn.Linear]] = []
        for name, mod in model.named_modules():
            if not isinstance(mod, nn.Linear):
                continue
            lname = name.lower()
            if not any(a in lname for a in HYBRID_FFN_ALLOW):
                report.skipped.append(f"{name} (not FFN allowlist)")
                continue
            if mod.in_features < min_in_features:
                report.skipped.append(f"{name} (in_features<{min_in_features})")
                continue
            to_replace.append((name, mod))

        def _set_module(root: nn.Module, path: str, new: nn.Module) -> None:
            parts = path.split(".")
            parent = root
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], new)

        for name, lin in to_replace:
            w = lin.weight.data
            b = lin.bias.data if lin.bias is not None else None
            fp_bytes = w.numel() * 4
            if mode == "binary_xnor":
                new: nn.Module = PackedBinaryXNORLinear(w, b)
                packed_b = new.packed_weight_bytes()
                report.native_kernel = getattr(new, "uses_native", False)
            elif mode == "ternary_weight_only":
                new = TernaryWeightOnlyLinear(w, b)
                packed_b = new.packed_weight_bytes()
            elif mode == "binary_weight_only_dequant":
                new = BinaryWeightOnlyDequantLinear(w, b)
                packed_b = new.packed_weight_bytes()
            else:
                raise ValueError(mode)
            report.replaced.append(name)
            report.fp32_weight_bytes_replaced += fp_bytes
            report.packed_weight_bytes += packed_b
            if inplace:
                _set_module(model, name, new)
        return model, report

    return wrap_linear_modules(
        model,
        mode,
        skip_name_substr=resolve_skip_list(policy, skip_name_substr),
        min_in_features=min_in_features,
        inplace=inplace,
    )


def wrap_linear_modules(
    model: nn.Module,
    mode: WrapMode = "binary_xnor",
    *,
    skip_name_substr: Iterable[str] = DEFAULT_SKIP,
    min_in_features: int = 64,
    inplace: bool = True,
) -> tuple[nn.Module, WrapReport]:
    report = WrapReport(mode=mode, native_kernel=native_kernel_available())
    to_replace: list[tuple[str, nn.Linear]] = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if _should_skip(name, skip_name_substr):
            report.skipped.append(f"{name} (skip list)")
            continue
        if mod.in_features < min_in_features:
            report.skipped.append(f"{name} (in_features<{min_in_features})")
            continue
        to_replace.append((name, mod))

    def _set_module(root: nn.Module, path: str, new: nn.Module) -> None:
        parts = path.split(".")
        parent = root
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], new)

    for name, lin in to_replace:
        w = lin.weight.data
        b = lin.bias.data if lin.bias is not None else None
        fp_bytes = w.numel() * 4
        if mode == "binary_xnor":
            new: nn.Module = PackedBinaryXNORLinear(w, b)
            packed_b = new.packed_weight_bytes()  # type: ignore[attr-defined]
            report.native_kernel = getattr(new, "uses_native", False)
        elif mode == "ternary_weight_only":
            new = TernaryWeightOnlyLinear(w, b)
            packed_b = new.packed_weight_bytes()
        elif mode == "binary_weight_only_dequant":
            new = BinaryWeightOnlyDequantLinear(w, b)
            packed_b = new.packed_weight_bytes()
        else:
            raise ValueError(mode)

        report.replaced.append(name)
        report.fp32_weight_bytes_replaced += fp_bytes
        report.packed_weight_bytes += packed_b
        if inplace:
            _set_module(model, name, new)

    return model, report


def model_param_bytes(model: nn.Module) -> dict:
    p_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    b_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return {"param_bytes": p_bytes, "buffer_bytes": b_bytes, "total_bytes": p_bytes + b_bytes}


class PackedBinaryConv2d(nn.Module):
    """Inference Conv2d with packed ±1 weights (size win).

    Forward uses dequant + ``F.conv2d`` — there is **no** native binary-conv DLL
    yet (unlike Linear XNOR GEMM). Honest: compression is real; speed is not a
    32× claim. Prefer wrapping large Linear/FFN for CPU speedups.
    """

    def __init__(
        self,
        weight: Tensor,
        bias: Tensor | None = None,
        *,
        stride: int = 1,
        padding: int = 0,
        alpha: Tensor | None = None,
    ):
        super().__init__()
        out_c, in_c, kh, kw = weight.shape
        self.in_channels = in_c
        self.out_channels = out_c
        self.kernel_size = (kh, kw)
        self.stride = stride
        self.padding = padding
        w_pm1 = sign_pm1(weight.detach().float().cpu()).numpy().astype(np.float32)
        flat = w_pm1.reshape(out_c, -1)
        packed, n = pack_binary_pm1(flat, axis=1)
        self._n = n
        self._wp_np = np.ascontiguousarray(packed)
        self.register_buffer("weight_pm1", torch.from_numpy(w_pm1))
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
            self.bias = None  # type: ignore[assignment]

    def extra_repr(self) -> str:
        return (
            f"in={self.in_channels}, out={self.out_channels}, "
            f"k={self.kernel_size}, mode=binary_conv_packed_dequant"
        )

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight_pm1.to(x.device) * self.alpha.view(-1, 1, 1, 1).to(x.device)
        # Sign activations for binary-act inference (matches BinaryConv2d sim)
        x_b = torch.where(x > 0, torch.ones_like(x), -torch.ones_like(x))
        return F.conv2d(
            x_b,
            w,
            None if self.bias is None else self.bias.to(x.device),
            stride=self.stride,
            padding=self.padding,
        )

    def packed_weight_bytes(self) -> int:
        return int(self._wp_np.nbytes)


def wrap_conv_modules(
    model: nn.Module,
    *,
    skip_name_substr: Iterable[str] = ("stem", "head", "skip"),
    min_weight_elems: int = 256,
    inplace: bool = True,
) -> tuple[nn.Module, WrapReport]:
    """Replace ``nn.Conv2d`` / ``BinaryConv2d`` with packed-weight ``PackedBinaryConv2d``.

    Compression is real (~32× on weights). Speed is **not** claimed — forward is
    dequant + FP conv. Use for size demos / export; Linear wrap for CPU speed.
    """
    from .layers import BinaryConv2d  # local import avoids cycle at module load

    report = WrapReport(mode="binary_xnor", native_kernel=False)
    to_replace: list[tuple[str, nn.Module]] = []
    for name, mod in model.named_modules():
        if isinstance(mod, BinaryConv2d):
            pass
        elif isinstance(mod, nn.Conv2d) and mod.groups == 1:
            pass
        else:
            continue
        if _should_skip(name, skip_name_substr):
            report.skipped.append(f"{name} (skip list)")
            continue
        w = mod.weight
        if w.numel() < min_weight_elems:
            report.skipped.append(f"{name} (too small)")
            continue
        to_replace.append((name, mod))

    def _set_module(root: nn.Module, path: str, new: nn.Module) -> None:
        parts = path.split(".")
        parent = root
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], new)

    for name, mod in to_replace:
        w = mod.weight.data
        b = mod.bias.data if getattr(mod, "bias", None) is not None else None
        stride = getattr(mod, "stride", 1)
        padding = getattr(mod, "padding", 0)
        if isinstance(stride, tuple):
            stride = stride[0]
        if isinstance(padding, tuple):
            padding = padding[0]
        alpha = getattr(mod, "alpha", None)
        if alpha is not None:
            alpha = alpha.detach().reshape(-1)
        new = PackedBinaryConv2d(w, b, stride=stride, padding=padding, alpha=alpha)
        report.replaced.append(name)
        report.fp32_weight_bytes_replaced += int(w.numel() * 4)
        report.packed_weight_bytes += new.packed_weight_bytes()
        if inplace:
            _set_module(model, name, new)
    return model, report
