"""Straight-Through Estimators for sign / ternary quantization.

Forward is always hard ``sign`` (packing-friendly ±1).  Backward uses one of:

- **STE** (BNN / Courbariaux): identity clipped to ``|x| ≤ 1``
- **ApproxSign** (Bi-Real Net, Liu et al. 2018): piecewise-linear tent on [-1,1]
- **TanhSoft / IR-Net EDE** (Qin et al. 2020): ``g'(x) = k t (1 - tanh²(t x))``

References
----------
- Bi-Real Net: arXiv:1808.00278 (ApproxSign polynomial)
- IR-Net: arXiv:1909.10788 / CVPR 2020 (Error Decay Estimator)
"""

from __future__ import annotations

import math
from typing import Callable, Literal

import numpy as np
import torch
from torch import Tensor

SignMode = Literal["ste", "approx", "tanh_soft"]


class SignSTE(torch.autograd.Function):
    """Forward: sign to {-1, +1}. Backward: identity clipped to |x| <= 1."""

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        ctx.save_for_backward(x)
        # Map non-positive -> -1, positive -> +1 (no zeros — packing-friendly)
        return torch.where(x > 0, torch.ones_like(x), -torch.ones_like(x))

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (x,) = ctx.saved_tensors
        return grad_output * (x.abs() <= 1).to(grad_output.dtype)


def binary_sign(x: Tensor) -> Tensor:
    """Binarize to {-1, +1} with clipped STE."""
    return SignSTE.apply(x)


class ApproxSignSTE(torch.autograd.Function):
    """Bi-Real ApproxSign: forward sign; backward piecewise-linear soft gradient.

    ApproxSign derivative (Liu et al. 2018)::

        2 + 2x   for x in [-1, 0)
        2 - 2x   for x in [0, 1]
        0        otherwise
    """

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        ctx.save_for_backward(x)
        return torch.where(x > 0, torch.ones_like(x), -torch.ones_like(x))

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (x,) = ctx.saved_tensors
        grad = torch.zeros_like(x)
        mask1 = (x >= -1) & (x < 0)
        mask2 = (x >= 0) & (x <= 1)
        grad = torch.where(mask1, 2 + 2 * x, grad)
        grad = torch.where(mask2, 2 - 2 * x, grad)
        return grad_output * grad


def binary_sign_approx(x: Tensor) -> Tensor:
    """Binarize with ApproxSign STE (better deep BNN grads)."""
    return ApproxSignSTE.apply(x)


class TanhSoftSTE(torch.autograd.Function):
    """IR-Net-style Error Decay Estimator (fixed or scheduled ``t``, ``k``).

    Forward: hard sign.  Backward: ``∂g/∂x = k t (1 - tanh²(t x))`` where
    ``g(x) = k tanh(t x)`` approximates sign (Qin et al., IR-Net).

    Larger ``t`` → steeper (closer to sign, weaker early updates).
    ``k`` scales amplitude (IR-Net schedule decays toward Clip-like shape).
    """

    @staticmethod
    def forward(ctx, x: Tensor, t: Tensor, k: Tensor) -> Tensor:
        ctx.save_for_backward(x, t, k)
        return torch.where(x > 0, torch.ones_like(x), -torch.ones_like(x))

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        x, t, k = ctx.saved_tensors
        u = t * x
        soft = k * t * (1.0 - torch.tanh(u).square())
        return grad_output * soft, None, None


def binary_sign_tanh_soft(
    x: Tensor,
    *,
    t: float = 1.0,
    k: float = 1.0,
) -> Tensor:
    """Binarize with IR-Net EDE / tanh-soft gradient (fixed t, k)."""
    tt = torch.tensor(t, dtype=x.dtype, device=x.device)
    kk = torch.tensor(k, dtype=x.dtype, device=x.device)
    return TanhSoftSTE.apply(x, tt, kk)


def irnet_ede_schedule(
    epoch: int,
    n_epochs: int,
    *,
    t_min: float = 1e-1,
    t_max: float = 1e1,
) -> tuple[float, float]:
    """IR-Net two-stage (t, k) schedule (Qin et al. Eq. 18, simplified).

    Stage 1 (first half): grow ``t`` from ``t_min`` toward mid while ``k``
    keeps ``k t ≈ 1`` near 0.  Stage 2: push ``t`` toward ``t_max`` so the
    estimator approaches a staircase (sign).

    Returns ``(t, k)`` for ``binary_sign_tanh_soft``.
    """
    if n_epochs <= 0:
        raise ValueError("n_epochs must be > 0")
    i = max(0, min(epoch, n_epochs))
    progress = i / float(n_epochs)
    log_t = math.log(t_min) + progress * (math.log(t_max) - math.log(t_min))
    t = math.exp(log_t)
    if progress < 0.5:
        k = 1.0 / max(t, 1e-8)
    else:
        k = 1.0
    return float(t), float(k)


# Runtime switch used by layers via get_binary_sign_fn()
_SIGN_MODE: SignMode = "ste"
_TANH_T: float = 1.0
_TANH_K: float = 1.0


def set_approx_sign(enabled: bool) -> None:
    """Backward-compatible: ``True`` → ApproxSign, ``False`` → clipped STE."""
    global _SIGN_MODE
    _SIGN_MODE = "approx" if enabled else "ste"


def set_sign_mode(
    mode: SignMode,
    *,
    t: float = 1.0,
    k: float = 1.0,
) -> None:
    """Select STE / ApproxSign / tanh-soft for ``get_binary_sign_fn()``."""
    global _SIGN_MODE, _TANH_T, _TANH_K
    if mode not in ("ste", "approx", "tanh_soft"):
        raise ValueError(f"unknown sign mode: {mode}")
    _SIGN_MODE = mode
    _TANH_T = float(t)
    _TANH_K = float(k)


def get_sign_mode() -> SignMode:
    return _SIGN_MODE


def get_binary_sign_fn() -> Callable[[Tensor], Tensor]:
    if _SIGN_MODE == "approx":
        return binary_sign_approx
    if _SIGN_MODE == "tanh_soft":
        t, k = _TANH_T, _TANH_K

        def _fn(x: Tensor) -> Tensor:
            return binary_sign_tanh_soft(x, t=t, k=k)

        return _fn
    return binary_sign


def ste_grad_numpy(x: np.ndarray) -> np.ndarray:
    """Clipped-STE derivative mask."""
    x = np.asarray(x, dtype=np.float64)
    return (np.abs(x) <= 1).astype(np.float64)


def approx_sign_grad_numpy(x: np.ndarray) -> np.ndarray:
    """NumPy ApproxSign derivative (for math docs / cosine experiments)."""
    x = np.asarray(x, dtype=np.float64)
    g = np.zeros_like(x)
    m1 = (x >= -1) & (x < 0)
    m2 = (x >= 0) & (x <= 1)
    g = np.where(m1, 2 + 2 * x, g)
    g = np.where(m2, 2 - 2 * x, g)
    return g


def tanh_soft_grad_numpy(
    x: np.ndarray, t: float = 1.0, k: float = 1.0
) -> np.ndarray:
    """IR-Net EDE derivative ``k t (1 - tanh²(t x))``."""
    x = np.asarray(x, dtype=np.float64)
    u = t * x
    return k * t * (1.0 - np.tanh(u) ** 2)


def gradient_cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """Cosine similarity between two flat gradient fields."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class TernarySTE(torch.autograd.Function):
    """Absmean ternary quantization to {-1, 0, +1} (BitNet b1.58 style)."""

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        ctx.save_for_backward(x)
        gamma = x.abs().mean().clamp(min=1e-8)
        return (x / gamma).round().clamp(-1, 1)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (x,) = ctx.saved_tensors
        return grad_output * (x.abs() <= 1).to(grad_output.dtype)


def ternary_weight(w: Tensor) -> Tensor:
    return TernarySTE.apply(w)


def clip_weights_(module: torch.nn.Module, max_val: float = 1.0) -> None:
    """Clip latent binary/ternary weights after optimizer step."""
    with torch.no_grad():
        for _, p in module.named_parameters():
            if getattr(p, "_bnn_clip", False):
                p.clamp_(-max_val, max_val)
