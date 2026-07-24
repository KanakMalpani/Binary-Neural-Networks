"""Straight-Through Estimator (STE) for sign / ternary quantization."""

from __future__ import annotations

import torch
from torch import Tensor


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
    """Binarize to {-1, +1} with STE."""
    return SignSTE.apply(x)


class ApproxSignSTE(torch.autograd.Function):
    """Bi-Real ApproxSign: forward sign; backward piecewise-linear soft gradient."""

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        ctx.save_for_backward(x)
        return torch.where(x > 0, torch.ones_like(x), -torch.ones_like(x))

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (x,) = ctx.saved_tensors
        # ApproxSign: 2+2x for -1..0, 2-2x for 0..1, else 0
        grad = torch.zeros_like(x)
        mask1 = (x >= -1) & (x < 0)
        mask2 = (x >= 0) & (x <= 1)
        grad = torch.where(mask1, 2 + 2 * x, grad)
        grad = torch.where(mask2, 2 - 2 * x, grad)
        return grad_output * grad


def binary_sign_approx(x: Tensor) -> Tensor:
    """Binarize with ApproxSign STE (better deep BNN grads)."""
    return ApproxSignSTE.apply(x)


# Runtime switch used by layers when ``use_approx_sign=True``
_USE_APPROX_SIGN = False


def set_approx_sign(enabled: bool) -> None:
    global _USE_APPROX_SIGN
    _USE_APPROX_SIGN = bool(enabled)


def get_binary_sign_fn():
    return binary_sign_approx if _USE_APPROX_SIGN else binary_sign


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
