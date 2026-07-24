r"""Ternary {-1,0,+1} math sketches (BitNet-style absmean).

Absmean scale \(\gamma = \mathrm{mean}|w|\) is the closed-form minimizer of
\(\mathbb{E}[(w - \gamma\,q(w))^2]\) under the ternary grid after a uniform
threshold — see BitNet b1.58 analyses.  We expose the projector used in
``bnn.ste.TernarySTE`` for identity tests and accuracy-per-bit comparisons.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def absmean_scale(w: np.ndarray, eps: float = 1e-8) -> float:
    """γ = mean|w| (BitNet absmean)."""
    w = np.asarray(w, dtype=np.float64)
    return float(np.maximum(np.mean(np.abs(w)), eps))


def ternary_quantize_pm1_0(w: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, float]:
    """q = round(w/γ).clamp(-1,1); returns (q, γ)."""
    gamma = absmean_scale(w, eps=eps)
    q = np.clip(np.round(np.asarray(w, dtype=np.float64) / gamma), -1, 1)
    return q.astype(np.float64), gamma


def ternary_accuracy_per_bit(
    *,
    binary_acc: float,
    ternary_acc: float,
    binary_bits: float = 1.0,
    ternary_bits: float = 1.585,
) -> dict[str, Any]:
    """Compare accuracy per stored bit (entropy lower bound for ternary).

    Ternary *beats* binary on effectiveness when
    ``ternary_acc / ternary_bits > binary_acc / binary_bits`` — i.e. the
    accuracy gain pays for the ~1.58× bit cost (or 2-bit practical pack).
    """
    b_eff = binary_acc / binary_bits
    t_eff = ternary_acc / ternary_bits
    t_eff_2bit = ternary_acc / 2.0
    return {
        "binary_acc_per_bit": b_eff,
        "ternary_acc_per_entropy_bit": t_eff,
        "ternary_acc_per_practical_2bit": t_eff_2bit,
        "ternary_beats_binary_entropy": t_eff > b_eff,
        "ternary_beats_binary_2bit_pack": t_eff_2bit > b_eff,
        "note": (
            "Use measured accuracies from the same task; entropy 1.585 is a "
            "lower bound — I2_S packing uses 2 bits/weight."
        ),
    }
