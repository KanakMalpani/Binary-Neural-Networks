"""Refuse known-bad wrap shapes with a clear message (W3.T10)."""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

from .policy import MIN_WIDTH_BINARY_EFFICIENT

# Hard refuse: pack/accuracy collapse territory (not merely "suboptimal vs 512").
HARD_REFUSE_IN = 32
HARD_REFUSE_OUT = 8


@dataclass(frozen=True)
class GuardrailVerdict:
    ok: bool
    code: str
    message: str


def check_linear_wrap_guardrails(
    lin: nn.Linear,
    *,
    mode: str = "binary_xnor",
    force: bool = False,
) -> GuardrailVerdict:
    """Return whether wrapping this Linear is advisable.

    Hard-refuse binary XNOR on pathologically narrow dims. Widths below
    ``MIN_WIDTH_BINARY_EFFICIENT`` remain allowed when the caller set a low
    ``min_in_features`` (demos / pedagogy) — efficiency tip only.
    """
    inn, out = int(lin.in_features), int(lin.out_features)
    if mode != "binary_xnor":
        return GuardrailVerdict(True, "OK", "non-binary mode")

    if inn < HARD_REFUSE_IN or out < HARD_REFUSE_OUT:
        msg = (
            f"Refuse binary_xnor on pathological Linear ({inn}×{out}): "
            f"use ternary_weight_only / skip / INT8. "
            f"(Efficient binary usually wants ≥{MIN_WIDTH_BINARY_EFFICIENT}.) "
            f"Pass force_narrow/force=True to override."
        )
        if force:
            return GuardrailVerdict(True, "FORCED_NARROW", msg)
        return GuardrailVerdict(False, "NARROW_BINARY", msg)

    if inn < MIN_WIDTH_BINARY_EFFICIENT or out < MIN_WIDTH_BINARY_EFFICIENT:
        return GuardrailVerdict(
            True,
            "SUBOPTIMAL_WIDTH",
            (
                f"binary_xnor on {inn}×{out} may lose to FP/INT8 on wall-clock; "
                f"efficient regime is typically ≥{MIN_WIDTH_BINARY_EFFICIENT}."
            ),
        )
    return GuardrailVerdict(True, "OK", "shape acceptable")
