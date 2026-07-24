"""Wrap policies and hardware-aware auto recommendation.

Decision tree (honest):
- GPU serving → prefer INT4/FP8 (documented; not this kernel).
- CPU + native XNOR DLL + wide FFN → ``binary_xnor`` hybrid.
- Accuracy-first / no native kernel → ``ternary_weight_only`` hybrid.
- Narrow layers → skip or INT8 dynamic (documented fallback).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import torch.nn as nn

from ..kernels.packed import native_kernel_available

WrapMode = Literal["binary_xnor", "ternary_weight_only", "binary_weight_only_dequant"]
WrapPolicy = Literal[
    "default",
    "hybrid_ffn",
    "aggressive",
    "ternary_wo",
    "auto",
    "all_large_linear",
]

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
HYBRID_FFN_ALLOW = (
    "ffn",
    "intermediate",
    "mlp",
    "fc1",
    "fc2",
    "dense_h_to_4h",
    "dense_4h_to_h",
)
DEFAULT_SKIP = (
    "embed",
    "lm_head",
    "classifier",
    "stem",
    "head",
)
# Aggressive: still protect embed / lm_head / norms; wrap attn projections if wide.
AGGRESSIVE_SKIP = (
    "embed",
    "lm_head",
    "classifier",
    "stem",
    "head",
    "norm",
    "layernorm",
    "rmsnorm",
)

# Width below which binary XNOR packing overhead usually loses to FP/INT8.
MIN_WIDTH_BINARY_EFFICIENT = 512
MIN_WIDTH_DEFAULT = 64


@dataclass(frozen=True)
class HardwareInfo:
    native_binary_gemm: bool
    has_cuda: bool = False
    prefer_gpu_int4: bool = False


@dataclass(frozen=True)
class PolicyDecision:
    policy: WrapPolicy
    mode: WrapMode
    reason: str
    min_in_features: int
    min_out_features: int
    skip_attn: bool
    fallback_note: str = ""


def detect_hardware() -> HardwareInfo:
    has_cuda = False
    try:
        import torch

        has_cuda = bool(torch.cuda.is_available())
    except ImportError:
        has_cuda = False
    except Exception as exc:  # CUDA init can fail without ImportError
        if "cuda" not in str(exc).lower() and "CUDA" not in str(exc):
            raise
        has_cuda = False
    native = native_kernel_available()
    return HardwareInfo(
        native_binary_gemm=native,
        has_cuda=has_cuda,
        prefer_gpu_int4=has_cuda,
    )


def recommend_wrap_policy(
    layer: nn.Linear | None = None,
    hw: HardwareInfo | None = None,
    *,
    accuracy_first: bool = False,
) -> PolicyDecision:
    """Recommend wrap mode/policy for a layer (or globally if layer is None)."""
    hw = hw or detect_hardware()
    if hw.prefer_gpu_int4:
        return PolicyDecision(
            policy="hybrid_ffn",
            mode="ternary_weight_only",
            reason="CUDA present — prefer torchao/AWQ INT4/FP8 for GPU; ternary_wo only as size demo",
            min_in_features=MIN_WIDTH_DEFAULT,
            min_out_features=MIN_WIDTH_DEFAULT,
            skip_attn=True,
            fallback_note="Use GGUF/AWQ/torchao for production GPU; do not claim binary 32× on GPU",
        )

    in_f = out_f = None
    if layer is not None:
        in_f, out_f = int(layer.in_features), int(layer.out_features)

    wide = True
    if in_f is not None and out_f is not None:
        wide = in_f >= MIN_WIDTH_BINARY_EFFICIENT and out_f >= MIN_WIDTH_BINARY_EFFICIENT

    if accuracy_first or not hw.native_binary_gemm:
        return PolicyDecision(
            policy="ternary_wo" if accuracy_first else "hybrid_ffn",
            mode="ternary_weight_only",
            reason=(
                "accuracy_first"
                if accuracy_first
                else "native binary GEMM missing — ternary weight-only (size; FP GEMM)"
            ),
            min_in_features=MIN_WIDTH_DEFAULT,
            min_out_features=MIN_WIDTH_DEFAULT,
            skip_attn=True,
            fallback_note="For CPU LLM speed without BitNet kernels prefer GGUF Q4_K",
        )

    if layer is not None and not wide:
        return PolicyDecision(
            policy="hybrid_ffn",
            mode="ternary_weight_only",
            reason=f"narrow layer ({in_f}×{out_f}) — binary pack overhead; prefer ternary/INT8",
            min_in_features=MIN_WIDTH_DEFAULT,
            min_out_features=MIN_WIDTH_DEFAULT,
            skip_attn=True,
            fallback_note="INT8 dynamic or keep FP for narrow Linears",
        )

    return PolicyDecision(
        policy="hybrid_ffn",
        mode="binary_xnor",
        reason="CPU + native XNOR DLL + wide FFN → binary_xnor hybrid",
        min_in_features=MIN_WIDTH_BINARY_EFFICIENT,
        min_out_features=MIN_WIDTH_DEFAULT,
        skip_attn=True,
    )


def resolve_skip_list(
    policy: WrapPolicy = "default",
    skip_name_substr: Iterable[str] | None = None,
    *,
    skip_attn: bool = True,
) -> tuple[str, ...]:
    if skip_name_substr is not None:
        return tuple(skip_name_substr)
    if policy in ("hybrid_ffn", "ternary_wo", "auto"):
        return HYBRID_FFN_SKIP
    if policy == "aggressive":
        if skip_attn:
            return AGGRESSIVE_SKIP + ("attn", "attention", "qkv", "query", "key", "value")
        return AGGRESSIVE_SKIP
    if policy == "all_large_linear":
        return ()
    return DEFAULT_SKIP


def _should_skip(name: str, skip_substr: Iterable[str]) -> bool:
    lname = name.lower()
    return any(s.lower() in lname for s in skip_substr)


def select_linears(
    model: nn.Module,
    *,
    policy: WrapPolicy = "hybrid_ffn",
    skip_name_substr: Iterable[str] | None = None,
    min_in_features: int = 64,
    min_out_features: int = 0,
    skip_attn: bool = True,
    exclude_exact: Iterable[str] | None = None,
) -> tuple[list[tuple[str, nn.Linear]], list[str]]:
    """Return (to_replace, skipped_reasons).

    ``exclude_exact`` drops modules by full dotted name (not substring).
    """
    skipped: list[str] = []
    to_replace: list[tuple[str, nn.Linear]] = []
    exclude = set(exclude_exact or ())

    if policy in ("hybrid_ffn", "ternary_wo", "auto") and skip_name_substr is None:
        for name, mod in model.named_modules():
            if not isinstance(mod, nn.Linear):
                continue
            if name in exclude:
                skipped.append(f"{name} (sensitivity/exact exclude)")
                continue
            lname = name.lower()
            if not any(a in lname for a in HYBRID_FFN_ALLOW):
                skipped.append(f"{name} (not FFN allowlist)")
                continue
            if mod.in_features < min_in_features:
                skipped.append(f"{name} (in_features<{min_in_features})")
                continue
            if mod.out_features < min_out_features:
                skipped.append(f"{name} (out_features<{min_out_features})")
                continue
            to_replace.append((name, mod))
        return to_replace, skipped

    skip = resolve_skip_list(policy, skip_name_substr, skip_attn=skip_attn)
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if name in exclude:
            skipped.append(f"{name} (sensitivity/exact exclude)")
            continue
        if _should_skip(name, skip):
            skipped.append(f"{name} (skip list)")
            continue
        if mod.in_features < min_in_features:
            skipped.append(f"{name} (in_features<{min_in_features})")
            continue
        if mod.out_features < min_out_features:
            skipped.append(f"{name} (out_features<{min_out_features})")
            continue
        to_replace.append((name, mod))
    return to_replace, skipped
