"""Profile pack / GEMM / overhead for packed binary Linear path."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
import torch.nn as nn

from .kernels.packed import (
    binary_gemm_native_prepacked,
    binary_gemm_numpy_prepacked,
    native_kernel_available,
)
from .wrap.packed_linear import PackedBinaryXNORLinear, _pack_activations_fast

# Soft CI ceilings (ms) for the *smoke* profile shape used in tests / eval-suite.
# These are deliberately loose so machine variance cannot fail goldens; they only
# catch catastrophic regressions (W13.T03). Published benches stay in results/*.json.
SOFT_BUDGETS_MS: dict[tuple[int, int, int], dict[str, float]] = {
    # (m, n, k) — matches tests/test_profile.py smoke dims
    (8, 256, 256): {
        "gemm_ms": 25.0,
        "e2e_forward_ms": 40.0,
        "torch_fp32_ms": 40.0,
    },
    (64, 512, 512): {
        "gemm_ms": 80.0,
        "e2e_forward_ms": 120.0,
        "torch_fp32_ms": 120.0,
    },
}

# Soft speedup floors vs committed benchmark.json (compute-only vs NumPy FP32).
# Fail soft check only if a shape drops below half the historical headline —
# never invent new golden shapes (W7.T02 / W13.T03).
SOFT_SPEEDUP_FLOOR_FRACTION = 0.35


@dataclass
class ProfileBreakdown:
    m: int
    n: int
    k: int
    reps: int
    pack_weight_ms: float
    pack_act_ms: float
    gemm_ms: float
    scale_bias_ms: float
    e2e_forward_ms: float
    torch_fp32_ms: float
    native: bool
    overhead_vs_gemm: float  # (e2e - gemm) / gemm
    speedup_vs_fp32: float
    torch_int8_wo_ms: float = 0.0
    speedup_vs_int8_wo: float = 0.0
    baselines: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _time_int8_weight_only(
    x: torch.Tensor,
    lin: nn.Linear,
    *,
    reps: int,
    warmup: int,
) -> float:
    """Wall-clock for int8 weight-only dequant GEMM (CPU PTQ reference).

    Activations stay FP32; weights are rounded to int8 then dequantised. This is
    an honest INT8 *weight* baseline, not a claim of VNNI / AMX kernels.
    """
    w = lin.weight.detach().float()
    scale = float(w.abs().max().clamp(min=1e-8).item() / 127.0)
    w_i8 = torch.round(w / scale).clamp(-127, 127).to(torch.int8)
    w_dq = w_i8.float() * scale
    bias = lin.bias.detach().float() if lin.bias is not None else None
    for _ in range(warmup):
        y = x @ w_dq.T
        if bias is not None:
            y = y + bias
    t0 = time.perf_counter()
    for _ in range(reps):
        y = x @ w_dq.T
        if bias is not None:
            y = y + bias
    return (time.perf_counter() - t0) / reps * 1e3


def profile_packed_linear(
    *,
    m: int = 64,
    n: int = 4096,
    k: int = 4096,
    reps: int = 20,
    warmup: int = 5,
    compare_baselines: bool = True,
) -> ProfileBreakdown:
    """Break down pack_weight / pack_act / gemm / scale vs torch FP32 / INT8-WO."""
    torch.manual_seed(0)
    lin = nn.Linear(n, k, bias=True)
    x = torch.randn(m, n)
    # Weight pack (once)
    for _ in range(max(warmup, 1)):
        packed_mod = PackedBinaryXNORLinear(lin.weight.data, lin.bias.data)
    # timed pack
    t0 = time.perf_counter()
    for _ in range(reps):
        PackedBinaryXNORLinear(lin.weight.data, lin.bias.data)
    pack_w_ms = (time.perf_counter() - t0) / reps * 1e3

    packed_mod = PackedBinaryXNORLinear(lin.weight.data, lin.bias.data)
    x_np = x.detach().float().cpu().numpy()
    # Act pack
    for _ in range(warmup):
        _pack_activations_fast(x_np, n)
    t0 = time.perf_counter()
    for _ in range(reps):
        xp = _pack_activations_fast(x_np, n)
    pack_a_ms = (time.perf_counter() - t0) / reps * 1e3

    xp = _pack_activations_fast(x_np, n)
    gemm_fn = (
        binary_gemm_native_prepacked
        if packed_mod.uses_native
        else binary_gemm_numpy_prepacked
    )
    for _ in range(warmup):
        y = gemm_fn(xp, packed_mod._wp_np, n)
        assert y is not None
    t0 = time.perf_counter()
    for _ in range(reps):
        y = gemm_fn(xp, packed_mod._wp_np, n)
        assert y is not None
    gemm_ms = (time.perf_counter() - t0) / reps * 1e3

    y = gemm_fn(xp, packed_mod._wp_np, n)
    assert y is not None
    for _ in range(warmup):
        yy = y * packed_mod._alpha_np
        if packed_mod._bias_np is not None:
            yy = yy + packed_mod._bias_np
    t0 = time.perf_counter()
    for _ in range(reps):
        yy = y * packed_mod._alpha_np
        if packed_mod._bias_np is not None:
            yy = yy + packed_mod._bias_np
    scale_ms = (time.perf_counter() - t0) / reps * 1e3

    # e2e forward
    for _ in range(warmup):
        packed_mod(x)
    t0 = time.perf_counter()
    for _ in range(reps):
        packed_mod(x)
    e2e_ms = (time.perf_counter() - t0) / reps * 1e3

    # FP32 baseline
    for _ in range(warmup):
        lin(x)
    t0 = time.perf_counter()
    for _ in range(reps):
        lin(x)
    fp_ms = (time.perf_counter() - t0) / reps * 1e3

    int8_ms = 0.0
    if compare_baselines:
        int8_ms = _time_int8_weight_only(x, lin, reps=reps, warmup=warmup)

    overhead = (e2e_ms - gemm_ms) / max(gemm_ms, 1e-9)
    baselines = {
        "torch_fp32_ms": float(fp_ms),
        "torch_int8_weight_only_ms": float(int8_ms),
        "packed_e2e_ms": float(e2e_ms),
        "packed_gemm_ms": float(gemm_ms),
    }
    return ProfileBreakdown(
        m=m,
        n=n,
        k=k,
        reps=reps,
        pack_weight_ms=pack_w_ms,
        pack_act_ms=pack_a_ms,
        gemm_ms=gemm_ms,
        scale_bias_ms=scale_ms,
        e2e_forward_ms=e2e_ms,
        torch_fp32_ms=fp_ms,
        native=bool(native_kernel_available() and packed_mod.uses_native),
        overhead_vs_gemm=float(overhead),
        speedup_vs_fp32=float(fp_ms / max(e2e_ms, 1e-9)),
        torch_int8_wo_ms=float(int8_ms),
        speedup_vs_int8_wo=float(int8_ms / max(e2e_ms, 1e-9)) if int8_ms > 0 else 0.0,
        baselines=baselines,
    )


def check_soft_budgets(breakdown: ProfileBreakdown | dict[str, Any]) -> list[str]:
    """Return soft-budget violations (empty ⇒ within CI ceilings). Never hard-fails goldens."""
    if isinstance(breakdown, ProfileBreakdown):
        d = breakdown.to_dict()
    else:
        d = dict(breakdown)
    key = (int(d["m"]), int(d["n"]), int(d["k"]))
    ceilings = SOFT_BUDGETS_MS.get(key)
    if ceilings is None:
        return []
    violations: list[str] = []
    for metric, ceiling in ceilings.items():
        val = float(d.get(metric, 0.0) or 0.0)
        if val > ceiling:
            violations.append(f"{metric}={val:.3f}ms exceeds soft budget {ceiling}ms @ {key}")
    return violations


def check_committed_bench_soft_floors(
    bench: dict[str, Any],
    *,
    floor_fraction: float = SOFT_SPEEDUP_FLOOR_FRACTION,
) -> list[str]:
    """Soft-check committed ``results/benchmark.json`` speedups + thread curves."""
    rows = bench.get("results") or bench.get("rows") or bench.get("benchmarks") or []
    violations: list[str] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        s = r.get("speedup_compute_vs_numpy_fp32")
        if not isinstance(s, (int, float)):
            continue
        # Identity soft gate: published rows must themselves clear a tiny floor
        # so a corrupted JSON (speedup 0) cannot silently pass CI.
        if float(s) < floor_fraction:
            sh = r.get("shape") or {}
            violations.append(
                f"shape {sh}: speedup_compute_vs_numpy_fp32={s} "
                f"below soft absolute floor {floor_fraction}"
            )
        scaling = r.get("thread_scaling")
        if scaling is not None and not isinstance(scaling, list):
            violations.append(f"shape {r.get('shape')}: thread_scaling must be a list")
        elif isinstance(scaling, list) and len(scaling) < 2:
            violations.append(
                f"shape {r.get('shape')}: thread_scaling needs ≥2 points (W13.T04)"
            )
    return violations
