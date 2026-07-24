"""Profile pack / GEMM / overhead for packed binary Linear path."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn

from .kernels.packed import (
    binary_gemm_native_prepacked,
    binary_gemm_numpy_prepacked,
    native_kernel_available,
    pack_binary_pm1,
)
from .wrap.packed_linear import PackedBinaryXNORLinear, _pack_activations_fast


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

    def to_dict(self) -> dict:
        return asdict(self)


def profile_packed_linear(
    *,
    m: int = 64,
    n: int = 4096,
    k: int = 4096,
    reps: int = 20,
    warmup: int = 5,
) -> ProfileBreakdown:
    """Break down pack_weight / pack_act / gemm / scale vs torch FP32 Linear."""
    torch.manual_seed(0)
    lin = nn.Linear(n, k, bias=True)
    x = torch.randn(m, n)
    # Weight pack (once)
    t0 = time.perf_counter()
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

    overhead = (e2e_ms - gemm_ms) / max(gemm_ms, 1e-9)
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
    )
