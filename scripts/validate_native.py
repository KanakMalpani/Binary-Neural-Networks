"""Quick native kernel validation + timing."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bnn.kernels.packed import (
    native_kernel_available,
    pack_binary_pm1,
    binary_gemm_native_prepacked,
    binary_gemm_numpy_prepacked,
    fp32_gemm,
)


def main() -> None:
    print("native_available:", native_kernel_available())
    rng = np.random.default_rng(0)
    for B, N, M in [(128, 2048, 2048), (64, 4096, 4096), (32, 8192, 8192)]:
        x = rng.choice([-1.0, 1.0], size=(B, N)).astype(np.float32)
        w = rng.choice([-1.0, 1.0], size=(M, N)).astype(np.float32)
        xp, n = pack_binary_pm1(x, 1)
        wp, _ = pack_binary_pm1(w, 1)

        y_fp = fp32_gemm(x, w)
        y_np = binary_gemm_numpy_prepacked(xp, wp, n)
        y_nat = binary_gemm_native_prepacked(xp, wp, n)
        assert y_nat is not None
        err_np = float(np.max(np.abs(y_fp - y_np)))
        err_nat = float(np.max(np.abs(y_fp - y_nat)))

        def bench(fn, reps=8):
            for _ in range(3):
                fn()
            t0 = time.perf_counter()
            for _ in range(reps):
                fn()
            return (time.perf_counter() - t0) / reps

        t_fp = bench(lambda: fp32_gemm(x, w))
        t_np = bench(lambda: binary_gemm_numpy_prepacked(xp, wp, n), reps=3)
        t_nat = bench(lambda: binary_gemm_native_prepacked(xp, wp, n))
        print(
            f"B={B} N={N} M={M} | fp32={t_fp*1e3:.2f}ms numpy={t_np*1e3:.2f}ms "
            f"native={t_nat*1e3:.2f}ms | speedup_vs_fp32={t_fp/t_nat:.2f}x | "
            f"err_nat={err_nat} err_np={err_np}"
        )


if __name__ == "__main__":
    main()
