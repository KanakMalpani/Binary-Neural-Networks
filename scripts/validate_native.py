"""Quick native kernel validation + timing.

Exit codes:
  0 — native available, err=0 on all shapes
  1 — numerical failure
  2 — native DLL not available (fail loudly with remediation)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bnn.kernels.packed import (  # noqa: E402
    available_kernels,
    binary_gemm_native_prepacked,
    binary_gemm_numpy_prepacked,
    cpu_features,
    fp32_gemm,
    kernel_name,
    native_kernel_available,
    openmp_enabled,
    pack_binary_pm1,
    set_kernel,
)


def _bench(fn, reps: int = 8) -> float:
    for _ in range(3):
        fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


def main() -> int:
    available = native_kernel_available()
    print("native_available:", available, flush=True)
    if not available:
        print(
            "ERROR: native popcount DLL not loaded.\n"
            "  Windows: install VS 2022 Build Tools (x64), then:\n"
            "    python -m bnn.kernels.compile_native\n"
            "  Do NOT use MinGW 32-bit (WinError 193).\n"
            "  Linux/macOS: NumPy fallback is OK for correctness — use pytest;\n"
            "  this command specifically validates the native path.",
            file=sys.stderr,
            flush=True,
        )
        return 2

    feats = cpu_features()
    paths = available_kernels()
    print(
        f"kernel={kernel_name()} openmp={openmp_enabled()} "
        f"available={','.join(paths)} "
        f"cpu={{avx2:{feats['avx2']}, avx512_vpopcntdq:{feats['avx512_vpopcntdq']}, "
        f"neon:{feats['neon']}}}",
        flush=True,
    )

    # Cross-ISA equivalence: shipping one binary to many machines is only safe
    # if every path this CPU can run agrees exactly. Checked before timing.
    if kernel_name() != "unknown" and len(paths) > 1:
        rng_x = np.random.default_rng(3)
        xs = rng_x.choice([-1.0, 1.0], size=(9, 1000)).astype(np.float32)
        ws = rng_x.choice([-1.0, 1.0], size=(129, 1000)).astype(np.float32)
        xps, ns = pack_binary_pm1(xs, 1)
        wps, _ = pack_binary_pm1(ws, 1)
        ref = fp32_gemm(xs, ws)
        try:
            for path in paths:
                if set_kernel(path) != path:
                    print(f"ERROR: could not select kernel {path}", file=sys.stderr)
                    return 1
                y = binary_gemm_native_prepacked(xps, wps, ns)
                err = float(np.max(np.abs(ref - y)))
                if err != 0.0:
                    print(f"ERROR: kernel {path} err={err} (expected 0)", file=sys.stderr)
                    return 1
                print(f"  cross-isa {path:>6}: err=0", flush=True)
        finally:
            set_kernel(None)

    rng = np.random.default_rng(0)
    for B, N, M in [(128, 2048, 2048), (64, 4096, 4096), (32, 8192, 8192)]:
        x = rng.choice([-1.0, 1.0], size=(B, N)).astype(np.float32)
        w = rng.choice([-1.0, 1.0], size=(M, N)).astype(np.float32)
        xp, n = pack_binary_pm1(x, 1)
        wp, _ = pack_binary_pm1(w, 1)

        y_fp = fp32_gemm(x, w)
        y_np = binary_gemm_numpy_prepacked(xp, wp, n)
        y_nat = binary_gemm_native_prepacked(xp, wp, n)
        if y_nat is None:
            print("ERROR: native returned None after availability check", file=sys.stderr)
            return 2
        err_np = float(np.max(np.abs(y_fp - y_np)))
        err_nat = float(np.max(np.abs(y_fp - y_nat)))
        if err_nat != 0.0 or err_np != 0.0:
            print(f"ERROR: err_nat={err_nat} err_np={err_np}", file=sys.stderr)
            return 1

        # Bind loop variables as defaults: the closures must not see a later
        # iteration's arrays if this ever becomes deferred work.
        t_fp = _bench(lambda x=x, w=w: fp32_gemm(x, w))
        t_np = _bench(
            lambda xp=xp, wp=wp, n=n: binary_gemm_numpy_prepacked(xp, wp, n), reps=3
        )
        t_nat = _bench(lambda xp=xp, wp=wp, n=n: binary_gemm_native_prepacked(xp, wp, n))
        print(
            f"B={B} N={N} M={M} | fp32={t_fp*1e3:.2f}ms numpy={t_np*1e3:.2f}ms "
            f"native={t_nat*1e3:.2f}ms | speedup_vs_fp32={t_fp/t_nat:.2f}x | "
            f"err_nat={err_nat} err_np={err_np}",
            flush=True,
        )
    print("validate_native: PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
