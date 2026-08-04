#!/usr/bin/env python3
"""Post-install smoke test for a built wheel — NumPy only, no torch.

Run against an *installed* bnn to prove the shipped binary actually works on
this machine. Deliberately avoids ``import bnn`` (which pulls in torch, ~200 MB
and unavailable on some wheel targets); it locates the package with
``find_spec``, which does not execute it, then loads the shared library through
ctypes exactly as the runtime loader would.

Exit codes:
  0 — native library loaded and every ISA path returned err=0
  1 — native library present but numerically wrong (hard failure)
  2 — no native library in the installed package (wheel built without one)
"""

from __future__ import annotations

import ctypes
import importlib.util
import sys
from pathlib import Path

import numpy as np

KERNEL_NAMES = {0: "scalar", 1: "avx2", 2: "avx512", 3: "neon"}


def find_native_library() -> Path | None:
    spec = importlib.util.find_spec("bnn")
    if spec is None or not spec.submodule_search_locations:
        print("ERROR: package 'bnn' is not installed", file=sys.stderr)
        raise SystemExit(2)
    root = Path(next(iter(spec.submodule_search_locations)))
    kernels = root / "kernels"
    for pattern in ("_binary_gemm_native*.so", "_binary_gemm_native*.pyd",
                    "_binary_gemm_native*.dll", "_binary_gemm_native*.dylib"):
        hits = sorted(kernels.glob(pattern))
        if hits:
            return hits[0]
    return None


def pack_pm1(x: np.ndarray) -> tuple[np.ndarray, int]:
    """Mirror of pack_binary_pm1 — bit 1 means non-positive."""
    n = x.shape[1]
    bits = np.less_equal(x, 0)
    pad = (-n) % 64
    if pad:
        bits = np.pad(bits, ((0, 0), (0, pad)), constant_values=False)
    grouped = bits.reshape(bits.shape[0], -1, 64)
    u8 = np.packbits(grouped.astype(np.uint8), axis=-1, bitorder="little")
    packed = np.ascontiguousarray(u8).view("<u8").reshape(x.shape[0], -1)
    return np.ascontiguousarray(packed, dtype=np.uint64), n


def main() -> int:
    lib_path = find_native_library()
    if lib_path is None:
        print("NO NATIVE LIBRARY in installed package (NumPy fallback only).")
        return 2
    print(f"native library: {lib_path.name}")

    # Python 3.8+ on Windows defaults to a restricted LoadLibrary search path;
    # use winmode=0 so adjacent / system VC runtimes resolve like a normal app.
    if sys.platform == "win32":
        lib = ctypes.CDLL(str(lib_path), winmode=0)
    else:
        lib = ctypes.CDLL(str(lib_path))
    lib.binary_gemm_u64.argtypes = [
        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    lib.binary_gemm_u64.restype = None
    for fn in ("binary_gemm_kernel_id", "binary_gemm_cpu_features"):
        getattr(lib, fn).restype = ctypes.c_int
    lib.binary_gemm_set_kernel.argtypes = [ctypes.c_int]
    lib.binary_gemm_set_kernel.restype = ctypes.c_int

    auto = lib.binary_gemm_kernel_id()
    omp = bool(lib.binary_gemm_openmp_enabled())
    print(f"auto-selected : {KERNEL_NAMES.get(auto, auto)} (openmp={omp})")

    rng = np.random.default_rng(0)
    failures = 0
    for B, N, M in [(1, 4096, 64), (5, 65, 9), (8, 320, 40), (16, 1024, 128)]:
        x = rng.choice([-1.0, 1.0], size=(B, N)).astype(np.float32)
        w = rng.choice([-1.0, 1.0], size=(M, N)).astype(np.float32)
        ref = x @ w.T
        xp, n = pack_pm1(x)
        wp, _ = pack_pm1(w)
        for kid in (0, 1, 2, 3):
            if lib.binary_gemm_set_kernel(kid) != kid:
                continue  # unsupported on this CPU
            out = np.empty((B, M), dtype=np.float32)
            lib.binary_gemm_u64(
                xp.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                wp.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                B, M, xp.shape[1], n,
            )
            err = float(np.max(np.abs(ref - out)))
            if err != 0.0:
                print(
                    f"FAIL {KERNEL_NAMES[kid]} shape=({B},{N},{M}) err={err}",
                    file=sys.stderr,
                )
                failures += 1
    lib.binary_gemm_set_kernel(-1)

    if failures:
        print(f"wheel kernel check: FAIL ({failures})", file=sys.stderr)
        return 1
    print("wheel kernel check: PASS (all ISA paths err=0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
