"""Bit-packed XNOR + popcount kernels for real CPU inference speedups.

Encoding: bit 0 => +1, bit 1 => -1
Dot: <a,b> = N - 2 * popcount(a XOR b)

Includes:
  - NumPy reference (correct, may lose to BLAS without SIMD popcnt)
  - Optional native C kernel (compiled on first use if a compiler exists)
"""

from __future__ import annotations

import ctypes
import math
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

_NATIVE = None
_NATIVE_PATH = Path(__file__).with_name("_binary_gemm_native")


_C_SOURCE = r"""
#include <stdint.h>
#include <stddef.h>

#ifdef _MSC_VER
#include <intrin.h>
static inline int pop64(uint64_t x) { return (int)__popcnt64(x); }
#else
static inline int pop64(uint64_t x) { return __builtin_popcountll(x); }
#endif

#ifdef _WIN32
__declspec(dllexport)
#endif
void binary_gemm_u64(
    const uint64_t* X, /* B x words */
    const uint64_t* W, /* M x words */
    float* Y,          /* B x M */
    int B, int M, int words, int n
) {
    for (int b = 0; b < B; ++b) {
        const uint64_t* xb = X + (size_t)b * words;
        float* yb = Y + (size_t)b * M;
        for (int m = 0; m < M; ++m) {
            const uint64_t* wm = W + (size_t)m * words;
            int dist = 0;
            for (int w = 0; w < words; ++w) {
                dist += pop64(xb[w] ^ wm[w]);
            }
            yb[m] = (float)(n - 2 * dist);
        }
    }
}
"""


def _try_load_native():
    global _NATIVE
    if _NATIVE is not False and _NATIVE is not None:
        return _NATIVE

    ext = ".dll" if os.name == "nt" else ".so"
    dll_path = Path(str(_NATIVE_PATH) + ext)
    if dll_path.exists():
        try:
            lib = ctypes.CDLL(str(dll_path))
            lib.binary_gemm_u64.argtypes = [
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
            ]
            _ = lib.binary_gemm_u64
            _NATIVE = lib
            return lib
        except OSError as e:
            # WinError 193 = 32-bit DLL on 64-bit Python (typical MinGW mistake)
            print(
                f"WARNING: failed to load {dll_path}: {e}. "
                "On Windows use MSVC x64 via `python -m bnn.kernels.compile_native` "
                "(MinGW 32-bit DLLs raise WinError 193).",
                flush=True,
            )
            _NATIVE = False
            return False

    # Non-Windows: attempt compile from embedded source
    if os.name == "nt":
        _NATIVE = False
        return False

    src = Path(tempfile.gettempdir()) / "bnn_binary_gemm.c"
    src.write_text(_C_SOURCE, encoding="utf-8")
    for cmd in (
        ["gcc", "-O3", "-shared", "-fPIC", "-o", str(dll_path), str(src)],
        ["clang", "-O3", "-shared", "-fPIC", "-o", str(dll_path), str(src)],
    ):
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=60)
            if dll_path.exists():
                _NATIVE = None
                return _try_load_native()
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            continue

    _NATIVE = False
    return False


def pack_binary_pm1(x: np.ndarray, axis: int = -1) -> tuple[np.ndarray, int]:
    """Pack ±1 values along `axis` into uint64 words. bit1 => -1/non-positive."""
    x = np.ascontiguousarray(x)
    x = np.moveaxis(x, axis, -1)
    shape = x.shape
    n = shape[-1]
    bits = (x <= 0).astype(np.uint8)
    pad = (-n) % 64
    if pad:
        bits = np.pad(bits, [(0, 0)] * (bits.ndim - 1) + [(0, pad)], constant_values=0)
    bits = bits.reshape(*shape[:-1], -1, 64)
    weights = np.uint64(1) << np.arange(64, dtype=np.uint64)
    packed = (bits.astype(np.uint64) * weights).sum(axis=-1)
    return np.ascontiguousarray(packed, dtype=np.uint64), n


def binary_gemm_numpy_prepacked(
    xp: np.ndarray, wp: np.ndarray, n: int
) -> np.ndarray:
    """Y = binary GEMM from pre-packed uint64 matrices (NumPy path)."""
    B, words = xp.shape
    M = wp.shape[0]
    out = np.empty((B, M), dtype=np.float32)
    # Row-at-a-time to keep temporaries small and cache-friendly
    for b in range(B):
        xor = np.bitwise_xor(xp[b : b + 1], wp)  # (M, words) via broadcast
        dist = np.bitwise_count(xor).sum(axis=1).astype(np.int32)
        out[b] = n - 2 * dist
    return out


def binary_gemm_native_prepacked(
    xp: np.ndarray, wp: np.ndarray, n: int
) -> np.ndarray | None:
    lib = _try_load_native()
    if not lib:
        return None
    B, words = xp.shape
    M = wp.shape[0]
    out = np.empty((B, M), dtype=np.float32)
    lib.binary_gemm_u64(
        xp.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        wp.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        B,
        M,
        words,
        n,
    )
    return out


def binary_gemm_packed(
    x_pm1: np.ndarray,
    w_pm1: np.ndarray,
    *,
    prepacked_w: tuple[np.ndarray, int] | None = None,
) -> np.ndarray:
    """Compute Y = X @ W.T for ±1 matrices using packed XNOR-popcount."""
    x_pm1 = np.asarray(x_pm1)
    assert x_pm1.ndim == 2
    xp, n = pack_binary_pm1(x_pm1, axis=1)
    if prepacked_w is None:
        w_pm1 = np.asarray(w_pm1)
        wp, n2 = pack_binary_pm1(w_pm1, axis=1)
        assert n == n2
    else:
        wp, n2 = prepacked_w
        assert n == n2

    native = binary_gemm_native_prepacked(xp, wp, n)
    if native is not None:
        return native
    return binary_gemm_numpy_prepacked(xp, wp, n)


def fp32_gemm(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return x.astype(np.float32) @ w.astype(np.float32).T


def theoretical_ops(m: int, n: int, k: int) -> dict:
    fp32_macs = m * k * n
    binary_word_ops = m * k * math.ceil(n / 64)
    return {
        "fp32_macs": fp32_macs,
        "binary_word_xnor_popcount": binary_word_ops,
        "theoretical_word_reduction": fp32_macs / max(binary_word_ops, 1),
        "weight_bytes_fp32": k * n * 4,
        "weight_bytes_binary": k * math.ceil(n / 8),
        "weight_compression": (k * n * 4) / max(k * math.ceil(n / 8), 1),
    }


def native_kernel_available() -> bool:
    return bool(_try_load_native())
