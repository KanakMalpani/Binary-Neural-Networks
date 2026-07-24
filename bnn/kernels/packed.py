"""Bit-packed XNOR + popcount kernels for real CPU inference speedups.

Encoding: bit 0 => +1, bit 1 => -1
Dot: <a,b> = N - 2 * popcount(a XOR b)

Includes:
  - NumPy reference (correct, may lose to BLAS without SIMD popcnt)
  - Optional native C kernel (OpenMP + hardware popcnt when compiled)
  - Thread control via BNN_NUM_THREADS / set_num_threads()
"""

from __future__ import annotations

import ctypes
import math
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .popcount import bitwise_count

_NATIVE = None
_NATIVE_PATH = Path(__file__).with_name("_binary_gemm_native")
_THREADS_APPLIED: int | None = None

# Keep embedded source in sync with binary_gemm.c for non-Windows auto-build.
_C_SOURCE = (Path(__file__).with_name("binary_gemm.c")).read_text(encoding="utf-8") if (
    Path(__file__).with_name("binary_gemm.c").exists()
) else ""


def _env_num_threads() -> int | None:
    raw = os.environ.get("BNN_NUM_THREADS") or os.environ.get("OMP_NUM_THREADS")
    if raw is None or raw.strip() == "":
        return None
    try:
        n = int(raw)
    except ValueError:
        return None
    return n if n > 0 else None


def set_num_threads(n: int | None) -> None:
    """Set native OpenMP thread count (None / 0 = library default).

    Also honors process env when first applied via ensure_native_threads().
    """
    global _THREADS_APPLIED
    lib = _try_load_native()
    if not lib:
        _THREADS_APPLIED = n if n and n > 0 else None
        return
    val = int(n) if n and n > 0 else 0
    if hasattr(lib, "binary_gemm_set_num_threads"):
        lib.binary_gemm_set_num_threads(val)
    _THREADS_APPLIED = val if val > 0 else None


def get_num_threads() -> int:
    """Effective native thread count (1 if no OpenMP / no DLL)."""
    lib = _try_load_native()
    if lib and hasattr(lib, "binary_gemm_get_num_threads"):
        return int(lib.binary_gemm_get_num_threads())
    return 1


def openmp_enabled() -> bool:
    lib = _try_load_native()
    if lib and hasattr(lib, "binary_gemm_openmp_enabled"):
        return bool(lib.binary_gemm_openmp_enabled())
    return False


def ensure_native_threads() -> None:
    """Apply BNN_NUM_THREADS / OMP_NUM_THREADS once after DLL load."""
    global _THREADS_APPLIED
    if _THREADS_APPLIED is not None:
        return
    env_n = _env_num_threads()
    if env_n is not None:
        set_num_threads(env_n)
    else:
        _THREADS_APPLIED = 0  # mark applied; leave OpenMP default


def _bind_native(lib: ctypes.CDLL) -> ctypes.CDLL:
    lib.binary_gemm_u64.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.binary_gemm_u64.restype = None
    if hasattr(lib, "binary_gemm_set_num_threads"):
        lib.binary_gemm_set_num_threads.argtypes = [ctypes.c_int]
        lib.binary_gemm_set_num_threads.restype = None
    if hasattr(lib, "binary_gemm_get_num_threads"):
        lib.binary_gemm_get_num_threads.argtypes = []
        lib.binary_gemm_get_num_threads.restype = ctypes.c_int
    if hasattr(lib, "binary_gemm_openmp_enabled"):
        lib.binary_gemm_openmp_enabled.argtypes = []
        lib.binary_gemm_openmp_enabled.restype = ctypes.c_int
    if hasattr(lib, "ternary_gemm_u64"):
        lib.ternary_gemm_u64.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_float,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.ternary_gemm_u64.restype = None
    return lib


def _try_load_native():
    global _NATIVE
    if _NATIVE is not False and _NATIVE is not None:
        return _NATIVE

    ext = ".dll" if os.name == "nt" else ".so"
    dll_path = Path(str(_NATIVE_PATH) + ext)
    if dll_path.exists():
        try:
            lib = _bind_native(ctypes.CDLL(str(dll_path)))
            _ = lib.binary_gemm_u64
            _NATIVE = lib
            ensure_native_threads()
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

    # Non-Windows: attempt compile from binary_gemm.c / embedded source
    if os.name == "nt":
        _NATIVE = False
        return False

    src_text = _C_SOURCE
    src_file = Path(__file__).with_name("binary_gemm.c")
    if src_file.exists():
        src = src_file
    else:
        if not src_text:
            _NATIVE = False
            return False
        src = Path(tempfile.gettempdir()) / "bnn_binary_gemm.c"
        src.write_text(src_text, encoding="utf-8")

    for cmd in (
        ["gcc", "-O3", "-fopenmp", "-shared", "-fPIC", "-o", str(dll_path), str(src)],
        ["gcc", "-O3", "-shared", "-fPIC", "-o", str(dll_path), str(src)],
        ["clang", "-O3", "-fopenmp", "-shared", "-fPIC", "-o", str(dll_path), str(src)],
        ["clang", "-O3", "-shared", "-fPIC", "-o", str(dll_path), str(src)],
    ):
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=120)
            if not dll_path.exists():
                continue
            try:
                lib = _bind_native(ctypes.CDLL(str(dll_path)))
                _ = lib.binary_gemm_u64
                _NATIVE = lib
                ensure_native_threads()
                return lib
            except OSError:
                # OpenMP .so may compile but fail to load (missing libgomp) —
                # delete and try next (usually no-OpenMP) command.
                try:
                    dll_path.unlink()
                except OSError:
                    pass
                continue
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            continue

    _NATIVE = False
    return False


def pack_binary_pm1(x: np.ndarray, axis: int = -1) -> tuple[np.ndarray, int]:
    """Pack ±1 values along `axis` into uint64 words. bit1 => -1/non-positive.

    Uses NumPy ``packbits`` (little bit-order) — much faster than per-bit multiply-sum.
    """
    x = np.ascontiguousarray(np.asarray(x))
    if x.size == 0:
        raise ValueError("pack_binary_pm1: empty array")
    if not np.issubdtype(x.dtype, np.floating) and not np.issubdtype(x.dtype, np.integer):
        raise TypeError(f"pack_binary_pm1: expected numeric dtype, got {x.dtype}")
    x = np.moveaxis(x, axis, -1)
    shape = x.shape
    n = int(shape[-1])
    bits = np.less_equal(x, 0)
    pad = (-n) % 64
    if pad:
        bits = np.pad(bits, [(0, 0)] * (bits.ndim - 1) + [(0, pad)], constant_values=False)
    # (..., words, 64) → packbits → (..., words, 8) uint8 → view uint64
    packed_shape = bits.shape[:-1] + (bits.shape[-1] // 64, 64)
    bits64 = bits.reshape(packed_shape)
    u8 = np.packbits(bits64.astype(np.uint8, copy=False), axis=-1, bitorder="little")
    # Force little-endian uint64 so bit j of the word matches bits[..., j]
    u8 = np.ascontiguousarray(u8)
    packed = u8.view("<u8").reshape(*shape[:-1], -1).astype(np.uint64, copy=False)
    return np.ascontiguousarray(packed, dtype=np.uint64), n


def _validate_prepacked(xp: np.ndarray, wp: np.ndarray, n: int) -> tuple[int, int, int]:
    if xp.dtype != np.uint64 or wp.dtype != np.uint64:
        raise TypeError(
            f"prepacked matrices must be uint64, got xp={xp.dtype} wp={wp.dtype}"
        )
    if xp.ndim != 2 or wp.ndim != 2:
        raise ValueError(f"expected 2D packed mats, got xp.ndim={xp.ndim} wp.ndim={wp.ndim}")
    if xp.shape[1] != wp.shape[1]:
        raise ValueError(
            f"packed word mismatch: xp words={xp.shape[1]} wp words={wp.shape[1]}"
        )
    words = xp.shape[1]
    expected = (n + 63) // 64
    if words != expected:
        raise ValueError(f"n={n} implies {expected} words, got {words}")
    return int(xp.shape[0]), int(wp.shape[0]), words


def binary_gemm_numpy_prepacked(
    xp: np.ndarray, wp: np.ndarray, n: int
) -> np.ndarray:
    """Y = binary GEMM from pre-packed uint64 matrices (NumPy path)."""
    B, M, _words = _validate_prepacked(xp, wp, n)
    out = np.empty((B, M), dtype=np.float32)
    # Row-at-a-time to keep temporaries small and cache-friendly
    for b in range(B):
        xor = np.bitwise_xor(xp[b : b + 1], wp)  # (M, words) via broadcast
        dist = bitwise_count(xor).sum(axis=1).astype(np.int32)
        out[b] = n - 2 * dist
    return out


def binary_gemm_native_prepacked(
    xp: np.ndarray, wp: np.ndarray, n: int
) -> np.ndarray | None:
    lib = _try_load_native()
    if not lib:
        return None
    ensure_native_threads()
    B, M, words = _validate_prepacked(xp, wp, n)
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
    if x_pm1.ndim != 2:
        raise ValueError(f"x_pm1 must be 2D, got shape {x_pm1.shape}")
    xp, n = pack_binary_pm1(x_pm1, axis=1)
    if prepacked_w is None:
        w_pm1 = np.asarray(w_pm1)
        if w_pm1.ndim != 2:
            raise ValueError(f"w_pm1 must be 2D, got shape {w_pm1.shape}")
        if w_pm1.shape[1] != x_pm1.shape[1]:
            raise ValueError(
                f"in_features mismatch: x {x_pm1.shape[1]} vs w {w_pm1.shape[1]}"
            )
        wp, n2 = pack_binary_pm1(w_pm1, axis=1)
        if n != n2:
            raise ValueError(f"packed n mismatch: {n} vs {n2}")
    else:
        wp, n2 = prepacked_w
        if n != n2:
            raise ValueError(f"packed n mismatch: {n} vs {n2}")

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


def ternary_native_available() -> bool:
    lib = _try_load_native()
    return bool(lib and hasattr(lib, "ternary_gemm_u64"))
