"""Parity tests for pedagogy WASM / JS binary GEMM (W2.T06).

Always-on: Python pedagogy path vs FP32 and vs NumPy packed GEMM (err = 0).
Optional: Node demo when ``node`` is on PATH; optional compiled ``.wasm`` via
``wasm/build.py`` (not required for pytest green).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from bnn.kernels.packed import binary_gemm_numpy_prepacked, fp32_gemm, pack_binary_pm1
from bnn.kernels.wasm import binary_gemm_wasm_numpy, binary_gemm_wasm_prepacked, set_kernel

ROOT = Path(__file__).resolve().parents[1]
WASM_DIR = ROOT / "wasm"
DIST_WASM = WASM_DIR / "dist" / "binary_gemm_wasm.wasm"


@pytest.mark.parametrize(
    "B,N,M",
    [
        (1, 63, 7),
        (4, 64, 16),
        (5, 65, 9),
        (8, 128, 32),
        (3, 256, 17),
    ],
)
def test_wasm_python_matches_fp_and_numpy(B, N, M):
    rng = np.random.default_rng(42)
    x = rng.choice([-1.0, 1.0], size=(B, N)).astype(np.float32)
    w = rng.choice([-1.0, 1.0], size=(M, N)).astype(np.float32)
    y_fp = fp32_gemm(x, w)
    y_w = binary_gemm_wasm_numpy(x, w)
    xp, n = pack_binary_pm1(x, 1)
    wp, _ = pack_binary_pm1(w, 1)
    y_np = binary_gemm_numpy_prepacked(xp, wp, n)
    assert float(np.max(np.abs(y_fp - y_w))) == 0.0
    assert float(np.max(np.abs(y_w - y_np))) == 0.0


def test_wasm_prepacked_matches_numpy():
    rng = np.random.default_rng(7)
    B, N, M = 16, 512, 64
    x = rng.choice([-1.0, 1.0], size=(B, N)).astype(np.float32)
    w = rng.choice([-1.0, 1.0], size=(M, N)).astype(np.float32)
    xp, n = pack_binary_pm1(x, 1)
    wp, _ = pack_binary_pm1(w, 1)
    y_w = binary_gemm_wasm_prepacked(xp, wp, n)
    y_np = binary_gemm_numpy_prepacked(xp, wp, n)
    assert float(np.max(np.abs(y_w - y_np))) == 0.0


def test_wasm_kernel_label_simd128_still_correct():
    """simd128 is a pedagogy label in Python; math must stay err=0."""
    set_kernel("simd128")
    try:
        rng = np.random.default_rng(3)
        x = rng.choice([-1.0, 1.0], size=(4, 128)).astype(np.float32)
        w = rng.choice([-1.0, 1.0], size=(8, 128)).astype(np.float32)
        y = binary_gemm_wasm_numpy(x, w)
        assert float(np.max(np.abs(y - fp32_gemm(x, w)))) == 0.0
    finally:
        set_kernel("scalar")


def test_wasm_c_source_exists_and_documents_nongoals():
    src = (WASM_DIR / "binary_gemm_wasm.c").read_text(encoding="utf-8")
    assert "N - 2 * popcount" in src or "n - 2" in src
    assert "NON-GOALS" in src or "Non-goal" in src or "non-goal" in src.lower()
    assert "wasm_simd128" in src or "i8x16_popcnt" in src or "BNN_WASM_SIMD" in src


@pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")
def test_node_demo_pass():
    demo = WASM_DIR / "js" / "demo_node.mjs"
    proc = subprocess.run(
        ["node", str(demo)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "PASS" in proc.stdout


@pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")
def test_js_scalar_parity_via_node_helper():
    """Inline Node check: pack + GEMM vs values computed in Python."""
    rng = np.random.default_rng(99)
    B, N, M = 4, 64, 8
    x = rng.choice([-1.0, 1.0], size=(B, N)).astype(np.float32)
    w = rng.choice([-1.0, 1.0], size=(M, N)).astype(np.float32)
    y_py = binary_gemm_wasm_numpy(x, w)
    payload = {
        "B": B,
        "N": N,
        "M": M,
        "x": x.reshape(-1).tolist(),
        "w": w.reshape(-1).tolist(),
        "y": y_py.reshape(-1).tolist(),
    }
    script = r"""
import { readFileSync } from 'node:fs';
import { binaryGemmScalar, packPm1 } from './wasm/js/binary_gemm.mjs';
const p = JSON.parse(readFileSync(0, 'utf8'));
const x = Float32Array.from(p.x);
const w = Float32Array.from(p.w);
const { packed: xp, words } = packPm1(x, p.B, p.N);
const { packed: wp } = packPm1(w, p.M, p.N);
const y = binaryGemmScalar(xp, wp, p.B, p.M, words, p.N);
let max = 0;
for (let i = 0; i < y.length; i++) {
  const d = Math.abs(y[i] - p.y[i]);
  if (d > max) max = d;
}
if (max !== 0) {
  console.error('maxErr=' + max);
  process.exit(1);
}
console.log('OK');
"""
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=str(ROOT),
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK" in proc.stdout


def test_dist_wasm_optional_documented():
    """Spike/README document optional compile; committed artifact preferred."""
    readme = (WASM_DIR / "README.md").read_text(encoding="utf-8")
    assert "optional" in readme.lower()
    spike = ROOT / "docs" / "spikes" / "WASM_SIMD.md"
    assert spike.is_file(), "docs/spikes/WASM_SIMD.md missing"
    body = spike.read_text(encoding="utf-8")
    assert "DELIVERED" in body or "delivered" in body.lower()
    assert "native" in body.lower()
    if DIST_WASM.is_file():
        assert DIST_WASM.stat().st_size > 100


@pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")
@pytest.mark.skipif(not DIST_WASM.is_file(), reason="wasm/dist artifact not present")
def test_node_instantiates_committed_wasm():
    """Committed .wasm must load via fs path (not broken by Node global fetch)."""
    script = r"""
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadWasm, binaryGemm, packPm1, wasmKernelId } from './wasm/js/binary_gemm.mjs';

const root = process.cwd();
const wasmPath = join(root, 'wasm', 'dist', 'binary_gemm_wasm.wasm');
const inst = await loadWasm(wasmPath);
if (!inst) {
  console.error('FAIL: loadWasm returned null for', wasmPath);
  process.exit(1);
}
const kid = wasmKernelId();
const B = 4, N = 64, M = 8;
const x = new Float32Array(B * N);
const w = new Float32Array(M * N);
for (let i = 0; i < x.length; i++) x[i] = (i & 1) ? -1 : 1;
for (let i = 0; i < w.length; i++) w[i] = (i % 3) ? -1 : 1;
const { packed: xp, words } = packPm1(x, B, N);
const { packed: wp } = packPm1(w, M, N);
const { y, backend } = binaryGemm(xp, wp, B, M, words, N);
if (backend !== 'wasm') {
  console.error('FAIL: expected backend=wasm got', backend, 'kernel_id=', kid);
  process.exit(1);
}
if (y.length !== B * M) {
  console.error('FAIL: bad Y length', y.length);
  process.exit(1);
}
console.log('OK kernel_id=' + kid + ' backend=' + backend);
"""
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK" in proc.stdout
    assert "backend=wasm" in proc.stdout


def test_lane_note_has_no_absolute_windows_home_path():
    """docs/lanes/f.md must not leak C:\\Users\\<name>\\ paths."""
    text = (ROOT / "docs" / "lanes" / "f.md").read_text(encoding="utf-8")
    assert not __import__("re").search(
        r"[A-Za-z]:\\Users\\(?!<)[^\\\s<>]+\\", text
    ), "lane note still contains an absolute local path"
