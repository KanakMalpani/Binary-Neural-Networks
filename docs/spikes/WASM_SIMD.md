# Spike note — WASM SIMD popcount pedagogy (W2.T06 / M1)

| Field | Value |
|-------|-------|
| **Status** | **DELIVERED** (pedagogy) — `wasm/` + `bnn/kernels/wasm/` + parity tests |
| **Date** | 2026-08-04 |
| **Blocker** | ~~None for pedagogy~~ — optional compile needs emcc / clang wasm32 / Rust `wasm32-unknown-unknown` |
| **Acceptance** | Met: SIMD128 + scalar pedagogy sources, Node/browser demo, NumPy/C-math parity (err=0), honest non-goals vs native CPU kernels |

## Intent

Ship a **browser / Node** teaching path for packed XNOR–popcount GEMM so the
bit encoding (`bit0=+1`, `bit1=-1`, `N - 2·popcount(a⊕b)`) is visible outside
the Windows/Linux native DLL story. This is moonshot **M1** / **W2.T06**, not a
`bnn repro` gate.

## Delivered

1. **`wasm/binary_gemm_wasm.c`** — scalar + `#ifdef __wasm_simd128__` (`i8x16.popcnt`) pedagogy kernel (read-only alignment with `bnn/kernels/binary_gemm.c`).
2. **`wasm/rust/`** — optional `cdylib` using `core::arch::wasm32` (`u8x16_popcnt`).
3. **`wasm/js/`** — always-on JS scalar GEMM + `demo_node.mjs` / `demo.html`.
4. **`wasm/build.py`** — best-effort emcc → clang → Rust; skip-friendly when no toolchain.
5. **`bnn/kernels/wasm/`** — Python pedagogy path for pytest parity (err=0 vs FP32 / NumPy packed).
6. **`tests/test_wasm_parity.py`** — always-on Python tests; Node tests when `node` is available.

## Explicit non-goals / honesty

| Claim | Status |
|-------|--------|
| Replace native AVX-512 / AVX2 / NEON + OpenMP kernels | **No** — see [`../41_PORTABLE_SIMD_KERNEL.md`](../41_PORTABLE_SIMD_KERNEL.md) |
| Wall-clock speedup vs host `binary_gemm.c` | **Not claimed** — pedagogy only |
| GPU / `sign()` 32× | **Forbidden** (thesis lock) |
| Required for `bnn repro` / PyPI wheel | **No** — optional moonshot |

Production inference remains the **native packed CPU** path. WASM is for
edge-browser demos and teaching the popcount identity.

## How to run

```bash
node wasm/js/demo_node.mjs
python wasm/build.py --rust   # optional artifact → wasm/dist/binary_gemm_wasm.wasm
pytest -q tests/test_wasm_parity.py
```

## Integrator note

Lane progress: [`../lanes/f.md`](../lanes/f.md). Flip ROADMAP `W2.T06` / moonshot
`M1` / capability row **WASM** when merging `lane/f-wasm`.
