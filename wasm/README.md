# Pedagogy WASM binary GEMM (W2.T06 / M1)

Browser / Node demo of packed **XNOR + popcount** GEMM. Same encoding as
`bnn/kernels/binary_gemm.c` (read-only reference for this lane):

| Symbol | Meaning |
|--------|---------|
| bit `0` | `+1` |
| bit `1` | `-1` |
| dot | `N - 2 * popcount(a XOR b)` |

## What this is

- Optional **WASM SIMD128** (`i8x16.popcnt`) + scalar fallback
- Node CLI demo + static HTML page
- Parity against NumPy / Python pedagogy path (`bnn.kernels.wasm`)

## What this is not

- **Not** a replacement for native CPU kernels (AVX-512 / AVX2 / NEON + OpenMP)
- **No** wall-clock speed claims vs host kernels — pedagogy only
- **Not** part of the `bnn repro` golden gate (moonshot / optional)

## Quick demo (no compiler)

```bash
# From repo root — Node 18+ (WebAssembly.SIMD preferred, scalar JS always works)
node wasm/js/demo_node.mjs

# Or open wasm/js/demo.html in a Chromium-based browser
```

## Optional compile

```bash
python wasm/build.py          # tries emcc, then clang wasm32, then Rust cdylib
python wasm/build.py --rust   # force Rust wasm32-unknown-unknown path
```

Artifacts land in `wasm/dist/` (`binary_gemm_wasm.wasm` when the build succeeds).
CI and pytest do **not** require a successful compile — Python + JS scalar paths
are the always-on parity gates.

## Layout

| Path | Role |
|------|------|
| `binary_gemm_wasm.c` | Pedagogy C (scalar + `#ifdef __wasm_simd128__`) |
| `rust/` | Optional `cdylib` with `core::arch::wasm32` SIMD |
| `js/binary_gemm.mjs` | JS scalar + optional WASM loader |
| `js/demo_node.mjs` / `js/demo.html` | Demos |
| `build.py` | Optional toolchain driver |

See [`docs/spikes/WASM_SIMD.md`](../docs/spikes/WASM_SIMD.md) and lane note
[`docs/lanes/f.md`](../docs/lanes/f.md).
