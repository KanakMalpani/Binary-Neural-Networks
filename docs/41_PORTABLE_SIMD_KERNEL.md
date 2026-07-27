# Portable SIMD kernel + fused epilogue (W2.T04 / W2.T05 delivered)

Closes the two deferred SIMD notes — [`spikes/ARM_NEON_SPIKE.md`](spikes/ARM_NEON_SPIKE.md)
and [`spikes/AVX512_MOONSHOT.md`](spikes/AVX512_MOONSHOT.md) — and removes the
glue overhead that dominated a wrapped `Linear` once the GEMM got faster.

## What changed

### 1. Runtime ISA dispatch (one binary, any CPU)

`bnn/kernels/binary_gemm.c` picks the fastest legal path at **run** time:

| Path | Instruction | Availability |
|------|-------------|--------------|
| `avx512` | `_mm512_popcnt_epi64` (VPOPCNTDQ) | Ice Lake+, Zen 4+ |
| `avx2` | `vpshufb` nibble-LUT popcount | Haswell+ (2013+) |
| `neon` | `vcntq_u8` + `vpadalq_u8` | all ARM64 |
| `scalar` | `__popcnt64` / `__builtin_popcountll` | everywhere |

Detection uses `cpuid` **plus** an `xgetbv` check that the OS actually enabled
YMM/ZMM state — a CPU flag alone is not enough and skipping this is a classic
source of `SIGILL` on otherwise-capable machines.

Critically, the build never passes `-march=native`. The object stays portable to
any CPU of the same architecture; baking in build-host ISA would defeat the
entire point of dispatching at run time.

Inspect and override:

```python
from bnn.kernels.packed import kernel_name, cpu_features, available_kernels, set_kernel

kernel_name()        # 'avx512'
available_kernels()  # ['scalar', 'avx2', 'avx512']
set_kernel('scalar') # force a path; returns what is actually in effect
```

```bash
BNN_KERNEL=scalar pytest -q tests/test_native_gemm.py
```

`BNN_KERNEL` accepts `scalar|avx2|avx512|neon`. An unsupported or misspelled
value falls back to auto-detection rather than failing.

### 2. Batch register blocking + one OpenMP region

The old kernel opened a **new parallel region per batch row** and re-streamed the
whole weight matrix `B` times:

```c
for (b = 0; b < B; ++b)          /* B fork/joins */
    #pragma omp parallel for
    for (m = 0; m < M; ++m) ...
```

Now a single team is forked per call, and each weight word is loaded once and
reused across a block of 4 batch rows (`BNN_BR`), cutting weight-side memory
traffic ~4x and giving 4 independent popcount dependency chains. Blocks use
`nowait` — safe because each block writes a disjoint slice of `Y` and only reads
`X`/`W`.

### 3. Fused alpha/bias epilogue

`binary_gemm_u64_scaled(X, W, Y, alpha, bias, ...)` computes
`Y = alpha * (n - 2*hamming) + bias` in one pass. Previously the wrapper did
`y *= alpha; y += bias` in NumPy — two extra passes over the `(B, M)` output that,
once the GEMM was vectorised, cost as much as the GEMM itself.

The unfused entry point `binary_gemm_u64` is unchanged, and Python falls back to
it automatically if the loaded library predates the fused symbol.

### 4. Activation packing

`_pack_activations_fast` expanded the batch into a `(B, words, 64)` uint64
temporary. It now delegates to `pack_binary_pm1` (`np.packbits`) — **bit-identical
output, ~6.5x faster**, and it had become the single largest cost in the forward
pass.

## Measured (this machine, 16 threads, AVX-512 VPOPCNTDQ)

Wall clock is machine-dependent; correctness is not. Same process, both
libraries loaded, runs interleaved, min-of-5.

### Packed GEMM, compute only

| Shape (B×N×M) | before | after | speedup |
|---|---|---|---|
| 8 × 4096 × 4096 | 0.212 ms | 0.062 ms | 3.4× |
| 64 × 4096 × 4096 | 1.999 ms | 0.437 ms | 4.6× |
| 128 × 2048 × 2048 | 0.626 ms | 0.152 ms | 4.1× |
| 32 × 8192 × 8192 | 4.748 ms | 1.054 ms | 4.5× |
| 256 × 1024 × 1024 | 0.739 ms | 0.119 ms | 6.2× |
| 512 × 512 × 512 | 2.038 ms | 0.103 ms | 19.9× |
| **aggregate (12 shapes)** | **10.74 ms** | **2.12 ms** | **5.1×** |

Tiny shapes (e.g. 4×63×7) are call-overhead bound and unchanged — as expected.

### Wrapped `Linear`, end to end (`bnn profile`)

At the committed golden shape `32 × 1024 × 1024`:

| Metric | committed `results/profile.json` | after |
|---|---|---|
| `gemm_ms` | 0.410 | 0.080 |
| `e2e_forward_ms` | 1.396 | 0.317 |
| `speedup_vs_fp32` | **0.60** (slower than FP32) | **1.83** |

The wrapper previously *lost* to torch FP32 at this size. At
`64 × 4096 × 4096`, `overhead_vs_gemm` fell from 2.98 to 0.20.

## Correctness

Binary GEMM is exact integer arithmetic, so the bar is **err = 0**, not "close".

- `tests/test_native_gemm.py` runs every ISA path the host supports across 8
  shapes chosen to hit each blocking and vector-remainder boundary (batch below /
  at / above the 4-row block; word counts hitting the 8-, 4- and 2-word vector
  tails; sub-64 padding), asserting each path equals both the FP32 reference and
  the NumPy path exactly.
- The fused epilogue is checked against the unfused two-pass form (`rtol=1e-5`;
  float32 re-association is the only permitted difference), and with
  `alpha=bias=None` it must reproduce the plain GEMM **exactly**.
- `scripts/validate_native.py` repeats the cross-ISA equivalence check and prints
  the selected path.
- CI runs the suite a second time with `BNN_KERNEL=scalar` on every platform, so
  the fallback that unknown CPUs will take is never untested.

## Non-claims

Unchanged from the project thesis:

- This is **CPU/edge inference**, not a GPU win, and not 32× end-to-end.
- Speedups are wall-clock on one machine; the ratio moves with core count,
  memory bandwidth and thermal state. Correctness (`err = 0`) is the invariant.
- AVX-512 is used **when present**. It is never required to install, build or run.
