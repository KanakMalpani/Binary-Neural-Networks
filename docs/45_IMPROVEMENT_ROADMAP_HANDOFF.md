# Improvement roadmap — agent handoff

**Written:** 2026-08-06 · **Repo state:** `bnn 1.0.0`, `main` @ `1728f9b`
**Audience:** an agent or engineer picking this up cold.

This is a *handoff*, not a wish list. Every item below is either **measured** (with
the number and how to reproduce it) or explicitly flagged as **unverified**. Where
something was investigated and deliberately not done, that is recorded too —
re-deriving a negative result is pure waste.

---

## 0. Orientation (do this first, ~3 minutes)

```bash
pip install -e ".[dev]" -c constraints.txt
python -m bnn.kernels.compile_native      # optional; NumPy fallback is correct without it
bnn repro                                  # must print REPRO: PASS
```

Then read, in order: [`AGENTS.md`](../AGENTS.md) → [`ROADMAP.md`](../ROADMAP.md) §0 →
this file. The canonical product plan is `ROADMAP.md`; this file only covers what
is *left* and what was *learned*.

### Verified current state

| | |
|---|---|
| Tests | **638 passed, 5 skipped** |
| Coverage (CI selection) | **~82%**, hard gate at 80% |
| Gates | `REPRO: PASS`, `export_check: PASS`, cross-ISA `err = 0` |
| Static | ruff clean, mypy clean (62 files), actionlint clean |
| Kernel here | `avx512` of `[scalar, avx2, avx512]` |

Re-verify everything with:

```bash
ruff check . && mypy && pytest -q -m "not slow and not hf" --cov=bnn --cov-fail-under=80
python scripts/repro_all.py --mode verify --skip-compile --skip-pytest
python scripts/validate_native.py
```

---

## 1. Invariants — breaking these is a regression, not a refactor

1. **Thesis lock.** Speedups come from packed CPU/edge kernels. Never claim GPU
   32× from `sign()`. Compression is *size*, not latency. See `AGENTS.md`.
2. **Dual metrics.** `compression_*` is theoretical; `*_ms` / `speedup_*` is
   wall-clock. `bnn/wrap/schema.py` enforces a note when both are large.
3. **`err = 0` is exact.** Binary/ternary GEMM is integer arithmetic. Every ISA
   path must be *bit-identical*, not merely close.
   (`tests/test_native_gemm.py`, `tests/test_ternary_isa_parity.py`)
4. **One bit layout.** `pack_bits_u64` in `bnn/kernels/packed.py` is the single
   definition (bit *j* of word *w* = element `64w+j`). Binary and both ternary
   bitplanes go through it; the C kernel decodes it. Drift here corrupts silently.
   (`tests/test_pack_layout_contract.py`)
5. **Fast tests never touch the network.** `tests/conftest.py` blocks sockets for
   non-`slow`/`hf` tests. A truncated CIFAR download is what last reddened CI.
6. **Goldens are authoritative.** `results/*.json` + `tests/golden_floors.json`.
   Regenerate deliberately, never as a side effect.
7. **No `-march=native`.** ISA is chosen at *run* time so one wheel stays valid
   across CPUs of the same architecture.

---

## 2. Priority 1 — the honest-claims bug

### P1. The NumPy fallback is slower than doing nothing

**Status:** measured, unfixed. **Impact:** high — it affects every user without a
compiler, and it undercuts a claim the README makes.

`binary_gemm_numpy_prepacked` is the fallback when no native library loads. For
batched shapes it is **5–11× slower than simply calling FP32 BLAS**:

| shape | numpy packed | plain fp32 | |
|---|---|---|---|
| 64 × 4096 × 4096 | 88.7 ms | 15.7 ms | **BLAS 5.7× faster** |
| 128 × 2048 × 2048 | 34.8 ms | 6.0 ms | **BLAS 5.8× faster** |
| 256 × 1024 × 1024 | 25.7 ms | 2.3 ms | **BLAS 11× faster** |
| 8 × 4096 × 4096 | 11.3 ms | 12.6 ms | packed wins |
| 1 × 4096 × 4096 | 1.4 ms | 2.4 ms | packed wins |

Crossover at N=M=4096 sits at **B ≈ 8–16** (it moves with thermal state and BLAS
thread count — two runs on the same machine put it at 8 and at 16). Below it
packed wins; above, the gap grows roughly linearly, because the packed path
loops over B in Python while BLAS threads and blocks. Treat the threshold as
needing a per-machine calibration, not a hard constant.

Reproduce: see the sweep in §6.

**Why this matters.** The README says "No compiler? The NumPy path keeps
correctness." True — but a reader reasonably infers it is also the *fast* path.
For B > 8 it is the slow path, and slower than not packing at all. The 32×
memory saving is untouched; only the speed story inverts.

**Suggested fix.** In `binary_gemm_packed` (and the wrap forward), when no native
library is loaded, dispatch on shape: use the packed NumPy path below the
crossover, else dequantise to ±1 and call BLAS. Both give `err = 0` — the packed
weights stay packed in memory, so compression is preserved either way.

Guard rails for whoever does this:
- Keep `binary_gemm_numpy_prepacked` public and unchanged; it is the reference
  implementation the ISA-parity tests compare against.
- The threshold is machine-dependent (see above). Make it a named constant with
  the measurement in a comment, overridable by env var, and pick it
  conservatively — being wrong toward BLAS costs little, being wrong toward
  the packed path costs 5x.
- Add a test asserting the fallback is never worse than FP32 by more than a small
  margin at B = 64.
- Update the README line so it distinguishes *correct* from *fast*.

---

## 3. Priority 2 — real but bounded

### P2. Ternary C kernel has no row blocking

`binary_gemm.c` gives the binary path 4-row register blocking (`BNN_BR`), so each
weight word is loaded once per 4 batch rows. `ternary_gemm_u64` still loops
`for (b = 0; b < B; ++b)` and re-streams both bitplanes per row. It *does* have
per-ISA `popcount_and`.

Measured ternary/binary ratio (ternary does 2 popcounts per output, so ~2× is the
floor):

| shape | ratio | headroom above floor |
|---|---|---|
| 64 × 4096 × 4096 | 1.22× | none — already under floor |
| 128 × 2048 × 2048 | 1.33× | none |
| 256 × 1024 × 1024 | 2.18× | ~1.1× |
| 8 × 4096 × 4096 | 2.91× | ~1.45× |

**Honest read:** worth ~1.1–1.45× on small-batch shapes only. Large batches are
already at or under the theoretical floor because both bitplanes share the loaded
X row. This is a nice-to-have, *not* a repeat of the 4–6× binary win. Do it only
if small-batch ternary latency matters to a real user.

### P3. Coverage gaps in newer modules

| module | coverage | missing |
|---|---|---|
| `bnn/energy/rapl.py` | **32%** | 63 |
| `bnn/cifar.py` | **43%** | 73 |
| `bnn/kernels/compile_native.py` | 62% | 54 |
| `bnn/codec/packfile.py` | **69%** | 93 |

`packfile.py` is the one to prioritise — 735 lines, the `.bnnpack` container is a
*persistence format*, and format bugs corrupt user artifacts silently. Prior art
for how to test it without network or big fixtures: `tests/test_pack_layout_contract.py`.

RAPL is hardware-dependent; test the parsing and unit conversion with synthetic
counter files rather than trying to read real MSRs in CI.

### P4. `bnn/cli.py` is 1060 lines

Every subcommand handler plus the parser in one module. It works and is tested at
the surface level, but it is the file most likely to cause merge pain. A
mechanical split (`bnn/cli/__init__.py` + a module per command group) is low risk
if `build_parser()` keeps its current shape — `tests/test_cli_surface.py`
enumerates subcommands from the parser, so it will catch a dropped command.

Do **not** do this at the same time as any behavioural change.

---

## 4. Priority 3 — needs a human or a budget

| Item | Blocker |
|---|---|
| **W8.T08 PyPI Trusted Publishing** | Human: configure the PyPI trusted publisher and the `pypi` GitHub environment. Workflow is already wired and gated behind a manual dispatch (`.github/workflows/wheels.yml`). See `docs/PYPI_PUBLISH.md`. |
| **W3.T08 distill integration** | Needs a real corpus and training budget; `scripts/distill_sketch.py` is a sketch. |
| **W4.T05 ResNet-BiReal reference** | Multi-hour CIFAR training to produce a golden. |
| Dependabot backlog | Several open PRs. **Note:** merging the `setuptools >= 83` bump means the `PYSEC-2026-3447` ignore in `.github/workflows/ci.yml` should be *removed*, or the gate silently carries a stale exemption. The `codeql-action` v3→v4 bumps rewrite the SHA pins. |
| Knowledge graph freshness | `knowledge_graph/` (165 nodes / 288 edges) is a new artifact with no CI check that it matches the repo. It will drift. Consider a test like `tests/test_docs_links.py` that validates node→file references still resolve. |

---

## 5. Measured and deliberately NOT done

Do not redo these without new evidence.

| Idea | Result |
|---|---|
| **Memory arena for packed buffers** (W2.T07) | Output allocation is **1.4–1.8%** of kernel time, and `torch.from_numpy(np.ascontiguousarray(y))` *aliases* the buffer — recycling would corrupt tensors the caller still holds. Full write-up: `docs/43_MEMORY_FOOTPRINT.md`. Revisit only above ~10%. |
| **Blocking the NumPy GEMM over batch** | **0.34×** at 256×1024×1024 — slower. Row-at-a-time is already cache-optimal. |
| Narrower reduce accumulators (`uint16`) in the NumPy GEMM | Within noise. |
| `np.unpackbits` for 2-bit decode | **0.73×** — slower than the 4-shift + LUT gather now in use. |
| Enforcing `ruff format` | Would reformat 42 files for no correctness gain. Lint is enforced; formatting is not. |
| `warn_unreachable` in mypy | 10 false positives from torch's `Module.__getattr__` union, 0 true. Off, documented in `pyproject.toml`. |

---

## 6. Traps discovered the hard way

**Unsigned wrap in the NumPy GEMM.** `bitwise_count(...).sum()` promotes
`uint8 → uint64` (*unsigned*). The `.astype(np.int32)` in
`binary_gemm_numpy_prepacked` is **load-bearing** — without it a negative dot
product becomes ~1.8e19. It looks exactly like a cast someone would tidy away.
Regression: `test_numpy_gemm_handles_negative_dot_products`.

**setuptools and `PyInit_`.** Building the kernel as an `Extension` makes
setuptools pass `/EXPORT:PyInit__binary_gemm_native`; this is a plain ctypes
library with no such symbol, so the link fails and leaves a **zero-byte** file.
`setup.py` overrides `get_export_symbols()` to `[]` and discards stale output
before retrying.

**`ascontiguousarray` does not copy** an already-contiguous array, so
`torch.from_numpy(np.ascontiguousarray(y))` shares memory with `y`. This is why
the arena idea is unsafe.

**macOS runner labels retire.** `macos-13` was removed by GitHub and silently
broke the Intel jobs. `actionlint` catches this — run it on every workflow edit.

**Reproducing the P1 measurement:**

```python
import time

import numpy as np

from bnn.kernels.packed import binary_gemm_numpy_prepacked, fp32_gemm, pack_binary_pm1


def best_ms(fn, reps=5, rounds=3, warmup=2):
    """Min-of-rounds mean; min is the least noisy estimator for short kernels."""
    for _ in range(warmup):
        fn()
    out = float("inf")
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        out = min(out, (time.perf_counter() - t0) / reps)
    return out * 1e3


rng = np.random.default_rng(0)
w = rng.choice([-1.0, 1.0], size=(4096, 4096)).astype(np.float32)
wp, _ = pack_binary_pm1(w, 1)

print(f"{'B':>4} {'packed ms':>10} {'fp32 ms':>9}  winner")
for B in (1, 4, 8, 16, 64):
    x = rng.choice([-1.0, 1.0], size=(B, 4096)).astype(np.float32)
    xp, n = pack_binary_pm1(x, 1)
    tp = best_ms(lambda: binary_gemm_numpy_prepacked(xp, wp, n))
    tf = best_ms(lambda: fp32_gemm(x, w))
    print(f"{B:>4} {tp:10.2f} {tf:9.2f}  {'packed' if tp < tf else 'BLAS'}")
```

---

## 7. How to work here

- **Measure before optimising.** Every perf change in this repo was profiled
  first; two of the last three ideas were abandoned on the evidence. Post the
  before/after in the commit message.
- **Prove equivalence before claiming speed.** Packed kernels are exact integer
  work: assert bit-identity against the previous implementation across odd shapes
  and all-zero / all-±1 edges, not just random matrices.
- **Commits are conventional** (`perf(kernels): …`) with the measurement in the
  body. See `1728f9b` for the expected shape.
- **CI is the contract:** ruff and mypy are hard gates, coverage floor is 80%,
  `docs` job runs `mkdocs build --strict`, `pip-audit` hard-gates the shipped
  dependency set with every ignore triaged inline.
- If you find something worth *not* doing, add it to §5 with the number.
