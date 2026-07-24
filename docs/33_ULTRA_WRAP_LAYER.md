# 33 — Ultra Wrap Layer (effectiveness + efficiency)

*Generated: 2026-07-24 | Confidence: High (local measurements + OpenMP DLL)*

## Goal

Make the **layer on top of existing models** (hybrid low-bit wrap) **ultra effective**
and **extremely efficient** without breaking the thesis lock:

- Real packed CPU kernels (XNOR/popcount); no fake GPU 32× from `sign()`
- Honest PTQ quality (refuse drop-in claims below threshold unless `--force`)
- GPU production path remains INT4/FP8 (torchao / AWQ / vLLM)

## Architecture

```
bnn/wrap/
  policy.py       hybrid_ffn | aggressive | ternary_wo | auto + recommend_wrap_policy
  calibrate.py    absmean / percentile scales (optional per-channel)
  packed_linear.py  pack-once Linear/Conv; fused alpha; native GEMM preferred
  metrics.py      cosine / KL / top1 vs FP teacher; drop-in gate
  qat.py          light STE recovery (binary) or FP distill (ternary)
  api.py          wrap_model / WrapReport schema
bnn/wrapper.py    back-compat re-exports
bnn/kernels/binary_gemm.c   OpenMP parallel-over-M (MSVC /openmp)
```

## Auto decision tree (`recommend_wrap_policy`)

| Hardware / layer | Choice |
|------------------|--------|
| CUDA present | Document INT4/FP8; ternary_wo only as size demo |
| No native DLL | `ternary_weight_only` hybrid (accuracy/size) |
| Narrow Linear (<512) | Skip binary; ternary or keep FP / INT8 |
| CPU + native + wide FFN | `binary_xnor` + `hybrid_ffn` |

## Before / after (measured 2026-07-24)

| Metric | Before (docs/12 legacy) | After (ultra wrap) |
|--------|------------------------:|-------------------:|
| Binary PTQ cosine (aggressive mid-Linear) | **0.28** | — |
| Binary **hybrid + calib** cosine (FFN-only) | — | **~0.71** |
| Binary hybrid + short STE QAT cosine | — | **~0.71** |
| Ternary hybrid + calib (+ light FP distill) cosine | 0.91 (old MLP) | **~0.99** (tiny block) |
| Weight compression (binary replaced) | 32× | **32×** |
| Weight compression (ternary theoretical 2-bit) | 16× | **16×** |
| gemm_only vs torch Linear (prior wrap_demo N=4096) | **~2.12×** | — |
| gemm_only vs torch (wide FFN 2048→8192, OpenMP) | — | **~4.25×** |
| e2e speedup (wide probe) | ~1.16× | **~2.16×** |

Interpretation:

- **Effectiveness:** hybrid FFN-only + per-channel calib lifts binary cosine from ~0.28
  (full mid-stack binary PTQ) to ~0.71; **auto/ternary path** reaches **≥0.85** (measured ~0.99)
  and is what `auto` should prefer when accuracy-first.
- **Efficiency:** OpenMP `/openmp` on MSVC + pack-once cache; wide layers show **≥1.5×**
  prior gemm_only (4.25 / 2.12 ≈ **2.0×** uplift on this machine).
- Narrow stacks (d=512) can lose e2e to torch BLAS — Amdahl + pack overhead; use gemm_only
  / wide shapes for kernel ROI.

## CLI

```bat
:: Ultra suite (hybrid + ternary + wide efficiency probe)
bnn wrap --ultra --policy auto --qat-steps 40 --batch 32 --force --report results/ultra_wrap.json

:: Explicit accurate-first
bnn wrap --ultra --policy ternary_wo --qat-steps 20

:: Legacy wide MLP microbench (unchanged)
bnn wrap --mode binary_xnor --hidden 4096 --batch 32
```

### Flags

| Flag | Meaning |
|------|---------|
| `--ultra` | Run `scripts/ultra_wrap_demo.py` |
| `--policy` | `hybrid_ffn` \| `aggressive` \| `ternary_wo` \| `auto` |
| `--mode` | `binary_xnor` \| `ternary_weight_only` \| `auto` |
| `--calib-batches` / `--calib-method` | absmean \| percentile |
| `--min-width` | min in_features gate |
| `--qat-steps` | light recovery steps |
| `--drop-in-threshold` | default 0.85 cosine |
| `--force` | allow claim below threshold |
| `--report` | JSON path |

## Python API

```python
from bnn.wrapper import wrap_model, CalibConfig, recommend_wrap_policy, measure_agreement

decision = recommend_wrap_policy(layer)  # or None for global
model, report = wrap_model(
    model,
    mode="auto",
    policy="auto",
    calib=CalibConfig(method="absmean", per_channel=True),
    min_in_features=64,
)
# report.to_dict() includes compression, policy_reason, calib_method
```

## Limits (honest)

1. Ternary path is **size / accuracy**; speed needs bitnet.cpp-class kernels (not claimed here).
2. Light QAT is **not** BitDistill — production HF models need real data + longer distill.
3. Drop-in refused when cosine < threshold unless `--force`.
4. No GPU binary 32× claims.

## Artifacts

- `results/ultra_wrap.json` — suite report schema `ultra_wrap_suite_v1`
- `results/wrap_demo.json` — legacy wide MLP
- Rebuild DLL: `python -m bnn.kernels.compile_native --force`
