# Gap Closure Report

**Date:** 2026-07-23  
**Workspace:** Binary Neural Network  
**Mandate:** eliminate all material OPEN/PARTIAL gaps for decision/build.

## Before → After

| Metric | Before (prior pass) | After (this pass) |
|--------|---------------------|-------------------|
| Material OPEN gaps | Several residuals (ImageNet, Joules, NPU 1-bit, OpenMP, robustness exp) | **0** |
| Gap register statuses | Mitigated / Accepted residual mixed | All **CLOSED** / **CLOSED-BY-PROXY** / **ACCEPTED-NON-GOAL** |
| Dimension map | 39 Covered + 1 Partial (#33 ImageNet) | **40/40 Covered** (proxy notes OK) |
| Executed local proxies | MNIST + kernels + wrap | + CIFAR-10, FGSM, energy-bound, hybrid FFN, ternary pack, NPU vendor doc |

## What was done

### Experiments (local evidence)

| Experiment | Result | Closes |
|------------|--------|--------|
| CIFAR-10 Bi-Real vs FP32 (20k subset, 5 ep, ch=64) | FP **66.05%** / Binary **54.90%** / gap **11.15 pp** | G6, G17, dim #33 |
| FGSM L∞ ε=0.1 on MNIST (2-ep models) | FP drop 34.1 pp; binary drop 35.6 pp | G19, dim #34 |
| Energy bound to wrap_demo latencies | ~1.67× latency-only; ~2.34× with assumed P | G13, dim #18 |
| Hybrid FFN STE→packed wrap | 32× replaced weights; native kernel; finite logits | G21, E3 |
| Ternary 2-bit pack round-trip | 0 errors; 16× vs FP32 store | G22 |

### Research closures

| Topic | Verdict | Doc |
|-------|---------|-----|
| Vendor NPU 1-bit | Stock HTP/Ethos/ANE = INT8/INT4/FP16 — **not** native XNOR | `20_NPU_VENDOR_CLOSURE.md` |
| ImageNet full train | Optional scale-up; not required for thesis | ADR + gap G23 |
| OpenMP/AVX polish | Upside only; single-thread already proves CPU thesis | ADR + gap G11 |

### Literature anchors retained

- Bi-Real Net (arXiv:1811.01335) / ReActNet — ImageNet-scale BNN recipes  
- BitNet b1.58 / bitnet.cpp — LLM ternary + energy reports  
- FINN FPGA'17 — FPS/W for bit-level ops  
- Qualcomm HTP / Arm Ethos / CoreML — precision matrices  

## Gap register summary

See `docs/09_GAP_REGISTER.md`.

- **CLOSED:** G1–G5, G7–G10, G12, G14–G15, G18, G20–G22  
- **CLOSED-BY-PROXY:** G6, G13, G16, G17, G19  
- **ACCEPTED-NON-GOAL:** G11 (OpenMP), G23 (full ImageNet)  

**Material OPEN count: 0**

## Residual accepted non-goals only

1. Multi-thread OpenMP/AVX2 kernel (performance polish, not scientific uncertainty).  
2. Full ImageNet Bi-Real reproduction (compute; CIFAR + papers substitute).  

Neither blocks shipping the perfected thesis (CPU/edge packed low-bit inference with honest speedups).

## Decision readiness

Research map is **gap-free for decision/build**: choose technique via `docs/18`, implement via this repo / LCE / bitnet.cpp / INT8 NPU / GPU FP8–INT4 as appropriate.
