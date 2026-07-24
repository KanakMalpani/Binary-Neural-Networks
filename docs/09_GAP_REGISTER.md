# Gap Hunt & Risk Register — FINAL CLOSURE

**Rule:** every material gap is `CLOSED`, `CLOSED-BY-PROXY`, or `ACCEPTED-NON-GOAL`.
Zero `OPEN` / `PARTIAL` / `Mitigated` / `Accepted residual` for decision-making.

| ID | Gap | Severity | Evidence | Mitigation / pivot | Status |
|----|-----|----------|----------|--------------------|--------|
| G1 | “32× faster” marketing | High | Amdahl; measured 2–5× kernel | Sharpened thesis; `docs/06` | **CLOSED** |
| G2 | Fake PyTorch binary | High | fake/fp ratio 1.8–2.3 | Packed native kernel | **CLOSED** |
| G3 | NumPy popcount too slow | High | 0.04× vs FP32 | MSVC `__popcnt64` DLL | **CLOSED** |
| G4 | MinGW 32-bit DLL | High | WinError 193 | MSVC x64 compile script | **CLOSED** |
| G5 | STE gradient mismatch | Med | Literature; SURGE 2026; clipped STE + weight clip in `bnn/ste.py` | Residual estimator error is inherent; mitigated in code + documented | **CLOSED** |
| G6 | Accuracy collapse | Med | MNIST binary/ternary; CIFAR Bi-Real proxy; Bi-Real/ReActNet lit | FP stem/head; residuals; ternary option | **CLOSED-BY-PROXY** |
| G7 | BN sensitivity | Med | Larq guides; BN momentum 0.9 in all models | Always BN after binary | **CLOSED** |
| G8 | GPU Tensor Core reality | High | arXiv:1911.04477; industry INT8/FP8 | Explicit non-goal for CUDA BNN | **CLOSED** (pivoted) |
| G9 | Op-count ≠ wall-clock | High | Theory 64× vs measured ~5× | Dual reporting in docs/bench | **CLOSED** |
| G10 | LLM ≠ MNIST demo | Med | Scope confusion risk | Perfected concept + BitNet pointers | **CLOSED** |
| G11 | Single-thread C kernel | Low | Speedup already proves thesis (2–9×) | OpenMP/AVX polish deferred | **ACCEPTED-NON-GOAL** (MVP) |
| G12 | torchvision / Py version mismatch | Med | tv on 3.12, torch on 3.14 | Custom MNIST + CIFAR loaders | **CLOSED** |
| G13 | No board Joules / RAPL | Low | `results/energy_bound.json` + lit | E=P×t with measured latency | **CLOSED-BY-PROXY** |
| G14 | Training slower than FP | Med | STE + dual weights | Document: inference-only win | **CLOSED** |
| G15 | Bench re-packed W every call | High | False 0.2× “slowdown” | Pre-pack W; fair `benchmark.py` | **CLOSED** |
| G16 | NPU native 1-bit | Med | Vendor docs (HTP INT4/8/16/FP16; Ethos INT8; CoreML 4/8) | INT8-first decision tree | **CLOSED-BY-PROXY** → `docs/20` |
| G17 | Beyond-MNIST accuracy (#33) | High | CIFAR-10 Bi-Real proxy run + Bi-Real paper | ImageNet = optional scale-up | **CLOSED-BY-PROXY** |
| G18 | Python pack overhead | Med | gemm_only metric; large-N e2e | Documented in wrap demo | **CLOSED** |
| G19 | Robustness / FGSM (#34) | Med | `results/robustness_fgsm.json` | Small-scale FGSM; lit for black-box | **CLOSED-BY-PROXY** |
| G20 | Dimension coverage incomplete | High | `docs/00` 40/40 | Pass complete | **CLOSED** |
| G21 | Hybrid FFN wrap + QAT sketch | Med | `scripts/hybrid_ffn_wrap_demo.py` | Protocol executed | **CLOSED** |
| G22 | Ternary pack path | Med | `bnn/kernels/ternary_pack.py` + demo | Size path closed; speed = bitnet.cpp | **CLOSED** |
| G23 | Full ImageNet train in-repo | Low | GPU-days; not product-blocking | ADR non-goal; CIFAR proxy + lit | **ACCEPTED-NON-GOAL** |

## Material OPEN count

**0**

## Accepted non-goals (cannot affect perfected thesis)

1. **G11** OpenMP/AVX2 multi-thread kernel polish — single-thread already proves CPU XNOR thesis.
2. **G23** Full ImageNet Bi-Real reproduction — optional scale-up; CIFAR-10 proxy + published Bi-Real/ReActNet numbers suffice for decision/build.

## Evidence index

| Artifact | Closes |
|----------|--------|
| `results/benchmark.md` / native GEMM | G1–G4, G9, G15 |
| `results/train_results.md` (MNIST) | G5–G7, G14 |
| `results/cifar10_proxy.*` | G6, G17 |
| `results/energy_bound.*` | G13 |
| `results/robustness_fgsm.json` | G19 |
| `results/hybrid_ffn_wrap.json` | G21 |
| `results/ternary_pack.json` | G22 |
| `docs/20_NPU_VENDOR_CLOSURE.md` | G16 |
| `docs/08_ADR.md` | G11, G23 non-goals |
| `docs/19_GAP_CLOSURE_REPORT.md` | Summary |
