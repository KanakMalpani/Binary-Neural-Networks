# Deep Research Report: Extreme Low-Bit Neural Inference

**Date:** 2026-07-23  
**Workspace:** `Binary Neural Network`  
**Scope:** First principles → SOTA → failures → perfected product idea → working CPU proof

---

## Executive verdict

The naive pitch (“binarize the net → ~32× faster everywhere”) is **false as stated**.
The perfected pitch is true under conditions:

> On **CPU/edge/NPU**, **inference** matmuls that dominate runtime can be cut by
> **several×** in wall-clock (measured **2.4–5.0×** here; literature **~2–18×**) and
> **~32×** in weight bytes by packing **1-bit** (or **~16–20×** for ternary BitNet),
> *if and only if* you use real packed kernels and an accuracy-preserving recipe
> (Bi-Real / ReActNet / BitNet b1.58). On commodity **NVIDIA GPUs**, prefer
> **FP8/INT4**; classic BNNs rarely win against Tensor Cores.

---

## 1. First principles (compressed)

Neural inference cost ≈ max(arithmetic, **memory bandwidth**, overhead).
Large models at batch-1 are often bandwidth-bound. Binary packing attacks bytes moved:

\[
\text{bytes}_{\mathrm{bin}} = \text{bytes}_{\mathrm{FP32}} / 32
\]

Arithmetic becomes XOR+popcount with ~64× fewer *word* ops than scalar MACs — an upper
bound, not e2e latency. Full calc: `docs/06_CALCULATED_SPEEDUP_MODEL.md`.

---

## 2. SOTA map (2016–2026)

| Era | Method | Practical takeaway |
|-----|--------|--------------------|
| 2016 | BinaryNet, XNOR-Net | Foundational STE + scaling |
| 2018–2020 | Bi-Real, ReActNet | FP shortcuts + activation reshape close CNN gap |
| 2023–2026 | BitNet → b1.58 + bitnet.cpp | Ternary LLMs match FP16 from ~3B; CPU kernels 1.4–6× |
| 2024–2026 | torchao FP8/INT4 | Datacenter default |
| 2026 | Litespark, Sparse-BitNet | Further CPU ternary / sparsity |

Primary sources: Courbariaux et al. 2016; Rastegari et al. ECCV 2016; Liu et al. Bi-Real /
ReActNet; Ma et al. BitNet b1.58 arXiv:2402.17764; Microsoft bitnet.cpp; Larq CE MLSys.

---

## 3. Perfected idea (product)

See `docs/05_PERFECTED_CONCEPT.md`.

**Kill:** universal 32× binary training/inference claim.  
**Keep:** extreme low-bit **inference** on bandwidth-bound devices with packed kernels.  
**Split lanes:** edge vision BNN | CPU ternary LLM | GPU INT8/FP8.

---

## 4. Local experimental evidence

### 4.1 Compression & correctness

- Packing: **32.00×** weight bytes (`scripts/export_check.py`)
- GEMM: native vs ±1 FP32 **max abs err = 0**

### 4.2 Kernel wall-clock (CPU, pre-packed W)

| \(B\times N\times M\) | NumPy FP32 | Native compute | \(S\) | vs Torch FP32 |
|----------------------|-----------:|---------------:|------:|--------------:|
| 128×2048×2048 | 18.01 ms | 4.13 ms | **4.36×** | ~0.88× (torch wins small) |
| 64×4096×4096 | 22.02 ms | 6.10 ms | **3.61×** | **1.84×** |
| 32×8192×8192 | 113.87 ms | 12.26 ms | **9.29×** | **3.44×** |

Trend: larger \(N\) → higher speedup. E2E with act-packing still **7.32×** at 8192.

### 4.3 MNIST accuracy (3 epochs, CPU)

| Model | Test acc |
|-------|----------|
| fp32_mlp | 97.67% |
| binary_mlp | 96.36% (−1.3 pp) |
| ternary_mlp | 97.16% (−0.5 pp) |
| fp32_cnn | 96.61% |
| binary_cnn (Bi-Real) | 94.79% (−1.8 pp) |

### 4.4 Negative results (also evidence)

- Re-packing weights every call → false “slowdown” (measurement bug; fixed)
- Pure NumPy popcount ≪ native popcount
- `sign` + torch linear: **slower** than FP32 (fake binary)

---

## 5. Architecture chosen

Train: latent FP weights + STE binary/ternary forward, FP stem/head, BN, Bi-Real residuals.  
Infer: pack → native `binary_gemm_u64`.  
ADR: `docs/08_ADR.md`. Requirements: `docs/07_REQUIREMENTS.md`.

---

## 6. What works / what doesn’t

| Surface | Works? | Numbers / why |
|---------|--------|---------------|
| CPU packed binary GEMM | **Yes** | 2.4–5.0× vs FP32 here |
| ARM phone LCE | **Yes** (lit.) | 8.5–18.5× |
| bitnet.cpp ternary LLM | **Yes** (lit.) | 1.4–6× CPU |
| PyTorch fake BNN on GPU | **No speedup** | Loses to cuDNN |
| Datacenter BNN vs FP8 | **Usually no** | Tensor Cores |
| Training speedup via binary | **No** | Latent FP + STE overhead |

---

## 7. Gaps closed

See `docs/09_GAP_REGISTER.md` and **`docs/19_GAP_CLOSURE_REPORT.md`**.
Material OPEN count: **0**. Accepted non-goals: OpenMP polish; full ImageNet in-repo.

## 9. Completeness (2026-07-23 dimension audit)

All **40** mandated research dimensions are **Covered** in `docs/00_DIMENSION_MAP.md`
(CIFAR-10 proxy covers #33; ImageNet full train = ACCEPTED-NON-GOAL).

Dimension docs: `13`–`20` (incl. gap closure + NPU vendor closure).

---

## 8. How to reproduce

```bat
cd "C:\Users\mrkan\CRAZZY\Binary Neural Network"
python bnn\kernels\compile_native.py
python scripts\export_check.py
python scripts\validate_native.py
python scripts\benchmark.py
python scripts\train.py --epochs 3
python scripts\wrap_existing_demo.py --mode binary_xnor --hidden 4096 --batch 32
python scripts\energy_estimate.py --latency-s 0.0126 --power-w 25 --baseline-latency-s 0.0211 --baseline-power-w 35
```
