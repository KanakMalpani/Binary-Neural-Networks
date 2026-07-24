# Evaluation, Robustness, Economics, Risk

## 32. Fair benchmark protocol

1. **Warm-up** ≥5 iters; report mean of ≥20 timed iters  
2. Pin threads / set `OMP_NUM_THREADS` / `torch.set_num_threads` explicitly  
3. Separate **kernel** vs **e2e** (this repo’s gemm_only vs model forward)  
4. Pre-pack weights once (deploy); don’t re-pack W every call  
5. Report device, CPU model, PyTorch build (CPU/CUDA), batch, shape  
6. Never equate theoretical 32×/64× with wall-clock  

Scripts: `benchmark.py`, `validate_native.py`, `wrap_existing_demo.py`.

## 33. Accuracy beyond MNIST — CLOSED-BY-PROXY

| Dataset | Metric | Binary protocol | Status here |
|---------|--------|-----------------|-------------|
| MNIST | Acc | 3–10 epochs Bi-Real/MLP | **Done** (`results/train_results.*`) |
| CIFAR-10 | Acc | Short Bi-Real vs FP twin | **Done** — 20k/5ep: FP 66.05% / Bin 54.90% (`results/cifar10_proxy.*`) |
| ImageNet | top-1/5 | Published Bi-Real/ReActNet | **ACCEPTED-NON-GOAL** in-repo (ADR); use papers for scale-up |
| GLUE / WikiText | Acc/PPL | BiBERT / BitNet eval harness | Protocol: lm-eval + published cards |
| MMLU / chat | Acc | BitNet-b1.58 HF cards | Use published |

**Decision:** beyond-MNIST vision risk is closed by local CIFAR proxy + literature ImageNet numbers. Full ImageNet train is optional product scale-up, not a research blank.

## 34. Robustness / adversarial / OOD — CLOSED-BY-PROXY

Local: `scripts/robustness_fgsm.py` → `results/robustness_fgsm.json`

| Model | Clean | FGSM ε=0.1 | Drop |
|-------|-------|------------|------|
| fp32_mlp | 97.08% | 62.99% | 34.1 pp |
| binary_mlp | 95.96% | 60.38% | 35.6 pp |

Both degrade under white-box FGSM; binary is not magically robust. Black-box / ImageNet attacks remain literature (discretization studies; ODG-Q). Do not claim security hardness from BNN alone.

## 35. Failure modes checklist (complete)

| ID | Failure | Mitigate |
|----|---------|----------|
| F1 | Acc collapse | Residuals, FP stem/head, ternary, distill |
| F2 | STE mismatch | ApproxSign/ReAct/SURGE |
| F3 | BN instability | BN+affine, momentum 0.9 |
| F4 | First/last binary | Keep FP |
| F5 | Train instability | Adam, clip, longer schedule |
| F6 | Fake PyTorch binary | Packed kernels |
| F7 | GPU XNOR lose to TC | Use FP8/INT4 |
| F8 | Amdahl / non-binary ops | Hybrid FFN wrap |
| F9 | Export latent FP | Pack assert ~32× |
| F10 | Ternary≠binary marketing | Name bits correctly |
| F11 | Bench re-pack W | Pre-pack |
| F12 | Wrong-arch DLL | MSVC x64 |
| F13 | PTQ BitNet wipe | Gradual λ / BitDistill |
| F14 | Dequant size illusion | Don’t keep float copy |
| F15 | Diffusion 1-bit | Prefer INT8/FP8 |

## 36. Licensing / reproducibility / deps

| Risk | Mitigation |
|------|------------|
| Python 3.14 vs package wheels | Pin 3.11/3.12 for torchvision/numba; custom CIFAR via HF `uoft-cs/cifar10` |
| Archived Larq | Vendor fork or pin commit |
| Model licenses (Llama, etc.) | Obey base license for GGUF/AWQ |
| Non-determinism | Seeds + deterministic cuDNN off for speed benches |
| Popcount numerical | Assert err=0 vs ±1 FP reference |

## 37. Cost economics

**Cloud GPU serve (order-of-magnitude):**

\[
\$/\mathrm{token} \propto \frac{\$/\mathrm{GPU\cdot hr}}{\mathrm{tokens/s}}
\]

INT4/FP8 increasing tokens/s and batch → lower $/token (~1.3–2× throughput often).

**Edge BOM:** MCU+FPGA BNN can beat GPU BOM for fixed vision tasks (FINN FPS/W).

**CPU local LLM:** GGUF/BitNet avoid GPU rent; electricity ≈ \(P\times t\); bitnet.cpp energy
cuts −55–82% (reported) → lower kWh/token. Local bind: `results/energy_bound.*`.

**This repo wrap:** RAM **147→13 MB** on demo MLP → denser multi-tenant CPU serving of small nets.

## E1–E2 extras

- **Privacy/side-channel:** smaller models reduce attack surface; binary may change leakage —
  don’t claim crypto hardness.
- **Popcount reproducibility:** cross-platform endianness of bit packing must be versioned.

## Closure

No material evaluation residuals remain. See `docs/19_GAP_CLOSURE_REPORT.md`.
