# Training Science: QAT, Distillation, Gradients, Scaling

## 1. Information capacity of ±1 weights & activations

A binary activation vector \(a\in\{\pm1\}^d\) has at most \(d\) bits of Shannon self-information
per sample (finite alphabet). A full-precision activation has continuous degrees of freedom
(practically ~16–24 bits of useful dynamic range after BN).

**Implication:** stacking \(L\) pure binary layers without residuals is a cascade of Hamming
maps. Representational capacity per layer collapses unless you restore information via:

- FP shortcuts (Bi-Real): \(y = \mathrm{BN}(\mathrm{BinConv}(x)) + x\)
- Multi-bit / multi-base (ABC-Net): \(K\) binary bases ≈ \(K\) bits capacity
- Ternary weights with 0 (BitNet b1.58): gating / feature selection
- Higher-bit activations (W1.58A8)

Rough VC-style intuition (not a theorem): hypothesis class of linear separators with ±1
weights is coarser; you need **more width or depth + residuals** to match FP VC capacity.
BitNet empirically: parity with FP16 from **~3B params** at equal tokens (arXiv:2402.17764).

## 2. STE and alternative gradient estimators

Forward: \(q = \mathrm{sign}(x)\). True \(\partial q/\partial x = 0\) a.e.

| Estimator | Backward approx | Source / note |
|-----------|-----------------|---------------|
| **STE (clip)** | \(1_{\|x\|\le1}\) | BinaryNet |
| **ApproxSign** (Bi-Real) | piecewise polynomial near 0 | tighterder STE |
| **IR-Net** | error-decay + libra parameter | Improves early training |
| **SWISH-Sign / Soft-Sign** | smooth surrogate | Research variants |
| **RSign / RPReLU** | learnable thresholds / slopes | ReActNet |
| **SURGE / DPGC** | learnable dual-path surrogate | ICML 2026 |

**Practice:** start with clipped STE + weight clip \(w\leftarrow\mathrm{clip}(w,-1,1)\); escalate to
ReActNet/SURGE if accuracy plateaus. Larq reports little difference among many STE variants
once architecture is Bi-Real-class.

## 3. Loss landscape, BN, optimizers

- **BN is mandatory** after binary layers: integer popcount outputs need affine rescale.
- Larq tip: BN momentum ≈ **0.9** (noisier updates than FP).
- **Adam** preferred over SGD for latent weights (faster, less LR brittle).
- Latent weights live in FP; only forward is discrete → landscape is piecewise-constant in
  discrete params but smooth in latents.
- Progressive quantization / λ-schedule (HF 1.58-bit FT blog) avoids “information wipe”
  when converting pretrained FP → ternary.

## 4. Scaling laws

| Regime | Observation |
|--------|-------------|
| Small CNN (MNIST) | Binary ≈ FP with Bi-Real (this repo: ≤2 pp) |
| ImageNet ResNet-18 class | Historical gap ~10 pp; ReActNet closes to ~3 pp on MobileNet-scale |
| LLM BitNet b1.58 | Matches FP16 twin from **~3B**; gap larger at ≤1B |
| Tokens | BitNet needs comparable token budgets; PTQ≠from-scratch |

**Rule:** extreme low-bit favors **scale + data + architecture**, not PTQ of tiny models.

## 5. Error propagation in deep binary stacks

Without residuals, each `sign` is a hard information bottleneck → compounding mismatch.
Mitigations: FP residual every block; real-valued shortcuts at downsample; keep stem/head FP;
optional knowledge distillation (logits + attention relations — BitDistill / MiniLM-style).

## 6–10. Algorithm family map (compressed)

| Family | W | A | Train | When |
|--------|---|---|-------|------|
| BinaryConnect | ±1 | FP | QAT | Memory |
| BinaryNet / XNOR | ±1 | ±1 | QAT | Edge CNN + kernels |
| Bi-Real / ReActNet | ±1 | ±1 | QAT+distill | Best 1-bit CNN recipe |
| DoReFa / LQ / ABC | 1–8 | 1–8 | QAT | Flexible bits |
| TWN / BitNet b1.58 | ternary | 8+ | from-scratch / distill | LLMs |
| BitNet a4.8 | 1–1.58 | 4 | QAT | Act compression |
| AWQ/GPTQ/torchao | 4–8 | 16/8 | PTQ | **Default LLM wrapper** |
| BitDistill | 1.58 | 8 | distill+CPT | Convert FP LLM→ternary for tasks |

**Weight-only vs W+A:** weight-only → memory; W+A → XNOR compute. LLMs usually W-low + A-8;
classic BNNs are W+A binary.

## 11–12. Transformers & sparsity

- **BiBERT** (arXiv:2203.06390): fully binarized BERT; Bi-Attention + direction-matching distill;
  claims ~56× FLOPs / ~31× size vs FP BERT (task-dependent).
- **BitNet Transformer**: replace `nn.Linear` with BitLinear (not full attention binary).
- **Sparse-BitNet / 2:4**: combine N:M sparsity with low-bit for Tensor Core speedups (~1.3× extra).

## Training-time vs inference-time

**Training is almost never faster** with binary: latent FP weights + STE + longer epochs.
**Inference-only** is the product win surface (this repo thesis).

## Residual

Full ImageNet/ReActNet reproduction is an **ACCEPTED-NON-GOAL** (ADR); local CIFAR-10
Bi-Real proxy + published Bi-Real/ReActNet ImageNet numbers close the accuracy dimension
(`results/cifar10_proxy.*`, `docs/17`, `docs/19`).
