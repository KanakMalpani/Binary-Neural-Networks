# Stop Claiming 32×: Honest Speedup Accounting for Binary Neural Networks

**Tech-report skeleton (W12 / Wave P2)** — in-repo markdown, not an arXiv upload.

| Field | Value |
|-------|-------|
| **Status** | Skeleton from committed goldens. **Human** submits arXiv / venue. |
| **Date** | 2026-08-15 |
| **Claims** | Whitelist **C1–C7** only ([`docs/PUBLICATION_PLAN.md`](../PUBLICATION_PLAN.md)) |
| **Software** | [`bnn-lab` 1.0.0](https://pypi.org/project/bnn-lab/1.0.0/) · [`CITATION.cff`](../../CITATION.cff) |
| **Official code** | <https://github.com/KanakMalpani/Binary-Neural-Networks> |
| **Papers with Code** | Code-link note below; register **after** arXiv (human) |
| **Machine-check** | `bnn bridge figures --out results/figures_manifest.json` |

> **Thesis lock.** Packed CPU / edge XNOR–popcount kernels plus honest STE *simulation*.
> Weight compression **32.00×** is exact for aligned uint64 pack. It is **not**
> end-to-end latency. Never claim GPU 32× from `sign()` / STE.

---

## Abstract

Binary neural network (BNN) write-ups still advertise **32× weight compression**
or **~64× word-op reduction** as if those ratios were wall-clock speedups. On
commodity CPUs they are not: a real packed XNOR–popcount kernel can be several
times faster than a NumPy FP32 GEMM, while a PyTorch `sign()` + FP `Linear`
(“fake binary”) is often *slower* than the FP32 baseline, and Amdahl’s law caps
end-to-end gains once stems, heads, and Python overhead remain in float.

This tech report states a **dual-metric** reporting protocol — packing /
arithmetic ratios *separately from* measured \(S_{\mathrm{compute}}\) and
\(S_{\mathrm{e2e}}\) — with an explicit fake-binary negative control. Every
numerical claim is taken from committed `results/*.json` and
`tests/golden_floors.json` (whitelist C1–C7). On published kernel shapes,
uint64 pack compression is **32.00×** (C1) and native GEMM error is **0** (C2);
measured compute speedups vs NumPy FP32 are **12–29×**, not 32–64×. A
Sequential wrap demo on hidden=4096 reports the same **32.00×** pack ratio
beside **2.65×** e2e wall-clock after short QAT — dual metrics, not a 32×
latency claim. MNIST / CIFAR-proxy / synthetic-audio canaries sit within
published floors (C4–C6). Linux + Windows CI validates a native Linux `.so`
(C7). We do not report ImageNet SOTA, production ASR, or GPU e2e 32×.

**Contribution class:** measurement and reporting standard with reproducible
lab evidence — not a new binarization algorithm.

---

## 1. Introduction

The marketing conflation is old: 1-bit weights occupy \(1/32\) the bytes of
FP32, and packing 64 ±1 values into a `uint64` word reduces the *count* of
word-level XOR+popcount ops by \(\approx 64\times\) versus scalar FP MACs.
Neither quantity is a stopwatch.

Three failure modes keep showing up in demos and papers:

1. **Theory-as-latency** — quoting 32× / 64× in a speedup table without
   \(S_{\mathrm{e2e}}\).
2. **Fake binary** — `torch.sign` (or STE graphs) still executing FP GEMM.
3. **Hybrid Amdahl** — replacing a fraction \(f\) of runtime while stem, head,
   attention, LayerNorm, and framework overhead stay float.

This lab’s public artefact (`bnn-lab`) already refuses the conflation in
software: compression fields are pack ratios; latency fields are wall-clock;
drop-in quality is a separate cosine / KL / top-1 record. The paper’s job is to
write that culture down so authors, reviewers, and coding agents can check it
against goldens rather than anecdotes.

**Non-goals (forbidden here):** GPU Tensor-Core 32× from `sign()`; invented
bench shapes sold as “the” golden; bit-identical floats across machines as a
pass criterion; production ASR; full ImageNet SOTA.

---

## 2. Claims whitelist (C1–C7)

Allowed claims — each row is machine-checkable. Do not add a C8 in this
skeleton.

| ID | Claim | Committed evidence | Recorded value |
|----|-------|--------------------|----------------|
| **C1** | Aligned uint64 binary pack compression **32.00×** (theory / size, not latency) | `tests/golden_floors.json` `compression_exact_when_uint64_pack` + [`results/wrap_demo.json`](../../results/wrap_demo.json) `weight_compression_replaced_layers` + [`results/benchmark.json`](../../results/benchmark.json) `theoretical.weight_compression` | **32.0** |
| **C2** | Native XNOR-popcount GEMM **err = 0** vs ±1 FP when DLL / `.so` present | floors `native_err_max: 0` + `benchmark.json` `max_abs_error_vs_fp32` | **0.0** on all three published shapes |
| **C3** | Dual-metric culture: theory vs wall-clock; never GPU 32× from `sign()` | Thesis lock + floors `notes` + kernel / wrap tables below | policy; see §4.3 |
| **C4** | MNIST binary MLP within floors | `mnist.binary_mlp_min_acc` vs [`results/train_results.json`](../../results/train_results.json) | **96.36%** (floor **95.0%**) |
| **C5** | CIFAR Bi-Real *proxy* within floors | `image_cifar.binary_bireal_min_acc` vs [`results/image_cifar.json`](../../results/image_cifar.json) | **61.14%** (floor **55.0%**) |
| **C6** | Audio *synth* binary CNN within floors | `audio_synth.binary_cnn_min_acc` vs [`results/audio_synth.json`](../../results/audio_synth.json) | **96.0%** (floor **85.0%**) |
| **C7** | Linux + Windows CI; Linux native `.so` validated in Actions | [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml) jobs `windows`, `linux-native` | native `validate_native` on Ubuntu |

Machine-check:

```bat
bnn bridge figures --out results/figures_manifest.json
bnn repro
```

Expect `claims_ok` for C1–C6 in the manifest (C7 is the CI matrix, not a JSON
floor) and `REPRO: PASS`.

### 2.1 Explicitly not claimed

| Tempting sentence | Why it is out |
|-------------------|---------------|
| “32× faster inference” | C1 is pack density; wall-clock is C3 |
| “GPU 32× from binary `sign()`” | Thesis lock; STE is simulation |
| “ImageNet SOTA Bi-Real / ReActNet” | C5 is a CIFAR-10 *proxy* (30k subset, 8 epochs) |
| “Production ASR” | C6 is synthetic tones; audio JSON says INT8 Whisper/ORT for real speech |
| “Ultra wrap TinyBlock meets the 0.85 ∧ 1.5× AND-gate” | [`results/ultra_wrap.json`](../../results/ultra_wrap.json) hybrid cosine **~0.70**, `REFUSE_DROP_IN_CLAIM` |
| “Ternary 0.991 cosine is the wrap win” | Same file: e2e **0.73×** (slower than FP). Cosine-only is not dual-metric |

---

## 3. Dual-metric protocol

Protocol companion: [`docs/FAIR_EVAL_PROTOCOL.md`](../FAIR_EVAL_PROTOCOL.md),
[`docs/BENCH_SHAPES.md`](../BENCH_SHAPES.md),
[`docs/06_CALCULATED_SPEEDUP_MODEL.md`](../06_CALCULATED_SPEEDUP_MODEL.md).

### 3.1 Four quantities (never collapse them)

| Symbol | Meaning | Typical source |
|--------|---------|----------------|
| Compression | FP32 bytes / packed bytes for **replaced** binary weights | uint64 pack; **32.00×** when aligned (C1) |
| \(R_{\mathrm{arith}}\) | FP MACs per out / uint64 XOR+popcount words | \(\approx 64\) for \(N \bmod 64 = 0\) — **not** latency |
| \(S_{\mathrm{compute}}\) | Pre-packed binary GEMM vs FP GEMM (weights packed **once**) | `benchmark.json` `speedup_compute_vs_numpy_fp32` |
| \(S_{\mathrm{e2e}}\) | Forward including activation pack / framework | `speedup_e2e_vs_numpy_fp32` or wrap `e2e_speedup` |

Identity used by the kernel (XOR form; pad bits encode \(+1\)):

\[
\langle x, w \rangle = n - 2\cdot\mathrm{popcount}(x_{\mathrm{bits}} \oplus w_{\mathrm{bits}})
\]

for \(x,w \in \{+1,-1\}^n\). See [`docs/35_BINARY_MATH_EFFECTIVENESS.md`](../35_BINARY_MATH_EFFECTIVENESS.md).

### 3.2 Fake-binary negative control (required)

**Fake binary** = apply `sign()` (or STE) in-graph while the matmul remains FP
GEMM. Extra kernels, no packing, no bandwidth win. Any BNN *systems* table that
omits this control can launder simulation graphs as speed.

Committed ratios `fake_binary_vs_torch_fp32` (time of `sign`+FP Linear /
time of torch FP32 Linear; **>1 means slower**):

| Shape \(B \times N \times M\) | Fake-bin / torch FP32 |
|------------------------------|----------------------:|
| 128×2048×2048 | **1.44×** (slower) |
| 64×4096×4096 | **1.46×** (slower) |
| 32×8192×8192 | **1.29×** (slower) |

Source: [`results/benchmark.json`](../../results/benchmark.json). This is C3
evidence, not a new claim ID.

### 3.3 Amdahl (why e2e ≠ kernel)

Let \(f\) be the fraction of runtime in replaceable matmuls, sped up by
\(S_{\mathrm{kernel}}\):

\[
S_{\mathrm{e2e}} = \frac{1}{(1-f) + f/S_{\mathrm{kernel}}}
\]

Even \(S_{\mathrm{kernel}}=32\) cannot yield 32× e2e unless matmuls dominate.
Hybrid nets (FP stem/head/attention) make \(f < 1\) by construction.

### 3.4 Fair bench rules

- Published shapes only ([`docs/BENCH_SHAPES.md`](../BENCH_SHAPES.md)).
- Weights packed **once** (deploy-time). Re-packing \(W\) every call is a known
  false-slowdown.
- Disclose CPU, OS, torch pin, native vs NumPy, thread count, warmup.
- Floats need not be bit-identical across machines; **conclusions** vs
  `golden_floors.json` must agree ([`REPRODUCIBILITY.md`](../../REPRODUCIBILITY.md)).

### 3.5 Energy proxy (supporting; not C1–C7)

Board Joules / RAPL are a documented moonshot on Windows.
[`results/energy_bound.json`](../../results/energy_bound.json) records a
**proxy** \(E \approx P \cdot t\) with assumed power brackets. That file’s
`measured_latency_s` is a **prior wrap-demo latency snapshot** (FP 34.14 ms →
wrapped 7.09 ms) and must **not** be mixed with the current
[`results/wrap_demo.json`](../../results/wrap_demo.json) stopwatch
(50.24 ms → 18.94 ms). Cite the proxy as methodology, not as a C-claim, and
not as the 2026-08-15 wrap golden.

---

## 4. Experiments (committed goldens)

Headline kernel table: [`docs/34_COMPUTE_SPEEDUP.md`](../34_COMPUTE_SPEEDUP.md)
tracks the same JSON. Numbers below are copied from
[`results/benchmark.json`](../../results/benchmark.json) (native kernel true,
OpenMP true, warmup=5, reps=10).

### 4.1 C1 — compression is 32.00× pack density

Every published kernel row sets `theoretical.weight_compression = 32.0`.
`wrap_demo.json` sets `weight_compression_replaced_layers = 32.0` with
FP32 replaced bytes 134 217 728 → packed 4 194 304.

That ratio is **bytes of replaced weights**, not model FPS and not GPU
throughput.

### 4.2 C2 — native GEMM err = 0

| Shape | `max_abs_error_vs_fp32` |
|-------|------------------------:|
| 128×2048×2048 | **0.0** |
| 64×4096×4096 | **0.0** |
| 32×8192×8192 | **0.0** |

When the platform DLL / `.so` is absent, the NumPy packed path remains the
correctness fallback (`err = 0` in pytest). C2 is the *native* gate.

### 4.3 C3 — theory vs wall-clock on the same shapes

| Shape | Compression (theory) | \(R_{\mathrm{arith}}\) (theory) | \(S_{\mathrm{compute}}\) vs NumPy | \(S_{\mathrm{e2e}}\) vs NumPy | \(S_{\mathrm{compute}}\) vs torch | Fake-bin / torch |
|-------|---------------------:|--------------------------------:|----------------------------------:|------------------------------:|----------------------------------:|-----------------:|
| 128×2048×2048 | 32.0 | 64 | **11.99×** | **8.20×** | 5.94× | 1.44× slower |
| 64×4096×4096 | 32.0 | 64 | **23.86×** | **14.18×** | 18.10× | 1.46× slower |
| 32×8192×8192 | 32.0 | 64 | **29.25×** | **20.72×** | 50.73× | 1.29× slower |

Read the table left-to-right: 32× and 64× never equal the stopwatch. \(S\) vs
torch is machine- and BLAS-dependent; do not promote a single cell (including
the large 8192-vs-torch number) as “the” BNN speedup. Prefer dual reporting
vs **both** NumPy FP32 and torch FP32, plus e2e vs compute.

### 4.4 C4 — MNIST binary MLP canary

From [`results/train_results.json`](../../results/train_results.json)
(CPU, 3 epochs, seed in floors):

| Model | Test acc % | Floor min |
|-------|-----------:|----------:|
| fp32_mlp | 97.67 | 96.0 |
| **binary_mlp** | **96.36** | **95.0** |
| ternary_mlp | 97.16 | 95.0 |

Pedagogy / regression canary — not a vision SOTA claim.

### 4.5 C5 — CIFAR-10 Bi-Real *proxy*

From [`results/image_cifar.json`](../../results/image_cifar.json)
(train subset **30 000**, **8** epochs):

| Model | Test acc % | Floor min |
|-------|-----------:|----------:|
| fp32_cifar_cnn | 71.14 | 60.0 |
| **binary_cifar_bireal** | **61.14** | **55.0** |
| Gap (pp) | 10.0 | max 15.0 |

**Not ImageNet. Not ReActNet-A.** Protocol canary only.

### 4.6 C6 — synthetic-audio binary CNN

From [`results/audio_synth.json`](../../results/audio_synth.json)
(8-class musical-tone spectrograms, 800 train / 200 test):

| Model | Test acc % | Floor min |
|-------|-----------:|----------:|
| fp32_cnn | 94.5 | 85.0 |
| **binary_cnn** | **96.0** | **85.0** |

JSON note: production ASR → INT8 Whisper / ORT; classic BNN ASR is
research-grade. **Do not productize this row.**

### 4.7 C7 — CI matrix

[`.github/workflows/ci.yml`](../../.github/workflows/ci.yml):

- `windows`: install, best-effort MSVC `compile_native`, pytest, `export_check`, `repro_all --mode verify`.
- `linux-native` (**hard**): GCC OpenMP `.so`, `scripts/validate_native.py` (err = 0), pytest, export-check, repro verify.

That is the C7 sentence: Linux + Windows CI; Linux native `.so` validated in
Actions. Portability jobs (ARM64 / macOS) exist but are not extra claims.

---

## 5. Wrap demo as dual-metric illustration (still C1 + C3)

Not a new whitelist ID. Quality (cosine) and speed (e2e) and size (32× pack)
are three columns.

### 5.1 `wrap_demo` — AND-gate on **that** shape

Committed golden [`results/wrap_demo.json`](../../results/wrap_demo.json)
(PR #39; Sequential hidden=4096, replace layers `3` and `5`, batch=64,
`binary_xnor`, MSE STE QAT 200 steps, `fold_alpha=true`, **no `--force`**):

| Metric | Value | Kind |
|--------|------:|------|
| Weight compression (replaced) | **32.00×** | C1 theory |
| Output cosine vs FP | **0.999** | quality |
| E2E latency FP → wrapped | 50.24 ms → 18.94 ms (**2.65×**) | C3 wall-clock |
| GEMM-only vs torch Linear | **5.17×** | kernel ROI |
| `drop_in_ok` / `forced` / `status` | true / false / `OK` | honesty flags |

QAT JSON note: “Light STE only; production needs BitDistill-scale QAT on real
data.” Cosine is measured on the demo protocol batch, not ImageNet.

**Sentence this paper is allowed to say:** on this committed MLP wrap shape,
short QAT recovers drop-in cosine **and** keeps e2e **>1.5×**, while
compression remains the **32× pack ratio** — three numbers, not one.

Recipe (same shape; do not invent a bench):

```bash
python scripts/wrap_existing_demo.py --mode binary_xnor --hidden 4096 --batch 64 --qat-steps 200
```

Spike table: [`docs/spikes/WRAP_HYBRID_085.md`](../spikes/WRAP_HYBRID_085.md);
search notes: [`docs/42_QAT_AND_LAYER_SEARCH.md`](../42_QAT_AND_LAYER_SEARCH.md).

### 5.2 Ultra TinyBlock — AND-gate does **not** hold

[`results/ultra_wrap.json`](../../results/ultra_wrap.json) `primary`
(`hybrid_ffn` / `binary_xnor`, TinyBlock \(d=512\), \(ff=2048\), batch=64):

| Metric | Value |
|--------|------:|
| Compression (replaced) | 32.0 |
| Cosine | **0.699** |
| E2E speedup | **1.61×** |
| `drop_in_ok` | **false** |
| `status` | **`REFUSE_DROP_IN_CLAIM`** |

The optimiser already refuses a drop-in claim. **Do not write that the
0.85 ∧ 1.5× AND-gate holds for TinyBlock.** Cosine is the miss; e2e on the
committed snapshot already clears 1.5×.

### 5.3 Ternary does not count as the wrap win

Same file, `ternary_accurate_path`: cosine **0.991**, e2e **0.73×**
(`forced: true`). Meeting cosine while **losing** wall-clock fails dual-metric
accounting. Compression there is **16×** (2-bit ternary pack), not 32×.

---

## 6. Figures (from goldens, not invented)

Generate — do not hand-draw new benches:

```bat
bnn bridge figures --out results/figures_manifest.json --plot-dir results/figures
bnn pareto --from-results --out results/pareto_from_results.json --plot results/pareto_from_results.png
```

| Figure | File (after generate) | What it must show |
|--------|----------------------|-------------------|
| F1 Dual-metric kernel | `results/figures/dual_metric_speedup.png` | \(S_{\mathrm{compute}}\) vs \(S_{\mathrm{e2e}}\) vs a **32× theory** reference line that is *not* latency |
| F2 Canaries vs floors | `results/figures/canary_vs_floors.png` | C4–C6 recorded acc vs floor mins |
| F3 Wrap dual-metric | table in §5.1 (JSON) | 32× pack **beside** 2.65× e2e **beside** 0.999 cosine |
| F4 Fake-binary | table in §3.2 | `sign`+FP slower than torch FP32 |

PNGs are optional polish. The JSON + this skeleton are the citable record.
**No invented goldens.**

---

## 7. Related work (positioning, not a survey)

Full landscape: [`docs/02_SOTA_SURVEY.md`](../02_SOTA_SURVEY.md). This paper’s
wedge is **reporting**, not a new accuracy recipe.

| Prior art | What it contributes | Gap this skeleton fills |
|-----------|---------------------|-------------------------|
| BinaryNet / XNOR-Net / Bi-Real / ReActNet | Accuracy recipes, α-scales, FP shortcuts | Do not standardize fake-binary controls or dual metrics |
| BitNet b1.58 / bitnet.cpp | Ternary LLMs; honest-ish latency *and* energy tables | Arithmetic energy ≠ SoC; not a general BNN accounting protocol |
| Larq Compute Engine (MLSys) | Strong ARM wall-clock for TF/Keras BNNs | Larq archived 2026-06-15; we do **not** quote LCE FPS as ours |
| Classic Amdahl | General systems bound | Instantiate for packing + STE-sim vs packed kernels |

Do not claim we reproduced ReActNet ImageNet, bitnet.cpp tok/s, or LCE Pixel
FPS. GPU datacenter default remains INT4/FP8 (torchao / vLLM) — this lab
*bridges* there (`bnn bridge gpu`) instead of selling XNOR on Tensor Cores.

---

## 8. Reporting checklist (authors / reviewers / agents)

A BNN systems paragraph is complete only if it includes:

- [ ] Pack compression **and** a wall-clock \(S_{\mathrm{e2e}}\) (or an explicit
      “simulation only; no latency claim”).
- [ ] Fake-binary (`sign`+FP) negative control, or a statement that no packed
      kernel exists.
- [ ] Weights pre-packed once in the timed path.
- [ ] Shape listed in [`docs/BENCH_SHAPES.md`](../BENCH_SHAPES.md) or labeled
      **non-golden**.
- [ ] Quality metric (cosine / acc) **separate** from speedup.
- [ ] Hardware / native-vs-NumPy / thread disclosure.
- [ ] No GPU 32× from STE/`sign()`.

Agents: follow [`AGENTS.md`](../../AGENTS.md) / [`REPRODUCIBILITY.md`](../../REPRODUCIBILITY.md);
compare only to `tests/golden_floors.json` and committed `results/*.json`.

---

## 9. Limitations

- CPU / edge packed kernels. Commodity GPU → INT4/FP8 bridges, not binary 32×.
- Canaries (C4–C6) are floors, not leaderboard submissions.
- Wrap AND-gate is **shape-specific**: `wrap_demo` hidden=4096 yes; TinyBlock
  hybrid **REFUSE**.
- Energy is \(E=P\cdot t\) proxy unless a privileged Linux RAPL run exists
  (human / moonshot).
- QAT on `wrap_demo` is 200-step logit MSE on a synthetic batch — not
  BitDistill-scale KD on real data.
- Kernel speedups are host-dependent; floors use tolerances, not bit-identical
  floats.

---

## 10. Papers with Code — official code link

**Do not register a paper page until an arXiv ID exists (human).**

Until then, the official implementation is this repository:

| PwC field | Value |
|-----------|-------|
| Title | Stop Claiming 32×: Honest Speedup Accounting for Binary Neural Networks |
| Official code | <https://github.com/KanakMalpani/Binary-Neural-Networks> |
| Framework | PyTorch |
| Suggested tasks | Quantization, Model Compression |
| Suggested datasets | MNIST (C4); CIFAR-10 **proxy** (C5) |
| **Do not** add | ImageNet classification SOTA; ASR / LibriSpeech product numbers |
| Citation file | [`CITATION.cff`](../../CITATION.cff) (version **1.0.0**) |
| Reproduce | `pip install -e ".[dev]" -c constraints.txt` then `bnn repro` |

After arXiv (human):

1. Upload source (this markdown → optional LaTeX) to arXiv `cs.LG` and/or `cs.PF`.
2. Open <https://paperswithcode.com> → add paper by arXiv ID.
3. Add **official** code pointing at the GitHub repo (and optionally the
   `v1.0.0` tag / `bnn-lab==1.0.0` on PyPI).
4. Link only C1–C7 metrics; do not submit canary accuracies as ImageNet or ASR
   SOTA.

GitHub citation metadata already ships in `CITATION.cff`. A PwC badge belongs
in the root README **after** the paper page exists (README is out of scope for
this skeleton PR).

---

## 11. Human next steps (not this PR)

1. Optional LaTeX conversion (venue template); keep every table sourced from
   the same JSON.
2. Run `bnn bridge figures --plot-dir results/figures` and attach F1–F2 PNGs.
3. **Submit arXiv** (author account). Do not scrape-upload from CI.
4. Register Papers with Code official code (§10).
5. Optional workshop (MLSys / edge-AI). B2 / B3 remain companions
   ([`docs/32_NOVEL_PAPER_CANDIDATES.md`](../32_NOVEL_PAPER_CANDIDATES.md)).

---

## Appendix A — JSON field map

| Claim | Path | Fields |
|-------|------|--------|
| C1 | `results/wrap_demo.json` | `weight_compression_replaced_layers` |
| C1 | `results/benchmark.json` | `results[*].theoretical.weight_compression` |
| C2 | `results/benchmark.json` | `results[*].max_abs_error_vs_fp32` |
| C3 | `results/benchmark.json` | `speedup_compute_vs_numpy_fp32`, `speedup_e2e_vs_numpy_fp32`, `fake_binary_vs_torch_fp32` |
| C3 | `results/wrap_demo.json` | `e2e_speedup`, `e2e_latency_ms_*`, `output_cosine_vs_fp` |
| C4 | `results/train_results.json` | model `binary_mlp` `test_acc` |
| C5 | `results/image_cifar.json` | model `binary_cifar_bireal` `test_acc` |
| C6 | `results/audio_synth.json` | model `binary_cnn` `test_acc` |
| C7 | `.github/workflows/ci.yml` | jobs `windows`, `linux-native` |
| REFUSE (not a pass) | `results/ultra_wrap.json` | `primary.effectiveness.cosine`, `primary.status` |

## Appendix B — Suggested BibTeX (software; paper ID TBD)

```bibtex
@software{malpani2026bnnlab,
  author = {Malpani, Kanak},
  title  = {Binary Neural Networks --- Extreme Low-Bit Inference Lab / Optimiser},
  year   = {2026},
  version = {1.0.0},
  url    = {https://github.com/KanakMalpani/Binary-Neural-Networks},
  note   = {Does not claim GPU 32x from sign(). Tech-report skeleton:
            docs/papers/B1_STOP_CLAIMING_32X.md}
}
```

Replace with `@article` / `@misc` after arXiv assigns an identifier.
