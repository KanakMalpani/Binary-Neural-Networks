# BNN Knowledge Graph — Human View

Navigation companion to [`bnn_kg.json`](bnn_kg.json). Diagrams are Mermaid; numbers
come from committed `results/*.json` and cited papers — not invented.

---

## 1. Thesis map

```mermaid
flowchart TB
  TL[Thesis Lock<br/>packed CPU/edge XNOR + honest STE]
  DM[Dual-metric culture<br/>32× compression ≠ latency]
  XK[Packed XNOR GEMM]
  STE[STE training latents]
  FB[Fake sign + FP GEMM]
  GPU[Forbidden: GPU 32× from sign]

  TL --> DM
  TL --> XK
  TL --> STE
  TL -.->|contradicts| FB
  TL --> GPU
  FB -.->|alternative_to| XK
  STE -.->|≠| XK
```

---

## 2. Speedup accounting (op-count vs wall-clock)

```mermaid
flowchart LR
  subgraph Theory
    C32[Weight compression 32× exact]
    W64[~64× word-op reduction]
  end
  subgraph Measured
    SC[S_compute prepacked kernel]
    SE[S_e2e whole forward]
    AM[Amdahl S_e2e = 1/((1-f)+f/Sk)]
  end
  C32 -.->|do not claim as| SE
  W64 -.->|contradicts| SE
  SC --> AM --> SE
  FB2[Fake binary slower than torch FP<br/>benchmark.json ~1.4×] -.->|negative control| SC
```

**Lab anchors (CPU, committed):**

| Shape | S vs NumPy FP32 | Notes |
|-------|----------------:|-------|
| 128×2048×2048 | ~12× | err = 0 |
| 64×4096×4096 | ~24× | OpenMP thread curve recorded |
| 32×8192×8192 | ~29× | theory word red. still 64× |

Wrap demo e2e: FP ~34 ms → wrapped ~7 ms (**~4.82×**), compression still **32×**.

---

## 3. Wrap / deploy decision tree

```mermaid
flowchart TD
  START[Primary goal?]
  START --> GPU[Maximize quality on NVIDIA GPU]
  START --> CPU[Local CPU LLM chat]
  START --> EDGE[Edge vision camera]
  START --> NPU[Phone / NPU]
  START --> CONV[Convert HF LLM to 1.58-bit]
  START --> RES[Research / teach XNOR]
  START --> GEN[Diffusion / high-fidelity]

  GPU --> FP8[BF16/FP8 train → FP8 or AWQ-INT4<br/>vLLM / SGLang / TensorRT]
  CPU --> BN{BitNet checkpoint?}
  BN -->|yes| BCPP[bitnet.cpp]
  BN -->|no| GGUF[GGUF Q4_K_M llama.cpp<br/>or torchao CPU]
  EDGE --> RT{Can retrain?}
  RT -->|yes| BR[Bi-Real / ReActNet → LCE or FINN]
  RT -->|no| I8[INT8 TFLite / OpenVINO / ORT]
  NPU --> INT8N[Stock SDK → INT8/INT4<br/>NOT native 1-bit]
  CONV --> BD[BitDistill / gradual-λ QAT<br/>NOT absmean PTQ alone]
  RES --> LAB[This repo: bnn optimise + packed GEMM]
  GEN --> AVOID[INT8/FP8 weight PTQ; avoid full BNN]
```

---

## 4. Training / STE cluster

```mermaid
flowchart TB
  SIGN[Forward: sign / RSign]
  STE[Clipped STE — default]
  APX[ApproxSign — Bi-Real]
  EDE[IR-Net EDE / tanh_soft]
  RS[RSign/RPReLU — ReActNet<br/>documented, not default]
  SUR[SURGE learnable dual-path<br/>literature]
  CMP[results/math_ste_compare.json<br/>ApproxSign closer to sharp teacher]

  SIGN --> STE
  SIGN --> APX
  SIGN --> EDE
  SIGN --> RS
  STE --> CMP
  APX --> CMP
  EDE --> CMP
  STE -.->|mismatch risk| SUR
```

**Rule:** STE trains; packed kernels infer. Training throughput is not a BNN win.

---

## 5. Hardware map

```mermaid
flowchart LR
  subgraph Wins_binary_ternary
    CPU[x86 POPCNT / portable SIMD]
    ARM[ARM NEON + LCE / bitnet.cpp]
    FPGA[FPGA FINN dataflow]
  end
  subgraph Prefer_other
    NV[NVIDIA Tensor Cores<br/>FP8 / INT4 / AWQ]
    NPU[Phone NPU INT8-first]
  end
  LAB[bnn-lab kernels] --> CPU
  LAB --> ARM
  LCE[Larq CE] --> ARM
  FINN[Brevitas→FINN] --> FPGA
  TORCHAO[torchao / vLLM] --> NV
  QNN[QNN / CoreML / Ethos] --> NPU
```

---

## 5b. ImageNet 1-bit accuracy ladder (enrichment)

Canonical W1A1 top-1 ladder (literature — **not** lab goldens):

```mermaid
flowchart LR
  BN[BinaryNet 42.2%] --> XN[XNOR-Net 51.2%]
  XN --> BR[Bi-Real-18 56.4%]
  BR --> R2B[Real-to-Binary 65.4%]
  R2B --> RA[ReActNet-A 69.4%]
  RA --> RC[ReActNet-C 71.4%]
```

Node: `metric.imagenet.bnn_ladder`. Lab CIFAR Bi-Real (~61% vs FP ~71%) is a **canary**, not this ladder.

---

## 6. Modality map (lab canaries)

```mermaid
flowchart TB
  LAB[bnn-lab]
  LAB --> MNIST[MNIST<br/>bin ~96.4% vs FP ~97.7%]
  LAB --> CIFAR[CIFAR-10 Bi-Real<br/>61% vs FP 71% — 10pp]
  LAB --> AUD[Audio synth tones<br/>canary NOT ASR]
  LAB --> SEQ[Seq2seq encoder/decoder<br/>pedagogy]
  LAB --> WRAP[Wrap / ultra / optimise<br/>.bnnpack]
  CIFAR -.->|non-goal| IN[Full ImageNet SOTA]
  AUD -.->|use instead| WH[INT8 Whisper / ORT]
```

---

## 7. Novel paper candidates (local vault)

```mermaid
flowchart LR
  LAB[Lab evidence]
  LAB --> B1[B1 Stop Claiming 32×<br/>dual metrics + fake binary]
  LAB --> B2[B2 Packed XNOR productization<br/>goldens + repro contract]
  LAB --> B3[B3 When Not to Binarize<br/>decision tree + hybrid FFN]
  B1 --> VAULT["Maintainer-local idea vault"]
  B2 --> VAULT
  B3 --> VAULT
  B1 -.->|blocked_by| VENUE[Venue LaTeX OpenGap]
```

Idea-vault folders under `C:\00 Research Papers\` are **maintainer-local** — cite
[`docs/32_NOVEL_PAPER_CANDIDATES.md`](../docs/32_NOVEL_PAPER_CANDIDATES.md) in-repo.

---

## 8. Roadmap: v0.3.0 vs v1.0 leftovers

```mermaid
flowchart TB
  V03[v0.3.0 lab<br/>portable SIMD, optimise API, repro]
  V10[World-class v1.0 bar<br/>ROADMAP WC-* gates]
  V03 --> V10
  V10 -.->|blocked_by| G1[PyPI Trusted Publisher first upload]
  V10 -.->|shipped| G2[Distill integration W3.T08]
  V10 -.->|shipped| G3[.bnnpack v2 + safetensors]
  V10 -.->|pedagogy shipped| G4[WASM SIMD]
  V10 -.->|proxy OK| G5[Windows RAPL / board Joules]
  V10 -.->|accepted non-goal| G6[Full ImageNet SOTA schedule]
```

`gap_pypi_trusted` is the remaining human blocker. Distill, `.bnnpack` v2, WASM pedagogy, layer search, and bitnet.cpp pin are **merged / closed-by-policy** — do not re-open them from stale `open_pr` fields.

---

## 9. Contradiction cheatsheet

| Claim A | Claim B | Relation |
|---------|---------|----------|
| Compression **32×** | E2E latency **32×** | `contradicts` — report both |
| Theoretical **~64×** word ops | Measured `S_e2e` | `contradicts` — Amdahl |
| `sign()` simulation | Packed kernel speedup | `alternative_to` / fake binary |
| Absmean PTQ ternary LLM | BitDistill / CPT | `contradicts` quality path |
| Stock phone NPU 1-bit | Vendor INT8-first | `contradicts` drop-in hope |

---

## 10. Quick ID index (high-degree hubs)

| id | Why it matters |
|----|----------------|
| `thesis_lock` | Immutable north star |
| `dual_metric_culture` | How to report numbers |
| `algo_xnor_gemm` | Real speed path |
| `fake_binary_sign` | Negative control |
| `decision_wrap_tree` | Practitioner routing |
| `sys_recommend_stack` | `bnn recommend` CLI |
| `sys_eval_suite` | `bnn eval-suite` / fair shapes |
| `decision_wc_o_gates` | WC-O1–O4 (established on main) |
| `sys_kg` | Graph + CI integrity |
| `paper_bitnet_b158` | Ternary LLM era pivot |
| `paper_gptq` / `paper_bitdistiller` | INT4 distill disambiguation |
| `sys_repro_gates` | `bnn repro` / goldens |
| `paper_b1_honest_speedup` | Novel paper B1 |

---

## 11. CLI companions

```bash
bnn kg                  # counts + open gaps
bnn kg validate         # structural PASS/FAIL
bnn recommend --goal cpu-llm
bnn eval-suite --skip-pytest
```

See [`docs/44_KNOWLEDGE_GRAPH.md`](../docs/44_KNOWLEDGE_GRAPH.md) and [`docs/lanes/kg.md`](../docs/lanes/kg.md).
