# Transformation plan — exponential, not polish

| Field | Value |
|-------|-------|
| **Status** | Proposal (does **not** flip `ROADMAP.md` / `docs/37` checkboxes). **2026-08-14:** Wave 0 / lever 1 shipped — [`bnn-lab` 1.0.0](https://pypi.org/project/bnn-lab/1.0.0/) is live (OIDC). |
| **Date** | 2026-08-13 |
| **HEAD at writing** | `bc4aa7e` (`v1.0.0-3-gbc4aa7e`) |
| **Tag** | `v1.0.0` (2026-08-04) |
| **Package** | `bnn-lab` (import/CLI `bnn`) |
| **Thesis lock** | Packed CPU/edge XNOR–popcount + honest STE; **32× is uint64 pack compression, not GPU from `sign()`**; no invented goldens |

Canonical product plan remains [`ROADMAP.md`](../ROADMAP.md). This file answers a different question: *what would 10× this lab as a public repo in 2026, from first principles, given the lab is already “complete”?*

Related: [`45_IMPROVEMENT_ROADMAP_HANDOFF.md`](45_IMPROVEMENT_ROADMAP_HANDOFF.md) (measured leftovers), [`MOONSHOT_DEFERRALS.md`](MOONSHOT_DEFERRALS.md), [`PUBLICATION_PLAN.md`](PUBLICATION_PLAN.md), [`knowledge_graph/VIEW.md`](../knowledge_graph/VIEW.md).

---

## A. Current state (1 screen)

Audit: **2026-08-13**. Score is honesty vs a *public category-leading repo*, not vs the lab’s own WC gates (those are largely green).

| Area | Score | State | Residual |
|------|------:|-------|----------|
| **Kernels** | **9/10** | Portable SIMD (AVX-512 → AVX2 → NEON → scalar), OpenMP, `err = 0` bit-identity, 4-row blocking. Aggregate **5.1×** vs prior kernel; **~24×** vs NumPy FP32 at 64×4096×4096. | NumPy packed path is **5–11× slower than BLAS** for B≳8 when **native does not load** ([docs/45](45_IMPROVEMENT_ROADMAP_HANDOFF.md) P1) — not the typical pip wheel (Win/mac wheels already ship native SIMD; `BNN_NO_OPENMP=1` is thread scaling, not missing kernels). |
| **Wrap / WC-O** | **7/10** | `bnn optimise` + schema v1, auto policy, BN fuse, distill, drop-in **REFUSE**. Default `--policy auto` lands **hybrid cosine ~0.70** + `REFUSE_DROP_IN`. Legacy `wrap_demo.json` binary_xnor cosine **0.31** (no QAT). Ternary+QAT cosine **0.991**, drop-in OK — but e2e **0.73×** (slower than FP). WC-O4 is **[x]** in ROADMAP (short QAT vs cold PTQ on the documented demo). | Gap is **binary/hybrid still below 0.85 drop-in** while staying fast — not “QAT is a sketch.” |
| **Codec** | **8/10** | `.bnnpack` v2 + hashes + safetensors. | ONNX = bridge-only (policy). No Hub collection of packs. |
| **CLI** | **8/10** | Rich (`optimise`, `repro`, `bridge`, `kg`, `energy-bound`, …). | Clone-first; `bnn/cli.py` ~1k lines (split is 1.1×). |
| **Docs** | **8/10** | `GUIDE_E2E`, tutorials 01–08, MkDocs autodoc `--strict`, dual-metric README. | **GitHub Pages 404**. README conversion is clone+MSVC. Issue **#1 is still open for a reason**: it wants an **above-the-fold** “When NOT to use BNN” callout under the thesis; the README only has an Is/is not table **at the bottom**. |
| **CI / OSS** | **7/10** | Win+Linux native, py3.11–3.13, portability, CodeQL, OpenSSF Scorecard, LICENSE, templates, Discussions, branch protection. | **0 stars / 0 forks**. 2 stale good-first issues. 7+ Dependabot PRs. No Pages. |
| **Research / KG** | **7/10** | 165 nodes / 288 edges, `validate PASS`, claims whitelist, B1–B3 vault. | KG still marks Wave 1 lanes as `open_pr` after merge #26. No venue submit. Survey last reviewed 2026-08-04 (misses ScaleQ-1.58, BitEmbed, VibeASR-BitNet). |
| **Moonshots** | **8/10** | WASM pedagogy, RAPL proxy, ImageNet *protocol* (no SOTA gate), bitnet.cpp pin (no submodule). | Privileged RAPL, ORT custom op, BitDistill-scale KD — correctly deferred. |
| **PyPI** | **8/10** | `bnn-lab` **1.0.0** on PyPI (OIDC Trusted Publisher, 2026-08-14). | Recurring releases; no Windows ARM64 / no `cp313-win_amd64` in 1.0.0. Name `bnn` taken by Adrian Bulat. |

**Headline:** this is a **world-class *lab*** (repro, kernels, honesty) that is **not yet a public product**. The WC bar in ROADMAP §1 is mostly met; the *exponential* gap is distribution, conversion, wrap quality, and category occupancy after **Larq archived 2026-06-15**.

### Remaining ROADMAP `[ ]` / `[~]` (honest)

| Item | Kind |
|------|------|
| W8.T08 / WC-R2–R4 | **Shipped 2026-08-14:** `bnn-lab` 1.0.0 on PyPI (OIDC) |
| v1.0 checklist rows in §8 | Stale vs tagged `v1.0.0` — tag exists; PyPI line now `[x]` |
| Ternary kernels / audio / ONNX / leaderboard `[~]` | Polish or deferred-by-policy, not blockers |
| Non-goals in §0.3 | Stay `[ ]` forever (GPU 32×, ImageNet SOTA gate, Whisper product, NPU 1-bit) |

### Inventory snapshot

- **Git:** `main` = `bc4aa7e` (3 commits after `v1.0.0` / Wave 2 integrator `49de25b`: packing 2.5–10×, handoff doc, popcount signedness test).
- **Release:** `v1.0.0` 2026-08-04, **no attached wheel assets** (wheels live in Actions, unpublished).
- **KG OpenGaps still `open`:** `gap_venue_submit`, `gap_reactnet_in_repo`, `gap_litespark_local`, `gap_fbi_llm_repro`. **`gap_pypi_trusted` merged** 2026-08-14. Several `open_pr` gaps are **stale** (WASM, bnnpack v2, distill, RAPL, layer search, bitnet submodule — shipped or closed-by-policy).

---

## B. First-principles bottlenecks

### What the actual product is

A **lab that proves packed binary/ternary GEMM + honest wrap** for CPU/edge:

1. uint64 pack compression is **exactly 32×** (size).
2. Inference speed comes from **XNOR–popcount kernels**, not `sign()` + `nn.Linear`.
3. STE trains latents; packed kernels infer.
4. Reports **dual metrics** and **refuses** drop-in when cosine is junk.
5. When BitNet/INT4/FP8/GGUF wins, **`bnn bridge` says so**.

It is **not** a fake-binary GPU story, not llama.cpp, not bitnet.cpp, not ImageNet SOTA.

### Rate-limiting constraints (why they dominate)

| # | Constraint | Why it dominates |
|---|------------|------------------|
| **1. Discoverability / install physics** | A stranger **can** `pip install bnn-lab==1.0.0` (library). Search for “binary neural networks pytorch” still hits archived Larq, Adrian Bulat’s `bnn` 0.1.2, and student MNIST repos — **not this lab**. 0 stars after a complete v1.0. | OSS “best in category” is a **funnel**. Indexable install is shipped; category occupancy and conversion remain the gap. |
| **2. Wrap quality vs speed (Amdahl + STE)** | Default `bnn optimise --policy auto`: **hybrid cosine ~0.70**, e2e modest, `REFUSE_DROP_IN`. Legacy `wrap_demo.json` binary_xnor: cosine **0.31** / **~4.8×** e2e (no QAT). Ternary+QAT: cosine **0.991**, e2e **0.73×**. | The 10× product gap is **hybrid/binary ≥0.85 cosine and still ≥1.5× e2e**. Auto already refuses honestly — that is not the missing 10×. Ternary already meets cosine and **loses** wall-clock. |
| **3. Memory bandwidth vs popcount throughput** | Large GEMMs are DRAM-bound; packing wins by shrinking the stream. Small GEMMs / Python loops / act-pack overhead eat Amdahl. NumPy packed path **loses to BLAS** for batched shapes **when native is absent**. | Physics: 32× fewer bytes only helps if the runtime **streams packed bits**. Typical pip (Win/mac wheels) already loads native SIMD. The 5–11× inversion is the **no-native-load** audience (failed/`BNN_FORCE_NUMPY`/exotic platform), not “most `pip install` users.” |
| **4. STE / architecture gap vs literature** | Lab CIFAR Bi-Real **61% vs FP 71%** (10 pp). Literature ImageNet ladder: BinaryNet 42% → ReActNet-A **69.4%**. RSign/RPReLU is documented, not default (`gap_reactnet_in_repo`). | Training recipe, not kernel, sets whether wrap/train is a toy. Closing 10 pp on the **canary** is allowed; ImageNet SOTA as a **gate** is not. |
| **5. OSS trust / conversion** | README’s first action is `git clone` + `compile_native`. MkDocs builds in CI; **Pages not deployed**. HF Space: none. Issue #1 still needs the above-the-fold callout (not a paperwork close). KG `open_pr` after merge. | llama.cpp / bitnet.cpp / transformers win on **60-second success**. This lab currently onboards like a research archive. |
| **6. Category confusion (BitNet era)** | 2026 mindshare is **1.58-bit LLMs** (bitnet.cpp **~40k★**, 2B4T, BitEmbed, ScaleQ-1.58 PTQ, Litespark SIMD). Classic CNN BNN tooling (**Larq archived**) is vacant. | Competing with bitnet.cpp on LLM tok/s is suicide. Occupying **PyTorch packed BNN optimiser + honest routing** is the wedge. |

### Invert: what world-class looks like in 2026

| Class | DX bar this lab should steal (not copy the product) |
|-------|------------------------------------------------------|
| **llama.cpp** | One install → tokens/sec. GGUF on the Hub. Hardware matrix. |
| **bitnet.cpp** | Official kernels + HF artifacts + “when this is the tool”. |
| **transformers / bitsandbytes / torchao / AWQ** | `pip install` + 5-line snippet + Hub integration + docs site. |
| **Starship** | Instant first prompt; no tribal knowledge. |
| **Larq (while alive)** | JOSS paper, zoo, compute engine, `pip install larq`. |

**Target shape:** `pip install bnn-lab` → 60s dual-metric report → Hub `.bnnpack` → Pages docs → paper from goldens → bridge to bitnet.cpp/INT4 when they win.

---

## C. Exponential levers (top 10, ranked)

Each item: **what / why 10× not 1.1× / first principles / evidence / effort / thesis risk / next PR**.

**Kind (rank order unchanged):** **funnel 10×** = adoption/discoverability (levers **1–3, 6–8**) — not a measured kernel/wrap ratio. **Measured 10×** = wall-clock or cosine on committed shapes (levers **4, 5**). Lever 9 is agent-memory; lever 10 is bounded/bridge.

### 1. Ship `bnn-lab` on PyPI (Trusted Publisher) — *funnel 10×* — **SHIPPED 2026-08-14**

- **What:** Pending publisher for `bnn-lab` / `wheels.yml` / env `pypi`; dispatch `publish=true` on **`main`**; clean-venv `pip install bnn-lab==1.0.0` + `import bnn`. (`bnn repro` remains clone + `[dev]`.)
- **Why 10×:** Converts the lab from “clone a 3-week-old repo” to **the installable PyTorch BNN toolkit** the week Larq is archived. Zero → indexable on PyPI, Cursor, pip, HF snippets.
- **First principles:** Distribution is the scarce resource, not another SIMD path.
- **Evidence:** `docs/PYPI_PUBLISH.md`; KG `gap_pypi_trusted` **merged**; PyPI name `bnn` taken; run [31825286443](https://github.com/KanakMalpani/Binary-Neural-Networks/actions/runs/31825286443).
- **Effort:** **S** (human, ~30 min) + **S** post-upload README (`pip install bnn-lab` first).
- **Thesis risk:** **None** if dual-metric README stays.
- **Next PR:** This docs/ROADMAP/KG flip (W8.T08 `[x]`). Recurring releases stay OIDC-only.

### 2. Landing conversion: 60-second dual-metric demo, not clone+MSVC — *funnel 10×*

- **What:** README above-the-fold = one-liner install + one command that prints **compression 32×, cosine, wall-clock, REFUSE/OK**. Move thesis mermaid down. **Implement issue #1** as a short **above-the-fold** “When NOT to use BNN” callout **under the thesis** (GPU/INT4/bitnet.cpp/NPU INT8) — the bottom Is/is not table does **not** satisfy #1; do **not** paperwork-close it. Then close #1, or rewrite the issue if the callout is rejected. Deploy **GitHub Pages** from existing MkDocs CI artifact.
- **Why 10×:** llama.cpp/HF conversion is “first screen success.” Current first screen is a research manifesto. Pages 404 wastes a `--strict` docs job.
- **First principles:** Attention is bandwidth-limited; the README is the only kernel most visitors run.
- **Evidence:** Exa fetch of GitHub README; `gh api .../pages` 404; issue #1 still open because the callout is missing above the fold, not because nobody wrote “when not to use” anywhere.
- **Effort:** **M**.
- **Thesis risk:** **Low** — do not drop dual-metric warnings to look punchier.
- **Next PR:** `docs(W9): pip-first README + above-the-fold When-NOT callout (#1) + Pages workflow`.

### 3. One killer demo (HF Space): the wrap paradox, visualized — *funnel 10×*

- **What:** A Space (KanakMalpani) that runs `bnn optimise` on a **tiny public** MLP/CNN: three columns — FP32, binary packed, ternary+QAT — showing **size / cosine / latency** and the REFUSE badge. Not ImageNet. Not ASR. Label default auto (~0.70 hybrid, REFUSE) separately from legacy `wrap_demo.json` 0.31.
- **Why 10×:** bitnet.cpp has an Azure demo; transformers has Spaces. A 0-star repo with no try-before-clone cannot enter the category. The *unique* demo is honesty (binary/hybrid fast-ish + below drop-in vs ternary accurate + slower), which no fake-32× repo will ship.
- **First principles:** Product = decision under constraints. Show the Pareto, don’t hide it.
- **Evidence:** `results/ultra_wrap.json` hybrid cosine ~0.70 / `drop_in_ok: false`; ternary 0.991 cosine / 0.73× e2e; `results/wrap_demo.json` 0.31 / 4.82× is the **legacy binary_xnor, no-QAT** snapshot; HF Spaces search for `bnn-lab` empty.
- **Effort:** **L** (CPU Space, no GPU claim).
- **Thesis risk:** **Medium** if the Space implies drop-in; mitigate with the same schema flags.
- **Next PR:** `feat(demo): Gradio Space from committed wrap/ultra_wrap shapes only`.

### 4. Wrap accuracy leap: hybrid/binary ≥0.85 **and** e2e ≥1.5× — *measured 10×*

- **What:** One public recipe on **committed** `wrap_demo` / `ultra_wrap` shapes (not a new golden): **binary or hybrid** wrap + short QAT/distill reaches cosine **≥0.85 and** e2e **≥1.5×** vs FP, **without `--force`**. Ternary already has cosine **0.991** and e2e **0.73×** — that does **not** satisfy this lever.
- **1.1× fallback (not this lever, not Wave 3 exit):** make `policy=auto` never first-run a `REFUSE_DROP_IN` path (skip/ternary when hybrid would refuse). Default auto **already** lands hybrid cosine **~0.70** + REFUSE — so “auto never first-runs 0.3 cosine binary wrap” is **already true** (the 0.31 figure is legacy `wrap_demo.json`). Counting that OR as Wave 3 would ship a no-op.
- **If targeting `wrap_demo`:** `tests/golden_floors.json` has `wrap_demo.cosine_max_without_qat: 0.5` (low cosine without QAT is **expected**). A real win **updates that floor on the same shape** — do not invent a new bench.
- **Why 10×:** This is the product. A kernel that is 24× on a microbench is irrelevant if hybrid wrap cannot be both drop-in **and** faster. Crossing **both** gates on one honest demo changes “lab” → “tool.”
- **First principles:** STE mismatch + absmean PTQ wipe (`paper_bitdistill` vs `method_absmean_ptq`). BitDistill-scale KD is a moonshot; a **short, reproducible QAT** on the existing demo is the lever. Literature: ReActNet/Bi-Real recover accuracy via **architecture + distill**, not magically via `sign()`.
- **Evidence:** WC-O4 is **[x]** in ROADMAP (short QAT improves cosine vs cold PTQ on the documented demo) — not a sketch. The residual is **binary/hybrid still below 0.85** (`ultra_wrap` primary hybrid cosine ~0.70, `drop_in_ok: false`, e2e already **~1.61×** on that snapshot — so the missing AND is **cosine**, not speed, unless QAT eats the 1.5×). BitNet Distillation arXiv:2510.13998; ScaleQ-1.58 arXiv:2608.01078 — **do not claim** we reproduce them.
- **Effort:** **L**.
- **Thesis risk:** **High** if someone “fixes” cosine by changing golden **shapes** or claiming LLM chat quality. Stay on committed wrap_demo / ultra_wrap / CIFAR canary. Updating `cosine_max_without_qat` after a measured QAT win on the **same** shape is allowed.
- **Next PR:** `feat(W3): hybrid/binary QAT on wrap_demo/ultra_wrap that meets 0.85 cosine and 1.5× e2e without --force` + docs/42. No new benches.

### 5. Honest NumPy fallback: never slower than “doing nothing” — *measured 10×*

- **What:** When **native does not load**, dispatch packed NumPy vs dequant+BLAS by shape (docs/45 P1). Keep `binary_gemm_numpy_prepacked` as the **correctness reference**. README: *correct* ≠ *fast*.
- **Why 10× (for that audience):** Without a native library, batched packed NumPy is **5–11× slower than FP32 BLAS**. That inverts the thesis for the no-native-load path. It is **not** the typical `pip install` on Win/mac — those wheels already ship native SIMD (`BNN_NO_OPENMP=1` does not drop you onto NumPy).
- **First principles:** Bandwidth win requires a packed *or* BLAS-fast path; a Python loop over B is neither.
- **Evidence:** Measured table in docs/45; crossover B≈8–16 at 4096; wheel matrix in `docs/PYPI_PUBLISH.md`.
- **Effort:** **M**.
- **Thesis risk:** **Low** if `err = 0` both ways and compression of stored weights is unchanged.
- **Next PR:** `perf(kernels): BLAS fallback when NumPy packed loses` + test at B=64.

### 6. Occupy the Larq vacuum, explicitly — *funnel 10×*

- **What:** Positioning sentence: *PyTorch packed BNN optimiser now that Larq (TF/Keras) is archived (2026-06-15).* Comparison table: Larq / Brevitas / bitnet.cpp / torchao / this lab. Do **not** claim LCE FPS.
- **Why 10×:** Category leadership is **who inherits the search query**. 732★ Larq is read-only; LCE last release 2024. PyTorch users have no default BNN toolkit with packed kernels + honesty.
- **First principles:** Markets have one default. Vacancy is a larger delta than another tutorial.
- **Evidence:** larq/larq archived; docs/02 already notes it; Tavily/Exa did not surface this GitHub repo for generic BNN queries.
- **Effort:** **S–M** (README + `docs/02` row + maybe a blog/Show HN).
- **Thesis risk:** **Low** if we don’t claim Larq Zoo ImageNet numbers as ours.
- **Next PR:** `docs: Larq-archive positioning + competitor table`.

### 7. B1 tech report from goldens + Papers with Code — *funnel 10×*

- **What:** Ship **B1 — Stop claiming 32×** as an arXiv tech report **only** from `results/*.json` + claims whitelist (`docs/PUBLICATION_PLAN.md`). Register the repo on Papers with Code / CITATION.cff already exists. B2/B3 as companions later.
- **Why 10×:** Academic and HN funnels are paper-shaped. A citable “honest speedup accounting” paper is the unique research wedge (not another XNOR-Net survey). Citations compound; more CIFAR epochs do not.
- **First principles:** The scarce claim is *measurement culture*, which the lab already has.
- **Evidence:** `paper_b1_honest_speedup blocked_by gap_venue_submit`; fake-binary ~1.4× slower in committed benches; dual-metric schema.
- **Effort:** **L** (author time; LaTeX outside this repo OK).
- **Thesis risk:** **High** if the paper advertises 32× latency. Whitelist C1–C7 only.
- **Next PR:** `docs(W12): B1 preprint skeleton + PwC code link` (no invented figures).

### 8. Hub artifacts: `.bnnpack` + tiny zoo on Hugging Face — *funnel 10×*

- **What:** Upload 1–3 **tiny** packed artifacts (MNIST MLP, CIFAR Bi-Real canary weights if license-clean, wrap-demo pack) with model cards that quote floors, not SOTA. `from_pretrained`-style load in tutorial 08.
- **Why 10×:** llama.cpp won because GGUF is a **noun** on the Hub. `.bnnpack` is a format without a public object. Formats without objects don’t get copied.
- **First principles:** A codec is a product only if strangers can download a file.
- **Evidence:** W5.T03–T06 done in-tree; no HF collection; bitnet.cpp ships `BitNet-b1.58-2B-4T-gguf`.
- **Effort:** **M**.
- **Thesis risk:** **Low** if cards say canary, not ImageNet.
- **Next PR:** `feat(W5): HF collection + bnnpack model card`.

### 9. KG freshness + agent-facing honesty (compounding for AI users)

- **What:** Flip stale `open_pr` → `merged` / `closed_by_policy`; add 2026 nodes (ScaleQ-1.58 `2608.01078`, BitEmbed `2606.25674`, VibeASR-BitNet `2607.21075`, Litespark `2605.06485`) as **literature-only**. CI already validates structure; add “status vs ROADMAP” drift test.
- **Why ~5–10× for agents, 1.1× for humans:** This repo markets itself to coding agents (`AGENTS.md`). A graph that says Wave 1 is still open **after v1.0.0** trains agents to reimplement shipped work (R9).
- **First principles:** The KG is the lab’s memory; stale memory is a silent failure.
- **Evidence:** `bnn kg` 165/288 PASS; meta `lab_coverage_note` still “Wave 1 lanes A–I remain open PRs”; VIEW.md §8 still lists distill / bnnpack v2 as v1 leftovers.
- **Effort:** **M**.
- **Thesis risk:** **None** if unreproduced Litespark numbers stay `OpenGap`.
- **Next PR:** `chore(kg): post-v1.0.0 status + 2026 literature overlay`.

### 10. Kernel leap that stays thesis-honest (bounded)

- **What (pick one, not all):** (a) **Ternary 4-row blocking** — docs/45 P2, only ~1.1–1.5× on small B; (b) **optional T-MAC/LUT path** for BitNet-shaped ternary GEMMs, dual-metric vs current bitplane kernel, **no Litespark number copying**; (c) a **scripted bitnet.cpp 2B4T run** via `bnn bridge` that prints tok/s from *their* kernels.
- **Why it can 10× *a user metric*:** (c) is the honest 10× for “CPU 1-bit LLM” — because the physics lives in bitnet.cpp. (a) is 1.1×. (b) is research, high variance.
- **First principles:** Don’t spend kernel years on a 1.1× when the LLM user should be routed. Do spend kernel years on **this lab’s** wide binary GEMM remaining competitive with BLAS on *published shapes*.
- **Evidence:** bitnet.cpp Jan 2026 +1.15–2.1×; Litespark claims 18×/96× vs *naive PyTorch* (different baseline — **gap_litespark_local**); lab 64×4096 already ~24× vs NumPy FP32.
- **Effort:** (a) **M**, (b) **XL**, (c) **M**.
- **Thesis risk:** **High** for (b) if README quotes unreproduced 96×. **None** for (c) if clearly a bridge.
- **Next PR:** Prefer **(c)** `docs(W4): bitnet.cpp 2B4T smoke recipe` then **(a)** only if a user needs small-batch ternary.

---

## D. Explicit non-goals / anti-levers

Impressive-looking work that **violates the thesis** or **does not compound**:

| Anti-lever | Why not |
|------------|---------|
| GPU 32× from `sign()` / STE | Forbidden forever. |
| Invented golden shapes / Litespark local benches copied from the paper | R9; `gap_litespark_local` stays literature. |
| Full ImageNet SOTA as a **gate** | Non-goal. Protocol runner already exists. |
| Production Whisper / ASR | Audio lane is synthetic; BitNet-ASR papers are **their** stack (`2607.21075`). |
| Stock phone NPU native 1-bit | Vendors ship INT8/INT4 (`docs/20`). |
| Full BitNet pretrain / FBI-LLM repro in this repo | Bridge; `gap_fbi_llm_repro`. |
| ORT custom op | Closed-by-policy; revisit only with demand. |
| Memory arena | Measured 1.4–1.8% + aliasing (`docs/43`). |
| `ruff format` whole-repo / CLI split / more tutorials 09–20 | 1.1× maintainability, merge pain. |
| Competing with bitnet.cpp on tok/s using this GEMM | Wrong product; route. |
| Dependabot firehose as the roadmap | Hygiene, not 10×. |
| Paperwork-close of issue #1 | Bottom “When not to use” table ≠ above-the-fold callout under the thesis. Implement or rewrite; don’t close as completed. |
| Counting ternary 0.991 / 0.73× e2e, or auto-REFUSE, as Wave 3 | Ternary already meets cosine and loses speed; auto already refuses. The 10× is hybrid/binary **0.85 and 1.5×**. |
| WASM as a native-kernel substitute | Pedagogy only. |

**Moonshots allowed as labeled moonshots (not gates):** LUT ternary ASIC/T-MAC research, privileged RAPL Joules, community leaderboard submissions, ReActNet-A ImageNet *reproduction* if someone brings the schedule and hardware.

---

## E. 90-day sequence (waves, compounding)

Do **not** open 20 parallel lanes. Wave 2 already proved integration cost. Prefer: **PyPI → landing → killer demo → wrap accuracy → kernel honesty**.

```mermaid
flowchart LR
  W0[Wave 0 PyPI human]
  W1[Wave 1 landing + Pages + issues + KG status]
  W2[Wave 2 HF Space demo]
  W3[Wave 3 hybrid 0.85 AND 1.5x]
  W4[Wave 4 NumPy BLAS fallback]
  W5[Wave 5 B1 preprint]
  W6[Wave 6 Hub packs + Show HN]
  W0 --> W1 --> W2
  W1 --> W4
  W2 --> W3
  W3 --> W5
  W5 --> W6
```

| Wave | Days | Owner | Exit | Depends |
|------|------|-------|------|---------|
| **0** | 0–3 | **Human** | `pip install bnn-lab==1.0.0` + `import bnn` on clean venv; PyPI JSON 200 | **Shipped 2026-08-14** |
| **1** | 1–14 | Agent | Pip-first README; Pages live; **issue #1 implemented** as an above-the-fold “When NOT to use BNN” callout under the thesis (then close), **or** rewrite #1 — not a paperwork close of the bottom table; KG `open_pr` drift fixed | Wave 0 preferred, can draft README anyway |
| **2** | 7–28 | Agent | HF Space shows wrap paradox on **existing** shapes (label auto ~0.70 REFUSE vs legacy wrap_demo 0.31) | Wave 0 (install story) |
| **3** | 14–45 | Agent | **Same as lever 4 (AND, not OR):** hybrid/binary cosine **≥0.85 and** e2e **≥1.5×** on committed `wrap_demo` / `ultra_wrap` shapes, without `--force`. Ternary 0.991 / 0.73× does **not** count. Auto-never-first-run-REFUSE is a **1.1×** side quest (already nearly true today) — **not** this exit. If `wrap_demo` is the target, update `golden_floors.json` `cosine_max_without_qat: 0.5` on **that same shape**. | Wave 2 (demo must match recipe) |
| **4** | 21–45 | Agent | When native is **absent**, NumPy fallback never 5× slower than BLAS at B=64 (docs/45 P1). Typical Win/mac pip wheels already have native SIMD. | Independent of 3 |
| **5** | 30–75 | Author | B1 arXiv from goldens; PwC code link | Waves 1–2 (public artifact) |
| **6** | 45–90 | Mixed | HF `.bnnpack` collection; Show HN / r/MachineLearning with **honest** title; Larq-vacuum positioning | Waves 0–2 |

**Optional after day 60 (not on the critical path):** ReActNet RSign/RPReLU in `bnn.ste` as a CIFAR canary improvement (`gap_reactnet_in_repo`); ternary row-blocking (P2); bitnet.cpp 2B4T bridge smoke.

**Success metric for “best public repo in category” (90 days), not stars-as-vanity:**

1. `pip install bnn-lab` works.
2. A stranger gets a dual-metric report in <5 minutes without MSVC.
3. One Hub or Space artifact exists.
4. Hybrid/binary wrap on a committed shape is **drop-in (≥0.85) and faster (≥1.5× e2e)** — honest skip/REFUSE is already shipped and is **not** this bar.
5. Paper or tech report cites committed goldens only.
6. Search “pytorch binary neural network packed” can find this repo.

Stars follow those; they are not the input.

---

## Lab vs literature lag (KG + 2026 papers)

**This lab is ahead on:** dual-metric culture, fake-binary negative control, `err = 0` multi-ISA kernels, repro gates, wrap REFUSE, bridges CLI, energy-proxy honesty, agent-oriented docs.

**This lab is behind / not claiming:**

| Topic | SOTA (literature) | Lab stance |
|-------|-------------------|------------|
| 1.58-bit LLMs | BitNet b1.58, 2B4T, bitnet.cpp GPU+CPU, BitEmbed, VibeASR-BitNet | **Bridge only** |
| Ternary PTQ of existing LLMs | ScaleQ-1.58 / AYOT (Aug 2026) | Not claimed; absmean PTQ wipe documented |
| Extreme SIMD ternary | Litespark 18–96× vs naive PT | **OpenGap** — do not invent |
| Vision 1-bit ImageNet | ReActNet ~69–71% | CIFAR canary 61%; RSign not default |
| Training STE | SURGE (ICML 2026), ApproxSign, EDE | Clipped STE default; math compare JSON |
| Classic BNN DX | Larq archived | **Vacancy to occupy** |
| GPU datacenter | AWQ / GPTQ / torchao / FP8 | Bridge |

---

## Proposal vs ROADMAP

- Do **not** treat this file as a new WC gate.
- Do **not** invent benches or flip §10 boxes here.
- When a wave ships, update ROADMAP twins **in that PR** (W8.T08, KG, docs).
- If a wave conflicts with a WC gate, **WC gate wins**.
