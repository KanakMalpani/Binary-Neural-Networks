# Gap research notes

**Role:** Parallel enrichment for `knowledge_graph/` — do not rewrite `bnn_kg.json` if another agent owns it. Ingest via `gap_research_nodes.json` + `MERGE_INSTRUCTIONS.md`.

## What the lab already covers well

- Thesis lock: packed CPU XNOR, no fake GPU 32× from `sign()` (`AGENTS.md`, ADR).
- Survey-level names for Bi-Real / ReActNet / BitNet / Larq / FINN / AWQ / NPU INT8-first.
- Failure modes F1–F10 at architecture level (STE, BN, fake binary, Amdahl).
- Decision tree: GPU→INT4/FP8; CPU LLM→bitnet.cpp; vision edge→Bi-Real/Larq/FINN.

## Material gaps filled by this enrichment

1. **Precise ImageNet ladder** — docs said “~71% claimed” / “Bi-Real ~56%”; now node-level 56.4 / 65.4 / 69.4 / 71.4 with OPs and paper IDs.
2. **Real-to-Binary** — named in tables without arXiv:2003.11535 or 65.4% citation strength.
3. **bitnet.cpp claims** — ranges were correct but lacked device (M2 Ultra vs i7-13700H), thread mode, and J/token table footnotes. Distinguish **arithmetic** 71× (b1.58 paper, 7nm model) from **measured SoC** −55–82% (bitnet.cpp).
4. **BitDistill vs BitDistiller** — docs say “BitDistill” without ID; enrichment pins 2510.13998 and separates 2402.10631.
5. **BitNet a4.8 / Sparse-BitNet** — mentioned once; now full mechanisms + numbers.
6. **BiBERT** — GLUE/FLOPs claims + attention failure diagnosis missing from failure doc depth.
7. **FINN FPS/W** — FPS cited; chip vs wall power and FPS/W now explicit.
8. **AWQ/GPTQ vs binary** — decision tree existed; enrichment adds when-each-wins edge set + calib caveats.
9. **Mobile NPU** — already CLOSED-BY-PROXY in docs/20; mirrored as KG concept for merge.
10. **Failure modes** — attention binarization, small-K layers, PTQ→ternary scaled as first-class nodes.

## Confidence / caveats

- **Do not** treat FasterTransformer GPU latency from Ma et al. 2024 as bitnet.cpp CPU numbers.
- **Do not** treat 7nm arithmetic energy as wall-socket Joules.
- Sparse-BitNet 1.30× is vs dense BitNet on custom 6:8 kernels — not vs FP16 Transformer.
- Larq archived: pin commits before depending in CI.
- ImageNet full train remains lab ACCEPTED-NON-GOAL; enrichment is literature goldens only.

## Suggested human follow-ups (optional)

- Patch `docs/02_SOTA_SURVEY.md` citation block with the IDs in `citation.strengthen.lab_weak`.
- Add one sentence to `docs/03` pointing at BiBERT attention failure + BitDistill PTQ failure.
- Link HF `microsoft/bitnet-b1.58-2B-4T` from `docs/23`.

## 2026 literature overlay (2026-08-15)

- **ScaleQ-1.58** (`2608.01078`), **BitEmbed** (`2606.25674`), **VibeASR-BitNet** (`2607.21075`) are first-class `Paper` nodes with `status: literature`.
- **Litespark** (`2605.06485`) was already in the graph; keep `gap_litespark_local` **open** — do not invent local SIMD numbers.
- **FBI-LLM** stays an OpenGap at `accepted_non_goal`.
- Do not treat archival `enrichment_runs` notes that say Wave 1 PRs are still open as current after v1.0.0 + PyPI.

## Schema notes

Nodes: `id, label, type, summary, sources, confidence, status`  
Edges: `source, target, relation, evidence`  
`meta.researcher = "gap-parallel"`
