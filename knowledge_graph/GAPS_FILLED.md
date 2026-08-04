# Gaps found → research done → graph updates

This log records where the corpus was thin, what was researched (academia MCP /
Exa / Tavily / HF-aware literature), and which nodes/edges were added.
**No lab metrics were invented** — unreproduced numbers stay literature-grade
or `OpenGap`.

| Gap ID | Missing / thin | Research action | Added nodes | Added edges (examples) | Confidence |
|--------|----------------|-----------------|-------------|------------------------|------------|
| G-REACT | ReActNet details beyond survey blurb | Exa + arXiv `2003.03488` | `paper_reactnet`, `algo_rsign_rprelu`, `gap_reactnet_in_repo` | improves←`paper_bireal`; blocked_by→gap | 1.0 paper / 0.8 gap |
| G-IRNET | IR-Net / EDE citation incomplete | Docs cross-check + arXiv `1909.10788` | `paper_irnet`, `algo_irnet_ede` | introduces STE schedule; measured_on `math_ste_compare` | 0.95 |
| G-BIBERT | Transformer binarization node | arXiv `2203.06390` | `paper_bibert` | cites classic BNN line | 0.95 |
| G-LCE | Larq Compute Engine primary cite | arXiv `2011.09398` | `paper_larq_ce`, `tool_larq`, `org_plumerai_larq` | measured_on ARM; implements tool | 0.95 |
| G-BITNET-LINE | 2024–2026 BitNet family incomplete | academia `arxiv_search` + Tavily | `paper_bitnet`, `paper_bitnet_b158`, `paper_bitnet_cpp`, `paper_bitnet_a48`, `paper_bitnet_2b4t`, `paper_bitnet_v2`, `paper_bitdistill` | improves / requires / mitigates PTQ wipe | 0.9–1.0 |
| G-SPARSE | Sparse-BitNet mentioned in survey only | Tavily → arXiv `2603.05168` | `paper_sparse_bitnet`, `gap_litespark_local` | improves b1.58; blocked_by local repro gap | 0.7 |
| G-LITESPARK | Extreme SIMD claims | Survey arXiv `2605.06485` | `paper_litespark` → `gap_litespark_local` | **no local speedups claimed** | 0.65 |
| G-BITDISTILL | “Need distill” without paper ID | arXiv `2510.13998` (BitNet Distillation) | `paper_bitdistill`, `method_absmean_ptq`, `dataset_glue_downstream` | contradicts absmean PTQ; recommends_for decision tree | 0.95 |
| G-FBI | Fully binary LLM path | arXiv `2407.07093` | `paper_fbi_llm`, `gap_fbi_llm_repro` | alternative_to BitNet; non-goal | 0.8 / 0.75 |
| G-FINN | FPGA node without paper handle | Docs `14` + classic FINN FPGA'17 | `paper_finn`, `tool_brevitas_finn`, `hw_fpga_finn` | recommends_for FPGA | 0.9 |
| G-XNORPP | XNOR-Net++ absent | arXiv `1909.13863` | `paper_xnornetpp` | improves XNOR-Net | 0.9 |
| G-CLASSIC | BinaryConnect / DoReFa thin | Survey `02` | `paper_binaryconnect`, `paper_dorefa` | part_of / improves lineage | 0.9 |
| G-SURGE | STE future work unnamed | Survey / failure docs | `method_surge` | improves ste_mismatch | 0.7 lit |
| G-TMAC | LUT ternary tooling | Survey `02`/`16` | `method_tmac` | alternative_to bitnet.cpp | 0.75 |
| G-ENERGY | RAPL vs proxy unclear in graph | Spike `docs/spikes/RAPL_ENERGY_SPIKE.md` + results | `sys_energy_module`, `gap_rapl_windows`, `metric_energy_proxy_ratio` | implements proxy; Windows open | 0.9 |
| G-API | Optimise / HF / bridges underlinked | Repo docs + results bridges | `sys_optimise_api`, `sys_hf_optimiser`, `sys_bridge_recipes`, `sys_guide_e2e`, `sys_wrap_policy`, `sys_calibrate_qat` | part_of lab; recommends_for | 0.9 |
| G-PAPERS | Novel B1–B3 not as first-class Paper nodes | Vault READMEs + `docs/32` | `paper_b1_*`, `paper_b2_*`, `paper_b3_*`, `gap_venue_submit` | derived_from lab evidence | 0.85 idea_vault |
| G-ROADMAP | v0.3 vs v1.0 leftovers | `ROADMAP.md` + `MOONSHOT_DEFERRALS.md` | `roadmap_v030`, `world_class_v1`, multiple `gap_*` | blocked_by edges | 0.85–0.95 |
| G-CONTRA | Op-count vs wall-clock not explicit edge | Results SUMMARY honesty table | `concept_word_vs_wallclock` + `contradicts` edges | theory ↔ S_e2e | 1.0 |
| G-NPU | Vendor table not graph-native | `docs/20` | `npu_no_native_1bit`, `decision_int8_npu_first`, `hw_phone_npu` | recommends INT8 tools | 1.0 |

## Parallel research notes (session)

- **academia MCP** `arxiv_search`: BitNet Distillation `2510.13998`, a4.8 `2411.04965`,
  BitNet v2 `2504.18415`, BitNet `2310.11453`, 2B4T `2504.12285`, XNOR-Net `1603.05279`,
  Bi-Real `1811.01335`, bitnet.cpp `2410.16144`, Larq CE `2011.09398`, BiBERT `2203.06390`.
- **Exa** `web_search_exa`: ReActNet abstract / RSign / RPReLU / 69.4% ImageNet details.
- **Tavily** `tavily_search`: BitDistill memory/latency claims; Sparse-BitNet HTML `2603.05168`;
  Microsoft BitNet GitHub timeline.

## Enrichment overlay merge (commit `3460c94` → union)

Parallel agent shipped `knowledge_graph/enrichment/gap_research_nodes.json`
(**24 nodes / 29 edges**) with ImageNet ladder numbers, Real-to-Binary (`2003.11535`),
FINN FPS/W, bitnet.cpp device-footnoted energy tables, BiBERT attention failure modes,
Sparse-BitNet 2:4 deltas, HF `bitnet-b1.58-2B-4T` artifact, and AWQ/GPTQ decision detail.

| Step | Result |
|------|--------|
| Tool | `python scripts/merge_kg_enrichment.py` |
| Policy | Union-by-id; prefer enrichment summaries for citation-dense papers; keep all lab nodes |
| Aliases | `same_as` edges between dotted IDs (`paper.bireal.eccv2018`) and lab IDs (`paper_bireal`) |
| Post-merge | **154 nodes / 265 edges**; `kg_validate` PASS; `pytest tests/test_kg.py` green |

Notable enrichment-only IDs retained in the main graph:

- `paper.real2bin.iclr2020` — Real-to-Binary 65.4% ImageNet (was missing as first-class node)
- `metric.imagenet.bnn_ladder` — BinaryNet→XNOR→Bi-Real→R2B→ReActNet ladder
- `recipe.qat_158bit` — from-scratch vs BitDistill checklist
- `failure.attention_binarization`, `failure.small_k_layers`, `failure.bn_sensitivity`
- `artifact.hf_bitnet_2b`, `citation.strengthen.lab_weak`
- `decision.awq_gptq_vs_binary` (aliased to `decision_wrap_tree`)

Provenance folder `knowledge_graph/enrichment/` is **kept** (not deleted).

## Still open (intentionally)

See `OpenGap` nodes in the JSON. Highest leverage leftovers:

1. Venue drafting for B1–B3 (`gap_venue_submit`)
2. Distill / WC-O hardening (`gap_distill_integration`, `decision_wc_o_gates`) — **Lane A PR #19**
3. `.bnnpack` v2 + safetensors (`gap_bnnpack_v2`) — **Lane B PR #18**
4. Local Litespark / Sparse-BitNet benches (`gap_litespark_local`) — **do not invent**
5. Full ReActNet activations in `bnn.ste` (`gap_reactnet_in_repo`)
6. Wave 1 moonshots still on open PRs: WASM (#24), energy/RAPL (#22), bitnet pin (#17), ImageNet protocol (#21)

## Integrity enrichment (2026-08-04, `lane/kg-enrich`)

| Step | Result |
|------|--------|
| Overlay | `knowledge_graph/enrichment/integrity_wave1.json` |
| Tool | `python scripts/apply_kg_integrity.py` |
| Adds | WC-O gates, `sys_recommend_stack`, `sys_eval_suite`, `sys_kg`, BitDistiller/GPTQ/Q-Sparse, RAPL Result, moonshot non-goals |
| Fixes | Broken sources; over-aliased `same_as` → `implements`/`part_of`/`derived_from` |
| Honesty | Wave 1 statuses `open_pr` — **not** claimed merged |

## Rebuild after edits

```bash
python scripts/build_bnn_kg.py
python scripts/merge_kg_enrichment.py
python scripts/apply_kg_integrity.py
python scripts/kg_validate.py
pytest tests/test_kg.py -q
```
