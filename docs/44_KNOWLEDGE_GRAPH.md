# Knowledge Graph (lab corpus map)

| Field | Value |
|-------|-------|
| **Status** | Shipped with repo (integrity CI gated) |
| **Canonical** | [`knowledge_graph/`](../knowledge_graph/) |
| **Machine graph** | [`knowledge_graph/bnn_kg.json`](../knowledge_graph/bnn_kg.json) · [`.graphml`](../knowledge_graph/bnn_kg.graphml) |
| **Human view** | [`knowledge_graph/VIEW.md`](../knowledge_graph/VIEW.md) |
| **Gaps log** | [`knowledge_graph/GAPS_FILLED.md`](../knowledge_graph/GAPS_FILLED.md) |
| **CLI** | `bnn kg` · `bnn kg validate` |
| **Related CLI** | `bnn recommend --goal …` · `bnn eval-suite` |

Maps thesis lock, dual-metric culture, packed kernels, wrap decision tree, STE family,
BitNet/GGUF/torchao bridges, modality canaries, novel papers B1–B3, WC-O gates, and ROADMAP
open gaps into a validated node/edge graph.

> Note: doc id **44** (not 41) — `docs/41_*` is reserved for portable SIMD kernels.

```bash
bnn kg                  # summary + top OpenGaps
bnn kg validate         # same as scripts/kg_validate.py
pytest tests/test_kg.py -q
# rebuild lab corpus builder (optional; then re-merge + integrity)
python scripts/build_bnn_kg.py
python scripts/merge_kg_enrichment.py
python scripts/apply_kg_integrity.py
```

Loader: `from bnn.kg import load_kg, neighborhood, nodes_by_type`.

Stack routing (decision tree companion): `bnn recommend --goal cpu-llm` (see
[`scripts/recommend_stack.py`](../scripts/recommend_stack.py)). Fair published-shape
eval: `bnn eval-suite` → [`docs/FAIR_EVAL_PROTOCOL.md`](FAIR_EVAL_PROTOCOL.md).

Lane progress: [`docs/lanes/kg.md`](lanes/kg.md). Lanes A–I are merged; KG
`open_pr` fields were stale and are flipped to `merged` / `closed_by_policy` /
`established`. **`gap_pypi_trusted` stays `open`** — `bnn-lab` is not on PyPI.

Maintainer-local only (not required to clone/build): optional external mirror index at
`C:\00 Research Papers\BINARY_NEURAL_KG_INDEX.md` on the primary maintainer machine.

> Doc number **44** (not 41): `docs/41_*` is reserved for portable SIMD kernels.
