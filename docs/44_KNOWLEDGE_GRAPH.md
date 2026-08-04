# Knowledge Graph (lab corpus map)

| Field | Value |
|-------|-------|
| **Status** | Shipped with repo |
| **Canonical** | [`knowledge_graph/`](../knowledge_graph/) |
| **Machine graph** | [`knowledge_graph/bnn_kg.json`](../knowledge_graph/bnn_kg.json) · [`.graphml`](../knowledge_graph/bnn_kg.graphml) |
| **Human view** | [`knowledge_graph/VIEW.md`](../knowledge_graph/VIEW.md) |
| **Gaps log** | [`knowledge_graph/GAPS_FILLED.md`](../knowledge_graph/GAPS_FILLED.md) |

Maps thesis lock, dual-metric culture, packed kernels, wrap decision tree, STE family,
BitNet/GGUF/torchao bridges, modality canaries, novel papers B1–B3, and ROADMAP open gaps
into a validated node/edge graph.

> Note: doc id **44** (not 41) — `docs/41_*` is reserved for portable SIMD kernels.

```bash
python scripts/kg_validate.py
pytest tests/test_kg.py -q
# rebuild lab corpus builder (optional)
python scripts/build_bnn_kg.py
# union enrichment overlay into main graph
python scripts/merge_kg_enrichment.py
```

Loader: `from bnn.kg import load_kg, neighborhood, nodes_by_type`.

External mirror index: `C:\00 Research Papers\BINARY_NEURAL_KG_INDEX.md`.

> Doc number **44** (not 41): `docs/41_*` is reserved for portable SIMD kernels.
