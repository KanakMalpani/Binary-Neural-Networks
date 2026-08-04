# Binary Neural Networks — Knowledge Graph

Machine-readable map of this lab’s thesis, evidence, tooling, literature, novel paper
candidates, and open gaps.

| Artifact | Role |
|----------|------|
| [`bnn_kg.json`](bnn_kg.json) | Canonical graph (`nodes[]`, `edges[]`, `meta`) |
| [`bnn_kg.graphml`](bnn_kg.graphml) | Gephi / Neo4j / NetworkX import |
| [`VIEW.md`](VIEW.md) | Human navigation + Mermaid diagrams |
| [`GAPS_FILLED.md`](GAPS_FILLED.md) | Gaps found → research → new nodes/edges |
| Rebuild (lab corpus) | `python scripts/build_bnn_kg.py` |
| Merge enrichment | `python scripts/merge_kg_enrichment.py` |
| Integrity overlay | `python scripts/apply_kg_integrity.py` |
| Validate | `bnn kg validate` · `python scripts/kg_validate.py` · `pytest tests/test_kg.py` |
| Enrichment provenance | [`enrichment/`](enrichment/) (gap-parallel + integrity_wave1; do not delete) |

## Thesis lock (immutable)

> Packed CPU/edge **XNOR-popcount** + honest **STE** simulation.
> **Never** claim GPU **32×** from `sign()`.
> Weight compression **32×** (uint64 pack) is exact — it is **not** an end-to-end latency claim.
> Report **dual metrics**: theory (compression / word reduction) **and** wall-clock (`S_compute`, `S_e2e`).

## Schema

### Node

```json
{
  "id": "snake_case_id",
  "label": "Human title",
  "type": "Concept|Method|Paper|Algorithm|System|Hardware|Metric|Result|Claim|FailureMode|Tool|Dataset|Decision|PersonOrg|OpenGap",
  "summary": "1–3 sentences",
  "sources": ["path or arXiv:ID", "..."],
  "confidence": 0.0,
  "status": "established|locked|open|literature|idea_vault|..."
}
```

### Edge

```json
{
  "source": "node_id",
  "target": "node_id",
  "relation": "improves|contradicts|requires|measured_on|cites|implements|alternative_to|blocked_by|recommends_for|part_of|derived_from",
  "evidence": ["path or note"],
  "weight": 1.0
}
```

## How to query

### Python (in-repo loader)

```python
from bnn.kg import load_kg, neighborhood, nodes_by_type

g = load_kg()
print(g["meta"]["node_count"], g["meta"]["edge_count"])
for n in nodes_by_type(g, "Claim"):
    print(n["id"], n["summary"][:80])
for e in neighborhood(g, "thesis_lock"):
    print(e["relation"], e["source"], "→", e["target"])
```

### jq

```bash
# All OpenGaps
jq '.nodes[] | select(.type=="OpenGap") | {id,label,status}' knowledge_graph/bnn_kg.json

# Edges that contradict
jq '.edges[] | select(.relation=="contradicts")' knowledge_graph/bnn_kg.json

# Evidence for compression claim
jq '.nodes[] | select(.id=="compression_32x")' knowledge_graph/bnn_kg.json
```

### NetworkX / Gephi

```python
import networkx as nx
G = nx.read_graphml("knowledge_graph/bnn_kg.graphml")
print(G.number_of_nodes(), G.number_of_edges())
```

Import `bnn_kg.graphml` directly in Gephi (File → Open) or Neo4j (`apoc.import.graphml`).

## Clusters (mental model)

1. **Thesis & honesty** — `thesis_lock`, dual metrics, fake-binary failure, Amdahl
2. **First principles** — XNOR identity, bandwidth, 32× / ~64× accounting
3. **Kernels & results** — packed GEMM, OpenMP curves, wrap e2e, energy proxy
4. **Training** — STE / ApproxSign / IR-Net / ReActNet (partial)
5. **Wrap decision tree** — BitNet / GGUF / torchao / AWQ / NPU INT8-first
6. **Modalities** — MNIST / CIFAR Bi-Real / audio synth / seq enc-dec
7. **Novel papers B1–B3** — local research vault under `C:\00 Research Papers\`
8. **Roadmap gaps** — v0.3.0 vs v1.0 leftovers (`OpenGap` nodes)

## Gap fill policy

Missing literature was researched via arXiv (academia MCP), Exa, and Tavily
(2024–2026 BitNet line, BiBERT, Larq CE, Sparse-BitNet, BitDistill, etc.).
**No invented lab metrics** — unreproduced literature numbers stay at lower
`confidence` and/or link to `OpenGap` nodes. See [`GAPS_FILLED.md`](GAPS_FILLED.md).

## Connectivity bar

`scripts/kg_validate.py` enforces:

- every edge endpoint exists
- no orphan nodes (degree 0) unless listed in `meta.allowed_orphans` (none today)
- required thesis nodes present (`thesis_lock`, `dual_metric_culture`, `fake_binary_sign`, …)

## Related docs

- [`docs/44_KNOWLEDGE_GRAPH.md`](../docs/44_KNOWLEDGE_GRAPH.md) — docs-index pointer
- [`docs/32_NOVEL_PAPER_CANDIDATES.md`](../docs/32_NOVEL_PAPER_CANDIDATES.md)
- [`ROADMAP.md`](../ROADMAP.md) · [`AGENTS.md`](../AGENTS.md)
- External index: `C:\00 Research Papers\BINARY_NEURAL_KG_INDEX.md`
