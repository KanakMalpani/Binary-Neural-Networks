# MERGE_INSTRUCTIONS — append-only enrichment → main KG

Use when `knowledge_graph/bnn_kg.json` exists (owned by another agent/human). **Do not force-rewrite** their file; union by `id`.

## Files

| File | Role |
|------|------|
| `enrichment/gap_research_nodes.json` | Nodes + edges + meta (`researcher: gap-parallel`) |
| `enrichment/SOURCES.md` | Bibliography |
| `enrichment/NOTES.md` | Gap rationale |

## Union algorithm (append-only)

```text
1. Load main = bnn_kg.json, enrich = enrichment/gap_research_nodes.json
2. For each node in enrich.nodes:
     if node.id not in main.nodes_by_id:
         append node
     else if enrich.confidence > main.confidence OR main.status == "stub":
         merge fields: prefer enrich.summary/sources for citation-dense papers;
         keep main.lab_local fields if present; set status "merged"
     else:
         skip body; optionally append enrich.sources entries missing from main
3. For each edge in enrich.edges:
     key = (source, target, relation)
     if key not in main.edges: append
4. Append meta.enrichment_runs += [{researcher, generated, node_count, edge_count}]
5. Write bnn_kg.json; do not delete enrichment/ (provenance)
```

## Suggested one-liner (Python)

```python
# scripts/merge_kg_enrichment.py  (create if missing)
import json
from pathlib import Path
root = Path("knowledge_graph")
main_p, enr_p = root / "bnn_kg.json", root / "enrichment" / "gap_research_nodes.json"
main, enr = json.loads(main_p.read_text(encoding="utf-8")), json.loads(enr_p.read_text(encoding="utf-8"))
by_id = {n["id"]: n for n in main.get("nodes", [])}
for n in enr["nodes"]:
    if n["id"] not in by_id:
        main.setdefault("nodes", []).append(n)
        by_id[n["id"]] = n
ek = {(e["source"], e["target"], e["relation"]) for e in main.get("edges", [])}
for e in enr["edges"]:
    k = (e["source"], e["target"], e["relation"])
    if k not in ek:
        main.setdefault("edges", []).append(e)
        ek.add(k)
main.setdefault("meta", {}).setdefault("enrichment_runs", []).append(enr.get("meta", {}))
main_p.write_text(json.dumps(main, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"nodes={len(main['nodes'])} edges={len(main['edges'])}")
```

## ID namespace

Enrichment IDs use prefixes: `paper.*`, `tool.*`, `failure.*`, `decision.*`, `recipe.*`, `concept.*`, `metric.*`, `artifact.*`, `citation.*`.  
If main KG uses different IDs for the same paper, add an edge `same_as` rather than duplicating content, or alias in merge.

## Conflict policy

- Prefer enrichment for **external paper metrics** and arXiv IDs.
- Prefer main KG for **lab results** (`results/*.json`, MNIST/CIFAR floors).
- Never overwrite thesis lock nodes claiming GPU 32× from `sign()`.
