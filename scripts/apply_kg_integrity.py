#!/usr/bin/env python3
"""Apply knowledge_graph/enrichment/integrity_wave1.json onto bnn_kg.json.

Idempotent-ish: re-running skips duplicate edges by (source, target, relation).
Fixes broken sources, Wave-1 open-PR status honesty, same_as repairs, new nodes.
Regenerates bnn_kg.graphml. Does NOT re-run build_bnn_kg (preserves enrichment merge).
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from xml.dom import minidom

ROOT = Path(__file__).resolve().parents[1]
KG_DIR = ROOT / "knowledge_graph"
MAIN_P = KG_DIR / "bnn_kg.json"
PATCH_P = KG_DIR / "enrichment" / "integrity_wave1.json"
GRAPHML_P = KG_DIR / "bnn_kg.graphml"


def to_graphml(nodes: list[dict], edges: list[dict]) -> str:
    g = ET.Element("graphml", xmlns="http://graphml.graphdrawing.org/xmlns")
    for key, kind, attr_name, attr_type in [
        ("label", "node", "label", "string"),
        ("type", "node", "type", "string"),
        ("summary", "node", "summary", "string"),
        ("confidence", "node", "confidence", "double"),
        ("status", "node", "status", "string"),
        ("sources", "node", "sources", "string"),
        ("relation", "edge", "relation", "string"),
        ("evidence", "edge", "evidence", "string"),
        ("weight", "edge", "weight", "double"),
    ]:
        ET.SubElement(
            g, "key", id=key, **{"for": kind, "attr.name": attr_name, "attr.type": attr_type}
        )
    graph = ET.SubElement(g, "graph", id="bnn_kg", edgedefault="directed")
    for n in nodes:
        ne = ET.SubElement(graph, "node", id=n["id"])
        for k in ("label", "type", "summary", "confidence", "status"):
            d = ET.SubElement(ne, "data", key=k)
            d.text = str(n.get(k, ""))
        d = ET.SubElement(ne, "data", key="sources")
        d.text = " | ".join(n.get("sources", []))
    for i, e in enumerate(edges):
        ee = ET.SubElement(graph, "edge", id=f"e{i}", source=e["source"], target=e["target"])
        d = ET.SubElement(ee, "data", key="relation")
        d.text = e["relation"]
        d = ET.SubElement(ee, "data", key="evidence")
        d.text = " | ".join(e.get("evidence", []))
        d = ET.SubElement(ee, "data", key="weight")
        d.text = str(e.get("weight", 1.0))
    rough = ET.tostring(g, encoding="utf-8")
    return minidom.parseString(rough).toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")


def _norm_edge(raw: dict[str, Any]) -> dict[str, Any]:
    evidence = raw.get("evidence") or []
    if isinstance(evidence, str):
        evidence = [evidence]
    return {
        "source": raw["source"],
        "target": raw["target"],
        "relation": raw["relation"],
        "evidence": list(evidence),
        "weight": float(raw.get("weight", 1.0)),
    }


def apply_patch(main_g: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    by_id: dict[str, dict[str, Any]] = {n["id"]: dict(n) for n in main_g["nodes"]}

    for nid, sources in (patch.get("source_fixes") or {}).items():
        if nid in by_id:
            by_id[nid]["sources"] = list(sources)

    for nid, fields in (patch.get("status_patches") or {}).items():
        if nid not in by_id:
            continue
        for k, v in fields.items():
            by_id[nid][k] = v

    added_nodes = 0
    for raw in patch.get("nodes") or []:
        n = dict(raw)
        n.setdefault("confidence", 0.5)
        n["confidence"] = float(n["confidence"])
        n["sources"] = list(n.get("sources") or [])
        if n["id"] in by_id:
            # Prefer patch summary/sources/status for known integrity nodes
            cur = by_id[n["id"]]
            for k in ("label", "type", "summary", "sources", "confidence", "status", "arxiv"):
                if k in n and n[k] is not None:
                    cur[k] = n[k]
            by_id[n["id"]] = cur
        else:
            by_id[n["id"]] = n
            added_nodes += 1

    # Preserve order: existing first, then new
    nodes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for n in main_g["nodes"]:
        nid = n["id"]
        if nid in by_id and nid not in seen:
            nodes.append(by_id[nid])
            seen.add(nid)
    for nid, n in by_id.items():
        if nid not in seen:
            nodes.append(n)
            seen.add(nid)

    remove_pairs = {
        (a, b) for a, b in (patch.get("remove_same_as_pairs") or [])
    } | {(b, a) for a, b in (patch.get("remove_same_as_pairs") or [])}

    edges: list[dict[str, Any]] = []
    ek: set[tuple[str, str, str]] = set()
    removed_same_as = 0
    for e in main_g.get("edges") or []:
        if e.get("relation") == "same_as" and (e["source"], e["target"]) in remove_pairs:
            removed_same_as += 1
            continue
        ne = _norm_edge(e)
        k = (ne["source"], ne["target"], ne["relation"])
        if k in ek:
            continue
        edges.append(ne)
        ek.add(k)

    added_edges = 0
    for raw in patch.get("edges") or []:
        ne = _norm_edge(raw)
        k = (ne["source"], ne["target"], ne["relation"])
        if k in ek:
            continue
        edges.append(ne)
        ek.add(k)
        added_edges += 1

    ids = {n["id"] for n in nodes}
    dangling = [e for e in edges if e["source"] not in ids or e["target"] not in ids]
    if dangling:
        raise SystemExit(f"Dangling edges after integrity patch: {dangling[:5]}")

    meta = dict(main_g.get("meta") or {})
    meta["node_count"] = len(nodes)
    meta["edge_count"] = len(edges)
    meta["node_types"] = sorted({n["type"] for n in nodes})
    meta["relations"] = sorted({e["relation"] for e in edges})
    meta["version"] = "1.2.0"
    meta["integrity_patched_at"] = datetime.now(UTC).isoformat()
    meta["generated_by"] = (
        "scripts/build_bnn_kg.py + scripts/merge_kg_enrichment.py + "
        "scripts/apply_kg_integrity.py"
    )
    runs = list(meta.get("enrichment_runs") or [])
    runs.append(
        {
            **(patch.get("meta") or {}),
            "integrity_added_nodes": added_nodes,
            "integrity_added_edges": added_edges,
            "integrity_removed_same_as": removed_same_as,
        }
    )
    meta["enrichment_runs"] = runs
    return {"meta": meta, "nodes": nodes, "edges": edges}


def main() -> None:
    main_g = json.loads(MAIN_P.read_text(encoding="utf-8"))
    patch = json.loads(PATCH_P.read_text(encoding="utf-8"))
    out = apply_patch(main_g, patch)
    MAIN_P.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    GRAPHML_P.write_text(to_graphml(out["nodes"], out["edges"]), encoding="utf-8")
    print(
        f"Integrity KG: nodes={out['meta']['node_count']} edges={out['meta']['edge_count']} "
        f"(run meta appended)"
    )


if __name__ == "__main__":
    main()
