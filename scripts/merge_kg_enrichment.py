#!/usr/bin/env python3
"""Union-by-id merge of enrichment/gap_research_nodes.json into bnn_kg.json.

Preserves all lab nodes. Prefers enrichment summaries/sources for citation-dense
papers when confidence is higher or lab status is stub/idea_vault.
Adds same_as edges between dotted enrichment IDs and lab snake_case IDs.
Regenerates bnn_kg.graphml.
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.dom import minidom

ROOT = Path(__file__).resolve().parents[1]
KG_DIR = ROOT / "knowledge_graph"
MAIN_P = KG_DIR / "bnn_kg.json"
ENR_P = KG_DIR / "enrichment" / "gap_research_nodes.json"
GRAPHML_P = KG_DIR / "bnn_kg.graphml"

TYPE_MAP = {
    "paper": "Paper",
    "metric_table": "Metric",
    "metric": "Metric",
    "recipe": "Method",
    "tool": "Tool",
    "decision": "Decision",
    "concept": "Concept",
    "failure_mode": "FailureMode",
    "artifact": "System",
    "meta": "Decision",
}

REL_MAP = {
    "contributes_metric": "measured_on",
    "improves_upon": "improves",
    "outperforms_on_metric": "improves",
    "precedes": "derived_from",
    "extends": "improves",
    "enabled_by_runtime": "requires",
    "runs_artifact": "implements",
    "defines_recipe": "implements",
    "mitigates": "improves",
    "complements": "alternative_to",
    "diagnoses": "derived_from",
    "instance_of": "part_of",
    "deployed_via": "implements",
    "prefers_when_cpu_llm": "recommends_for",
    "prefers_when_fpga_vision": "recommends_for",
    "hybridizes_with": "improves",
    "mitigated_in_architecture": "improves",
    "constrains": "requires",
    "requires_cite": "cites",
    "partially_avoids": "improves",
    "informs": "part_of",
    "uses": "requires",
    "implements_flow_of": "implements",
}

# enrichment dotted id → lab snake_case id
SAME_AS: dict[str, str] = {
    "paper.bireal.eccv2018": "paper_bireal",
    "paper.reactnet.eccv2020": "paper_reactnet",
    "paper.bitnet.2023": "paper_bitnet",
    "paper.bitnet_b158.2024": "paper_bitnet_b158",
    "paper.bitnet_a48.2024": "paper_bitnet_a48",
    "paper.bitnet_cpp.2024": "paper_bitnet_cpp",
    "paper.bitdistill.2025": "paper_bitdistill",
    "paper.bibert.2022": "paper_bibert",
    "paper.finn.fpga2017": "paper_finn",
    "paper.sparse_bitnet.2026": "paper_sparse_bitnet",
    "tool.larq_lce": "tool_larq",
    "tool.brevitas_finn": "tool_brevitas_finn",
    "failure.ptq_to_ternary": "ptq_ternary_llm_wipe",
    "concept.mobile_npu_quant_reality": "npu_no_native_1bit",
    "decision.awq_gptq_vs_binary": "decision_wrap_tree",
    "artifact.hf_bitnet_2b": "paper_bitnet_2b4t",
}


def normalize_type(t: str) -> str:
    if t in TYPE_MAP:
        return TYPE_MAP[t]
    # already TitleCase lab types
    if t and t[0].isupper():
        return t
    return t[:1].upper() + t[1:] if t else "Concept"


def normalize_node(n: dict[str, Any]) -> dict[str, Any]:
    out = dict(n)
    out["type"] = normalize_type(str(n.get("type", "Concept")))
    src = n.get("sources") or []
    out["sources"] = list(src) if isinstance(src, list) else [str(src)]
    out["confidence"] = float(n.get("confidence", 0.5))
    out["status"] = n.get("status") or "literature"
    out.setdefault("label", out["id"])
    out.setdefault("summary", "")
    return out


def merge_node(lab: dict[str, Any], enr: dict[str, Any]) -> dict[str, Any]:
    """Prefer enrichment citation density; keep lab-local fields."""
    merged = dict(lab)
    enr_c = float(enr.get("confidence", 0))
    lab_c = float(lab.get("confidence", 0))
    lab_status = str(lab.get("status", ""))
    prefer_enr = enr_c > lab_c or lab_status in ("stub", "literature", "idea_vault")
    if prefer_enr or len(enr.get("summary", "")) > len(lab.get("summary", "")) + 40:
        # enrichment often has ImageNet numbers — keep if denser
        if "ImageNet" in enr.get("summary", "") or "arXiv" in enr.get("summary", "") or enr_c >= lab_c:
            merged["summary"] = enr["summary"]
            merged["confidence"] = max(lab_c, enr_c)
    # union sources
    seen = set(merged.get("sources") or [])
    for s in enr.get("sources") or []:
        if s not in seen:
            merged.setdefault("sources", []).append(s)
            seen.add(s)
    merged["status"] = "merged"
    merged["enrichment_id"] = enr["id"]
    if enr.get("arxiv") and not merged.get("arxiv"):
        merged["arxiv"] = enr["arxiv"]
    return merged


def normalize_edge(e: dict[str, Any]) -> dict[str, Any]:
    rel = e.get("relation", "part_of")
    mapped = REL_MAP.get(rel, rel)
    evidence = e.get("evidence", [])
    if isinstance(evidence, str):
        evidence = [evidence]
    out: dict[str, Any] = {
        "source": e["source"],
        "target": e["target"],
        "relation": mapped,
        "evidence": list(evidence),
        "weight": float(e.get("weight", 1.0)),
    }
    if mapped != rel:
        out["relation_original"] = rel
    return out


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
        ET.SubElement(g, "key", id=key, **{"for": kind, "attr.name": attr_name, "attr.type": attr_type})
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


def main() -> None:
    main_g = json.loads(MAIN_P.read_text(encoding="utf-8"))
    enr = json.loads(ENR_P.read_text(encoding="utf-8"))

    by_id: dict[str, dict[str, Any]] = {n["id"]: n for n in main_g["nodes"]}
    added_nodes = 0
    merged_nodes = 0
    aliased = 0

    for raw in enr["nodes"]:
        n = normalize_node(raw)
        eid = n["id"]
        lab_id = SAME_AS.get(eid)

        if eid in by_id:
            by_id[eid] = merge_node(by_id[eid], n)
            merged_nodes += 1
        elif lab_id and lab_id in by_id:
            # Enrichment ID is new; keep both — enrich lab node summary AND add enrichment node
            by_id[lab_id] = merge_node(by_id[lab_id], n)
            merged_nodes += 1
            if eid not in by_id:
                n["status"] = "merged"
                n["lab_alias"] = lab_id
                by_id[eid] = n
                added_nodes += 1
                aliased += 1
        else:
            by_id[eid] = n
            added_nodes += 1

    # rebuild nodes list preserving roughly lab-first order then new
    lab_order = [n["id"] for n in main_g["nodes"]]
    nodes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i in lab_order:
        if i in by_id and i not in seen:
            nodes.append(by_id[i])
            seen.add(i)
    for i, n in by_id.items():
        if i not in seen:
            nodes.append(n)
            seen.add(i)

    edges = list(main_g.get("edges", []))
    ek = {(e["source"], e["target"], e["relation"]) for e in edges}
    added_edges = 0

    for raw in enr.get("edges", []):
        e = normalize_edge(raw)
        k = (e["source"], e["target"], e["relation"])
        if k not in ek:
            edges.append(e)
            ek.add(k)
            added_edges += 1

    # same_as bridges
    for enr_id, lab_id in SAME_AS.items():
        if enr_id in by_id and lab_id in by_id:
            for src, tgt in ((enr_id, lab_id), (lab_id, enr_id)):
                k = (src, tgt, "same_as")
                if k not in ek:
                    edges.append(
                        {
                            "source": src,
                            "target": tgt,
                            "relation": "same_as",
                            "evidence": ["enrichment↔lab ID alias (merge_kg_enrichment.py)"],
                            "weight": 1.0,
                        }
                    )
                    ek.add(k)
                    added_edges += 1

    # dangling check
    ids = {n["id"] for n in nodes}
    dangling = [e for e in edges if e["source"] not in ids or e["target"] not in ids]
    if dangling:
        raise SystemExit(f"Dangling edges after merge: {dangling[:5]}")

    meta = dict(main_g.get("meta") or {})
    meta["node_count"] = len(nodes)
    meta["edge_count"] = len(edges)
    meta["node_types"] = sorted({n["type"] for n in nodes})
    meta["relations"] = sorted({e["relation"] for e in edges})
    meta["version"] = "1.1.0"
    meta["merged_at"] = datetime.now(timezone.utc).isoformat()
    meta["generated_by"] = "scripts/build_bnn_kg.py + scripts/merge_kg_enrichment.py"
    runs = list(meta.get("enrichment_runs") or [])
    runs.append(
        {
            **(enr.get("meta") or {}),
            "merged_added_nodes": added_nodes,
            "merged_updated_lab_nodes": merged_nodes,
            "merged_added_edges": added_edges,
            "same_as_aliases": aliased,
        }
    )
    meta["enrichment_runs"] = runs

    out = {"meta": meta, "nodes": nodes, "edges": edges}
    MAIN_P.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    GRAPHML_P.write_text(to_graphml(nodes, edges), encoding="utf-8")
    print(
        f"Merged KG: nodes={len(nodes)} edges={len(edges)} "
        f"(+{added_nodes} nodes, ~{merged_nodes} lab updates, +{added_edges} edges)"
    )


if __name__ == "__main__":
    main()
