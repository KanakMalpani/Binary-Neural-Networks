"""Load and query the Binary Neural Network knowledge graph."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_KG = _REPO_ROOT / "knowledge_graph" / "bnn_kg.json"


@lru_cache(maxsize=1)
def load_kg(path: str | Path | None = None) -> dict[str, Any]:
    """Load `bnn_kg.json` (cached)."""
    p = Path(path) if path is not None else _DEFAULT_KG
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def clear_kg_cache() -> None:
    load_kg.cache_clear()


def node_index(graph: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    g = graph if graph is not None else load_kg()
    return {n["id"]: n for n in g["nodes"]}


def nodes_by_type(graph: dict[str, Any] | None = None, type_name: str = "") -> list[dict[str, Any]]:
    g = graph if graph is not None else load_kg()
    return [n for n in g["nodes"] if n.get("type") == type_name]


def neighborhood(
    graph: dict[str, Any] | None = None,
    node_id: str = "",
    *,
    direction: str = "both",
) -> list[dict[str, Any]]:
    """Return edges touching `node_id`. direction: in|out|both."""
    g = graph if graph is not None else load_kg()
    out: list[dict[str, Any]] = []
    for e in g["edges"]:
        if direction in ("out", "both") and e["source"] == node_id:
            out.append(e)
        elif direction in ("in", "both") and e["target"] == node_id:
            out.append(e)
    return out


def dangling_edges(graph: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    g = graph if graph is not None else load_kg()
    ids = {n["id"] for n in g["nodes"]}
    return [e for e in g["edges"] if e["source"] not in ids or e["target"] not in ids]


def orphan_nodes(
    graph: dict[str, Any] | None = None,
    allowed: Iterable[str] | None = None,
) -> list[str]:
    g = graph if graph is not None else load_kg()
    allow = set(allowed or ())
    allow |= set(g.get("meta", {}).get("allowed_orphans") or ())
    connected: set[str] = set()
    for e in g["edges"]:
        connected.add(e["source"])
        connected.add(e["target"])
    return sorted(n["id"] for n in g["nodes"] if n["id"] not in connected and n["id"] not in allow)


_ALLOWED_RELATIONS = frozenset(
    {
        "improves",
        "contradicts",
        "requires",
        "measured_on",
        "cites",
        "implements",
        "alternative_to",
        "blocked_by",
        "recommends_for",
        "part_of",
        "derived_from",
        "same_as",
    }
)


def validate_graph(graph: dict[str, Any] | None = None) -> list[str]:
    """Return list of error strings (empty if OK)."""
    g = graph if graph is not None else load_kg()
    errors: list[str] = []
    ids = [n["id"] for n in g["nodes"]]
    if len(ids) != len(set(ids)):
        errors.append("duplicate node ids")
    for e in dangling_edges(g):
        errors.append(f"dangling edge {e['source']} -> {e['target']}")
    for oid in orphan_nodes(g):
        errors.append(f"orphan node {oid}")
    required = (
        "thesis_lock",
        "dual_metric_culture",
        "fake_binary_sign",
        "compression_32x",
        "algo_xnor_gemm",
        "decision_wrap_tree",
    )
    present = set(ids)
    for rid in required:
        if rid not in present:
            errors.append(f"missing required node {rid}")
    for n in g["nodes"]:
        for field in ("id", "label", "type", "summary", "sources", "confidence", "status"):
            if field not in n:
                errors.append(f"node {n.get('id','?')} missing field {field}")
        if not isinstance(n.get("sources"), list) or not n["sources"]:
            errors.append(f"node {n.get('id')} needs non-empty sources")
        c = n.get("confidence")
        if not isinstance(c, (int, float)) or not (0.0 <= float(c) <= 1.0):
            errors.append(f"node {n.get('id')} bad confidence {c}")
    for e in g["edges"]:
        for field in ("source", "target", "relation", "evidence"):
            if field not in e:
                errors.append(f"edge missing {field}: {e}")
        # allow enrichment originals via relation_original; relation itself must be canonical
        if e.get("relation") not in _ALLOWED_RELATIONS:
            errors.append(f"unknown relation {e.get('relation')} on {e.get('source')}->{e.get('target')}")
        if not isinstance(e.get("evidence"), list) or not e["evidence"]:
            errors.append(f"edge {e.get('source')}->{e.get('target')} needs evidence list")
    return errors
