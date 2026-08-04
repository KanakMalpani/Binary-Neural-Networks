"""Smoke tests for the BNN knowledge graph."""
from __future__ import annotations

from pathlib import Path

import pytest

from bnn.kg import load_kg, neighborhood, nodes_by_type, validate_graph

ROOT = Path(__file__).resolve().parents[1]
KG_JSON = ROOT / "knowledge_graph" / "bnn_kg.json"
KG_GRAPHML = ROOT / "knowledge_graph" / "bnn_kg.graphml"


@pytest.fixture(scope="module")
def graph():
    assert KG_JSON.is_file(), "missing knowledge_graph/bnn_kg.json — run scripts/build_bnn_kg.py"
    return load_kg()


def test_artifacts_exist():
    assert KG_JSON.is_file()
    assert KG_GRAPHML.is_file()
    assert (ROOT / "knowledge_graph" / "README.md").is_file()
    assert (ROOT / "knowledge_graph" / "VIEW.md").is_file()
    assert (ROOT / "knowledge_graph" / "GAPS_FILLED.md").is_file()


def test_validate_pass(graph):
    errs = validate_graph(graph)
    assert errs == [], errs


def test_counts_match_meta(graph):
    assert graph["meta"]["node_count"] == len(graph["nodes"])
    assert graph["meta"]["edge_count"] == len(graph["edges"])
    assert len(graph["nodes"]) >= 100
    assert len(graph["edges"]) >= 150


def test_thesis_and_contradictions(graph):
    ids = {n["id"] for n in graph["nodes"]}
    assert "thesis_lock" in ids
    assert "fake_binary_sign" in ids
    assert "dual_metric_culture" in ids
    contras = [e for e in graph["edges"] if e["relation"] == "contradicts"]
    assert len(contras) >= 3
    # compression vs e2e honesty somewhere in graph
    assert any(
        e["source"] in ("compression_32x", "theoretical_word_reduction_64x", "concept_word_vs_wallclock")
        or e["target"] in ("metric_s_e2e", "theoretical_word_reduction_64x")
        for e in contras
    )


def test_claim_nodes_have_evidence(graph):
    for n in nodes_by_type(graph, "Claim") + nodes_by_type(graph, "Result"):
        assert n["sources"], n["id"]
        assert float(n["confidence"]) >= 0.8 or n.get("status") in ("literature", "idea_vault", "open")


def test_neighborhood_thesis(graph):
    nb = neighborhood(graph, "thesis_lock")
    assert len(nb) >= 3


def test_opengaps_present(graph):
    gaps = nodes_by_type(graph, "OpenGap")
    assert len(gaps) >= 5


def test_graphml_nonempty():
    text = KG_GRAPHML.read_text(encoding="utf-8")
    assert "<graphml" in text
    assert 'edgedefault="directed"' in text
    assert text.count("<node ") >= 100
