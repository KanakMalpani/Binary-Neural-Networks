"""Smoke tests for the BNN knowledge graph."""
from __future__ import annotations

import re
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


def test_shipped_wave1_lanes_are_not_open_pr(graph):
    """v1.0.0 / Wave 2 integrator merged lanes A–I — do not leave open_pr drift."""
    idx = {n["id"]: n for n in graph["nodes"]}
    shipped = {
        "gap_wasm": ("merged",),
        "hw_wasm": ("merged",),
        "gap_bnnpack_v2": ("merged",),
        "gap_distill_integration": ("merged",),
        "gap_layer_search_full": ("merged",),
        "gap_bitnet_submodule": ("closed_by_policy",),
        "gap_rapl_windows": ("closed_by_policy",),
        "decision_wc_o_gates": ("established",),
        "result_energy_rapl_spike": ("merged",),
        "gap_pypi_trusted": ("merged",),
    }
    for nid, allowed in shipped.items():
        assert idx[nid]["status"] in allowed, (nid, idx[nid]["status"])


def test_integrity_nodes_present(graph):
    ids = {n["id"] for n in graph["nodes"]}
    for nid in (
        "decision_wc_o_gates",
        "sys_recommend_stack",
        "sys_eval_suite",
        "sys_kg",
        "paper_bitdistiller",
        "paper_gptq",
        "paper_qsparse",
        "result_energy_rapl_spike",
        "decision_nongoal_asr_whisper",
        "decision_nongoal_memory_arena",
        "decision_imagenet_protocol_not_sota",
    ):
        assert nid in ids, nid


def test_ternary_source_path(graph):
    idx = {n["id"]: n for n in graph["nodes"]}
    srcs = idx["algo_ternary_bitplane"]["sources"]
    assert "bnn/kernels/ternary_pack.py" in srcs
    assert "bnn/ternary_pack.py" not in srcs


def test_same_as_not_cross_type(graph):
    """System↔Paper / Decision-subset / Concept↔FailureMode must not use same_as or lab_alias."""
    forbidden = {
        frozenset({"artifact.hf_bitnet_2b", "paper_bitnet_2b4t"}),
        frozenset({"decision.awq_gptq_vs_binary", "decision_wrap_tree"}),
        frozenset({"concept.mobile_npu_quant_reality", "npu_no_native_1bit"}),
    }
    for e in graph["edges"]:
        if e["relation"] != "same_as":
            continue
        pair = frozenset({e["source"], e["target"]})
        assert pair not in forbidden, e
    idx = {n["id"]: n for n in graph["nodes"]}
    for nid in (
        "artifact.hf_bitnet_2b",
        "decision.awq_gptq_vs_binary",
        "concept.mobile_npu_quant_reality",
    ):
        assert "lab_alias" not in idx[nid], nid
    # No inverted paper→artifact derived_from (artifact implements paper instead)
    for e in graph["edges"]:
        if (
            e["source"] == "paper_bitnet_2b4t"
            and e["target"] == "artifact.hf_bitnet_2b"
            and e["relation"] == "derived_from"
        ):
            raise AssertionError(f"inverted derived_from still present: {e}")


def test_same_as_and_lab_alias_type_hygiene(graph):
    by_id = {n["id"]: n for n in graph["nodes"]}
    for n in graph["nodes"]:
        alias = n.get("lab_alias")
        if alias is None:
            continue
        assert alias in by_id, n["id"]
        assert by_id[alias]["type"] == n["type"], (n["id"], alias)
    for e in graph["edges"]:
        if e["relation"] != "same_as":
            continue
        assert by_id[e["source"]]["type"] == by_id[e["target"]]["type"], e


def test_repo_relative_sources_exist(graph):
    """Local path-like sources should resolve on the checked-out tree."""
    skip_prefixes = ("http://", "https://", "arXiv:", "arxiv:", "doi:", "hf:")
    abs_local = re.compile(r"^[A-Za-z]:[/\\]|^\\\\")
    missing: list[str] = []
    absolute: list[str] = []
    for n in graph["nodes"]:
        for s in n.get("sources") or []:
            if not isinstance(s, str) or s.startswith(skip_prefixes):
                continue
            if abs_local.match(s) or "Research Papers" in s:
                absolute.append(f"{n['id']}: {s}")
                continue
            # Allow bare filenames / section anchors only when they look like paths
            if "/" not in s and "\\" not in s:
                continue
            # Strip markdown anchors
            path_s = s.split("#", 1)[0].strip()
            if not path_s:
                continue
            p = ROOT / path_s
            if not p.exists():
                missing.append(f"{n['id']}: {s}")
    assert absolute == [], absolute
    assert missing == [], missing
