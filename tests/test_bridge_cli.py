"""Lane E: bnn bridge CLI + figure / claims pipeline (W12)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from bnn import cli
from bnn.cli import (
    BRIDGE_ALIASES,
    BRIDGE_RECIPES,
    _bridge_script_path,
    _run_bridge,
    build_parser,
    main,
)


def test_bridge_in_help():
    help_txt = build_parser().format_help()
    assert "bridge" in help_txt


def test_bridge_list(capsys):
    assert main(["bridge", "list"]) == 0
    out = capsys.readouterr().out
    assert "gpu" in out
    assert "cpu-llm" in out
    assert "torchao_int4_recipe.py" in out
    assert "llamacpp_bitnet_recipe.py" in out


def test_bridge_script_path_rejects_traversal():
    with pytest.raises(FileNotFoundError):
        _bridge_script_path("../pareto_report.py")
    with pytest.raises(FileNotFoundError):
        _bridge_script_path("missing_recipe.py")


@pytest.mark.parametrize("name", list(BRIDGE_RECIPES))
def test_bridge_recipe_scripts_exist(name: str):
    meta = BRIDGE_RECIPES[name]
    path = _bridge_script_path(meta["script"])
    assert path.is_file()


@pytest.mark.parametrize(("alias", "canon"), list(BRIDGE_ALIASES.items()))
def test_bridge_aliases_resolve(alias: str, canon: str):
    assert canon in BRIDGE_RECIPES
    assert main(["bridge", alias, "--help"]) == 0


def test_bridge_gpu_argv(tmp_path: Path):
    calls: list[tuple[str, list[str]]] = []

    def fake(script: str, extra: list[str] | None = None) -> int:
        calls.append((script, list(extra or [])))
        return 0

    out = tmp_path / "gpu.json"
    with patch.object(cli, "_run_bridge", side_effect=fake):
        assert main(["bridge", "gpu", "--probe", "--out", str(out)]) == 0
    assert calls == [("torchao_int4_recipe.py", ["--probe", "--out", str(out)])]


def test_bridge_cpu_llm_real(tmp_path: Path):
    out = tmp_path / "cpu.json"
    assert main(["bridge", "cpu-llm", "--out", str(out)]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload.get("lane") == "cpu-llm"
    assert "bitnet_cpp" in payload or "llama_cpp_gguf" in payload


def test_bridge_figures_from_committed_results(tmp_path: Path):
    out = tmp_path / "figures_manifest.json"
    assert main(["bridge", "figures", "--out", str(out)]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema"] == "bnn_figures_manifest_v1"
    assert payload["claims_blocked_count"] == 0
    assert payload["claims_allowed_count"] >= 6
    ids = {c["id"] for c in payload["claims"]}
    assert "C1_compression_32x" in ids
    assert "C3_dual_metric" in ids


def test_pareto_from_results(tmp_path: Path):
    out = tmp_path / "pareto.json"
    assert main(["pareto", "--from-results", "--out", str(out)]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema"] == "bnn_pareto_report_v1"
    assert len(payload["points"]) >= 2
    names = {p["name"] for p in payload["points"]}
    assert "reference_anchor" in names


def test_run_bridge_rejects_bad_names():
    with pytest.raises(FileNotFoundError):
        _run_bridge("not_a_bridge.py")
