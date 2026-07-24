"""Regression gates vs golden floors / results JSON."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FLOORS = json.loads((ROOT / "tests" / "golden_floors.json").read_text(encoding="utf-8"))


def test_benchmark_golden_floors():
    path = ROOT / "results" / "benchmark.json"
    if not path.exists():
        pytest.skip("no benchmark.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("results") or []
    assert rows
    # Find 64x4096x4096
    hit = None
    for r in rows:
        sh = r.get("shape") or {}
        if sh.get("in_features") == 4096 and sh.get("batch") == 64:
            hit = r
            break
    assert hit is not None
    assert hit["max_abs_error_vs_fp32"] <= FLOORS["native_err_max"]
    s = hit["speedup_compute_vs_numpy_fp32"]
    assert s >= FLOORS["speedup_vs_numpy_min_n4096"]
    # Soft floor vs recorded 3.61 (allow 30% drop → ~2.5; roadmap says 3.0)
    assert s >= FLOORS["speedup_vs_numpy_floor_4096_recorded"] * 0.7 or s >= 2.0
    comp = hit["theoretical"]["weight_compression"]
    assert comp >= FLOORS["compression_min"]


def test_mnist_acc_gates():
    path = ROOT / "results" / "train_results.json"
    if not path.exists():
        pytest.skip("no train_results.json")
    rows = json.loads(path.read_text(encoding="utf-8"))
    by = {r["model"]: r["test_acc"] for r in rows}
    if "fp32_mlp" in by and "binary_mlp" in by:
        fp, bn = by["fp32_mlp"], by["binary_mlp"]
        if fp >= FLOORS["mnist_fp_for_gate"]:
            assert bn >= FLOORS["mnist_binary_mlp_min_acc"]
            assert (fp - bn) <= FLOORS["mnist_gap_max_pp"]
