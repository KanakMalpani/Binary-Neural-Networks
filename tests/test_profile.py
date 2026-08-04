"""Profile CLI smoke + soft latency budgets (W13.T03) + baselines (W13.T06)."""

from __future__ import annotations

import json
from pathlib import Path

from bnn.cli import main as cli_main
from bnn.profile import (
    SOFT_BUDGETS_MS,
    check_committed_bench_soft_floors,
    check_soft_budgets,
    profile_packed_linear,
)


def test_profile_breakdown_keys():
    br = profile_packed_linear(m=8, n=256, k=256, reps=3, warmup=1)
    d = br.to_dict()
    for key in (
        "pack_weight_ms",
        "pack_act_ms",
        "gemm_ms",
        "e2e_forward_ms",
        "torch_fp32_ms",
        "torch_int8_wo_ms",
        "speedup_vs_fp32",
        "speedup_vs_int8_wo",
        "baselines",
    ):
        assert key in d
        if key == "baselines":
            assert "torch_fp32_ms" in d[key]
            assert "torch_int8_weight_only_ms" in d[key]
        else:
            assert d[key] >= 0
    assert "overhead_vs_gemm" in d
    assert isinstance(d["overhead_vs_gemm"], float)
    assert d["torch_int8_wo_ms"] > 0
    assert d["baselines"]["packed_e2e_ms"] == d["e2e_forward_ms"]


def test_soft_budgets_pass_on_smoke_shape():
    """CI hard-assert on the tiny smoke shape; eval-suite stays warn-only by default.

    Soft ceilings in ``SOFT_BUDGETS_MS`` are deliberately loose — this only fails
    on catastrophic regression. Product ``bnn eval-suite`` still warns unless
    ``--strict-budgets`` (see ``check_soft_budgets`` docstring).
    """
    br = profile_packed_linear(m=8, n=256, k=256, reps=3, warmup=1)
    assert (8, 256, 256) in SOFT_BUDGETS_MS
    assert check_soft_budgets(br) == []


def test_soft_budgets_detect_violation():
    fake = {
        "m": 8,
        "n": 256,
        "k": 256,
        "gemm_ms": 1e9,
        "e2e_forward_ms": 1e9,
        "torch_fp32_ms": 1.0,
    }
    v = check_soft_budgets(fake)
    assert any("gemm_ms" in x for x in v)
    assert any("e2e_forward_ms" in x for x in v)


def test_committed_benchmark_has_thread_scaling():
    """W13.T04 — published curves live in results/benchmark.json (+ docs/34)."""
    root = Path(__file__).resolve().parents[1]
    bench = json.loads((root / "results" / "benchmark.json").read_text(encoding="utf-8"))
    # Absolute corruption floor + thread_scaling length — not a relative golden.
    assert check_committed_bench_soft_floors(bench) == []
    rows = bench.get("results") or []
    assert rows, "committed benchmark.json missing results"
    for r in rows:
        scaling = r.get("thread_scaling") or []
        assert len(scaling) >= 2
        threads = [row["threads"] for row in scaling]
        assert threads[0] == 1
        assert all(row["speedup_vs_1thread"] >= 0.5 for row in scaling)


def test_cli_profile(tmp_path: Path):
    out = tmp_path / "prof.json"
    assert (
        cli_main(
            [
                "profile",
                "--batch",
                "8",
                "--in-features",
                "256",
                "--out-features",
                "256",
                "--reps",
                "3",
                "--warmup",
                "1",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["n"] == 256
    assert "soft_budget_ok" in data
    assert data["soft_budget_ok"] is True
    assert data["torch_int8_wo_ms"] > 0
