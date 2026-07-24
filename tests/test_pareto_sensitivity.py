"""Pareto schema + sensitivity scoring tests (Phase C)."""

from __future__ import annotations

import torch
import torch.nn as nn

from bnn.eval.pareto import (
    PARETO_SCHEMA_ID,
    build_pareto_report,
    demo_points,
    validate_pareto_report,
)
from bnn.wrap.sensitivity import score_layer_sensitivity


def test_pareto_demo_validates():
    report = build_pareto_report(demo_points(), warmup=3, threads=1)
    assert report["schema"] == PARETO_SCHEMA_ID
    assert validate_pareto_report(report) == []


def test_sensitivity_smoke():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.ffn_fc1 = nn.Linear(64, 128)
            self.ffn_fc2 = nn.Linear(128, 64)
            self.head = nn.Linear(64, 10)

        def forward(self, x):
            return self.head(self.ffn_fc2(torch.relu(self.ffn_fc1(x))))

    m = Tiny()
    x = torch.randn(8, 64)
    rep = score_layer_sensitivity(
        m,
        x,
        mode="binary_xnor",
        policy="all_large_linear",
        min_in_features=32,
        drop_in_threshold=0.5,
        fragile_drop=0.5,
    )
    assert rep.layers
    d = rep.to_dict()
    assert "thesis_note" in d
    assert d["baseline_cosine"] >= 0.99


def test_sensitivity_respects_hybrid_allowlist():
    """sensitivity must not exit hybrid_ffn allowlist via empty skip_name_substr."""
    from bnn.optimise import optimise_model

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn_qkv = nn.Linear(64, 64)
            self.ffn_fc1 = nn.Linear(64, 128)
            self.ffn_fc2 = nn.Linear(128, 64)
            self.head = nn.Linear(64, 10)

        def forward(self, x):
            h = self.attn_qkv(x)
            return self.head(self.ffn_fc2(torch.relu(self.ffn_fc1(h))))

    m = Tiny()
    x = torch.randn(4, 64)
    result = optimise_model(
        m,
        x,
        policy="hybrid_ffn",
        mode="binary_xnor",
        min_in_features=32,
        sensitivity=True,
        force=True,
    )
    replaced = set(result.report.replaced)
    assert "attn_qkv" not in replaced
    assert "head" not in replaced
    assert result.payload.get("sensitivity") is not None
