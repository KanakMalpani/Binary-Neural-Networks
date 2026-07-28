"""Per-layer binary / ternary / skip search (W3.T06)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from bnn.wrap.sensitivity import (
    ModeSearchReport,
    _relax,
    search_layer_modes,
)


class Net(nn.Module):
    """Names match the ffn heuristic so several layers are eligible."""

    def __init__(self, d: int = 128):
        super().__init__()
        self.ffn_fc1 = nn.Linear(d, d * 2)
        self.ffn_fc2 = nn.Linear(d * 2, d)
        self.head = nn.Linear(d, 64)

    def forward(self, x):
        return self.head(torch.relu(self.ffn_fc2(torch.relu(self.ffn_fc1(x)))))


@pytest.fixture
def calib():
    torch.manual_seed(0)
    return torch.randn(16, 128)


def test_relax_ladder_terminates():
    assert _relax("binary_xnor") == "ternary_weight_only"
    assert _relax("ternary_weight_only") == "skip"
    assert _relax("skip") is None


def test_zero_floor_keeps_everything_binary(calib):
    """With no quality requirement the search must stay maximally aggressive."""
    r = search_layer_modes(Net(), calib, quality_floor=0.0)
    assert r.met_floor
    assert r.ternary == [] and r.skipped == []
    assert len(r.binary) == 3
    assert r.compression() == pytest.approx(32.0)
    assert r.probes == 1, "no relaxation needed, so only the initial probe"


def test_impossible_floor_relaxes_everything(calib):
    """A floor only FP32 can meet must end with every layer skipped."""
    r = search_layer_modes(Net(), calib, quality_floor=0.999)
    assert r.skipped
    assert r.compression() == pytest.approx(1.0)
    assert r.final_cosine >= 0.999


def test_intermediate_floor_produces_a_mixed_assignment(calib):
    r = search_layer_modes(Net(), calib, quality_floor=0.90)
    assert r.met_floor
    assert r.final_cosine >= 0.90
    # Something was relaxed, but not necessarily everything.
    assert len(r.binary) < 3
    assert 1.0 <= r.compression() <= 32.0


def test_compression_decreases_monotonically_with_floor(calib):
    """Higher quality must never be cheaper — that would mean the search lies."""
    results = [
        search_layer_modes(Net(), calib, quality_floor=f) for f in (0.0, 0.90, 0.999)
    ]
    compressions = [r.compression() for r in results]
    assert compressions == sorted(compressions, reverse=True), compressions
    cosines = [r.final_cosine for r in results]
    assert cosines == sorted(cosines), cosines


def test_search_is_cheaper_than_exhaustive(calib):
    """O(L) relaxations, not 3**L combinations."""
    r = search_layer_modes(Net(), calib, quality_floor=0.95)
    assert r.probes < 3**3, "search degenerated to brute force"


def test_max_relaxations_caps_the_work(calib):
    r = search_layer_modes(Net(), calib, quality_floor=0.999, max_relaxations=1)
    # One relaxation cannot reach a near-perfect floor; must report honestly.
    assert r.met_floor is False
    assert r.final_cosine < 0.999


def test_report_serialises_with_dual_metric_note(calib):
    r = search_layer_modes(Net(), calib, quality_floor=0.90)
    d = r.to_dict()
    for key in ("baseline_cosine", "final_cosine", "quality_floor", "met_floor",
                "binary", "ternary", "skipped", "assignments", "probes"):
        assert key in d
    assert "theoretical" in d["thesis_note"].lower()
    assert len(d["assignments"]) == 3
    assert {a["mode"] for a in d["assignments"]} <= {
        "binary_xnor", "ternary_weight_only", "skip"
    }


def test_every_assignment_carries_a_reason(calib):
    r = search_layer_modes(Net(), calib, quality_floor=0.90)
    for a in r.assignments:
        assert a.reason
        assert a.weight_elems > 0
        assert a.packed_bytes > 0


def test_model_without_eligible_linears_returns_empty_report():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4)  # below min_in_features

        def forward(self, x):
            return self.fc(x)

    r = search_layer_modes(Tiny(), torch.randn(2, 4), quality_floor=0.9)
    assert isinstance(r, ModeSearchReport)
    assert r.assignments == []
    assert r.compression() == 1.0


def test_search_does_not_mutate_the_input_model(calib):
    """Probing must work on copies; the caller's model stays FP32."""
    model = Net()
    before = model.ffn_fc1.weight.detach().clone()
    search_layer_modes(model, calib, quality_floor=0.90)
    assert isinstance(model.ffn_fc1, nn.Linear)
    assert torch.equal(model.ffn_fc1.weight.detach(), before)
