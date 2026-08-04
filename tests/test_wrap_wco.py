"""W3.T01 / T02 / T03 / T04 — calibrate unify, effectiveness, policy, drop-in."""

from __future__ import annotations

import torch
import torch.nn as nn

from bnn.wrap import (
    CalibConfig,
    CalibReport,
    attach_effectiveness,
    calibrate,
    calibrate_model,
    drop_in_ok,
    measure_agreement,
    recommend_wrap_policy,
    unmeasured_effectiveness,
    wrap_model,
)


class TinyFFN(nn.Module):
    def __init__(self, d: int = 64):
        super().__init__()
        self.embed = nn.Linear(d, d)
        self.ffn_fc1 = nn.Linear(d, d * 2)
        self.ffn_fc2 = nn.Linear(d * 2, d)
        self.lm_head = nn.Linear(d, 10)

    def forward(self, x):
        h = torch.relu(self.embed(x))
        h = self.ffn_fc2(torch.relu(self.ffn_fc1(h)))
        return self.lm_head(h)


def test_calibrate_tensor_matches_linear_scales():
    w = torch.randn(16, 32)
    a = calibrate(w, CalibConfig(method="absmean", per_channel=True))
    assert a.shape == (16,)
    assert torch.isfinite(a).all()


def test_calibrate_model_returns_report():
    m = TinyFFN()
    report = calibrate(m, CalibConfig(), policy="hybrid_ffn", min_in_features=32)
    assert isinstance(report, CalibReport)
    names = {L.name for L in report.layers}
    assert "ffn_fc1" in names
    assert "ffn_fc2" in names
    assert "embed" not in names
    d = report.to_dict()
    assert d["n_layers"] == len(report.layers)
    assert report.scales_by_name()["ffn_fc1"].numel() > 0


def test_calibrate_model_all_large_policy():
    m = TinyFFN(d=128)
    report = calibrate_model(m, policy="all_large_linear", min_in_features=64)
    assert len(report.layers) >= 2


def test_wrap_report_always_has_effectiveness():
    m = TinyFFN()
    _, report = wrap_model(m, policy="hybrid_ffn", min_in_features=32)
    assert report.effectiveness is not None
    assert report.effectiveness.get("measured") is False
    assert report.drop_in_ok is False


def test_attach_effectiveness_overwrites_stub():
    m = TinyFFN()
    x = torch.randn(4, 64)
    teacher = TinyFFN()
    with torch.no_grad():
        t = teacher(x)
    _, report = wrap_model(m, policy="hybrid_ffn", min_in_features=32)
    with torch.no_grad():
        s = m(x)
    eff = measure_agreement(t, s, drop_in_threshold=0.5)
    attach_effectiveness(report, eff)
    assert report.effectiveness["measured"] is True
    assert report.effectiveness["cosine"] is not None


def test_unmeasured_refuses_drop_in_without_force():
    stub = unmeasured_effectiveness()
    assert drop_in_ok(stub) is False
    assert drop_in_ok(stub, force=True) is True
    d = stub.to_dict()
    assert d["measured"] is False
    assert d["cosine"] is None  # NaN serialised as null for JSON honesty


def test_swap_and_restore_preserve_device_dtype():
    """Linear↔BinaryLinear replacements must match source device/dtype."""
    from bnn.wrap.qat import _restore_binary_to_linear, _swap_linear_to_binary

    lin = nn.Linear(6, 3, bias=True)
    bl = _swap_linear_to_binary(lin)
    assert bl.weight.device == lin.weight.device
    assert bl.weight.dtype == lin.weight.dtype
    restored = _restore_binary_to_linear(bl)
    assert restored.weight.device == bl.weight.device
    assert restored.weight.dtype == bl.weight.dtype
    assert torch.allclose(restored.weight, bl.weight)


def test_wrap_always_sets_policy_reason():
    m = TinyFFN()
    _, report = wrap_model(m, policy="hybrid_ffn", min_in_features=32)
    assert isinstance(report.policy_reason, str)
    assert len(report.policy_reason) > 0
    assert "hybrid_ffn" in report.policy_reason


def test_auto_policy_reason_from_recommender():
    m = TinyFFN(d=128)
    _, report = wrap_model(m, policy="auto", min_in_features=64)
    assert report.policy_reason
    assert len(report.policy_reason) > 8
    d = recommend_wrap_policy(None)
    assert d.reason


def test_drop_in_ok_when_cosine_meets_threshold():
    t = torch.randn(8, 10)
    s = t + 0.001 * torch.randn(8, 10)
    eff = measure_agreement(t, s, drop_in_threshold=0.85)
    assert eff.cosine >= 0.85
    assert eff.drop_in_ok is True
    assert drop_in_ok(eff) is True


def test_drop_in_refused_below_threshold():
    torch.manual_seed(0)
    t = torch.randn(8, 10)
    s = torch.randn(8, 10)
    eff = measure_agreement(t, s, drop_in_threshold=0.99)
    assert eff.drop_in_ok is False
    assert drop_in_ok(eff) is False
    assert "Below drop-in" in eff.notes or "threshold" in eff.notes.lower()


def test_force_overrides_drop_in_refusal():
    t = torch.randn(8, 10)
    s = -t
    eff = measure_agreement(t, s, drop_in_threshold=0.99)
    assert eff.drop_in_ok is False
    m = TinyFFN()
    _, report = wrap_model(m, policy="hybrid_ffn", min_in_features=32)
    attach_effectiveness(report, eff, force=True)
    assert report.drop_in_ok is True
    assert report.forced is True
