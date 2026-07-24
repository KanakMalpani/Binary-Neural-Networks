"""Ultra wrap layer tests: policy, no re-pack, calib, report schema, native err."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from bnn.kernels.packed import binary_gemm_packed, fp32_gemm, native_kernel_available
from bnn.wrapper import (
    CalibConfig,
    PackedBinaryXNORLinear,
    WrapReport,
    calibrate_linear_scales,
    measure_agreement,
    recommend_wrap_policy,
    wrap_linear_modules,
    wrap_model,
)


def _named_ffn():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(64, 64)
            self.attn_qkv = nn.Linear(64, 64)
            self.ffn_fc1 = nn.Linear(64, 256)
            self.ffn_fc2 = nn.Linear(256, 64)
            self.lm_head = nn.Linear(64, 10)

        def forward(self, x):
            h = self.embed(x) + self.attn_qkv(x)
            return self.lm_head(self.ffn_fc2(torch.relu(self.ffn_fc1(h))))

    return Tiny()


def test_policy_hybrid_skips_attn_embed():
    m = _named_ffn()
    _, report = wrap_model(m, policy="hybrid_ffn", min_in_features=32)
    assert "ffn_fc1" in report.replaced
    assert "ffn_fc2" in report.replaced
    assert not any(r.startswith("embed") for r in report.replaced)
    assert not any("attn" in r for r in report.replaced)


def test_policy_ternary_wo_mode():
    m = _named_ffn()
    _, report = wrap_model(m, policy="ternary_wo", min_in_features=32)
    assert report.mode == "ternary_weight_only"
    assert report.replaced


def test_recommend_wrap_policy_returns_decision():
    lin = nn.Linear(1024, 1024)
    d = recommend_wrap_policy(lin)
    assert d.mode in ("binary_xnor", "ternary_weight_only")
    assert d.policy
    assert d.reason


def test_no_repack_weights_cached():
    w = torch.randn(128, 256)
    mod = PackedBinaryXNORLinear(w)
    assert mod._packed_once is True
    ptr_before = mod._wp_np.__array_interface__["data"][0]
    x = torch.randn(4, 256)
    _ = mod(x)
    ptr_after = mod._wp_np.__array_interface__["data"][0]
    assert ptr_before == ptr_after
    # state_dict round-trip keeps packed weights
    sd = mod.state_dict()
    assert "weight_packed_i64" in sd
    mod2 = PackedBinaryXNORLinear(torch.zeros_like(w))
    mod2.load_state_dict(sd)
    y1 = mod(x)
    y2 = mod2(x)
    assert torch.allclose(y1, y2)


def test_calib_per_channel_scales():
    w = torch.randn(32, 64)
    s = calibrate_linear_scales(w, cfg=CalibConfig(method="absmean", per_channel=True))
    assert s.shape == (32,)
    s2 = calibrate_linear_scales(w, cfg=CalibConfig(method="percentile", percentile=99.0, per_channel=True))
    assert s2.shape == (32,)


def test_wrap_report_schema():
    m = _named_ffn()
    _, report = wrap_model(m, policy="hybrid_ffn", min_in_features=32, calib=CalibConfig())
    d = report.to_dict()
    assert "compression" in d
    assert "replaced" in d
    assert "mode" in d
    assert "policy" in d
    assert isinstance(report, WrapReport)
    assert report.compression >= 15.0


def test_effectiveness_metrics():
    t = torch.randn(8, 10)
    s = t + 0.01 * torch.randn(8, 10)
    eff = measure_agreement(t, s, drop_in_threshold=0.85)
    assert eff.cosine > 0.9
    assert eff.drop_in_ok
    assert eff.top1_agreement is not None


def test_wrap_compression_binary():
    m = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
    _, report = wrap_linear_modules(
        m,
        mode="binary_xnor",
        skip_name_substr=(),
        min_in_features=100,
    )
    assert report.replaced
    assert report.compression >= 30.0


def test_native_gemm_err_zero_when_available():
    if not native_kernel_available():
        return
    rng = np.random.default_rng(0)
    x = np.where(rng.random((8, 256)) >= 0.5, 1.0, -1.0).astype(np.float32)
    w = np.where(rng.random((64, 256)) >= 0.5, 1.0, -1.0).astype(np.float32)
    y_bin = binary_gemm_packed(x, w)
    y_fp = fp32_gemm(x, w)
    err = float(np.max(np.abs(y_bin - y_fp)))
    assert err == 0.0


def test_aggressive_can_wrap_more_than_hybrid():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(128, 128)
            self.proj = nn.Linear(128, 128)  # not FFN name — hybrid skips
            self.ffn_fc1 = nn.Linear(128, 256)
            self.lm_head = nn.Linear(128, 10)

        def forward(self, x):
            return self.lm_head(self.ffn_fc1(self.proj(self.embed(x))))

    h = M()
    _, rh = wrap_model(h, policy="hybrid_ffn", mode="binary_xnor", min_in_features=64)
    a = M()
    _, ra = wrap_model(a, policy="aggressive", mode="binary_xnor", min_in_features=64)
    assert "ffn_fc1" in rh.replaced
    assert "proj" not in rh.replaced
    assert "proj" in ra.replaced
    assert "embed" not in ra.replaced
