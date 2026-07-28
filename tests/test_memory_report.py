"""Memory footprint accounting (W13.T05).

The point of these tests is that the report cannot overclaim: resident bytes are
measured from real buffers, theoretical bytes are the encoding ratio, and the
whole-model number must include the FP parts that were deliberately not wrapped.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from bnn.memory import forward_transient_bytes, memory_report
from bnn.wrap import wrap_model
from bnn.wrap.packed_linear import PackedBinaryXNORLinear, TernaryWeightOnlyLinear


def _mlp(dim: int = 512, ff: int = 2048) -> nn.Module:
    return nn.Sequential(nn.Linear(dim, ff), nn.ReLU(), nn.Linear(ff, dim))


def test_fp32_model_reports_no_packed_layers():
    r = memory_report(_mlp()).to_dict()
    assert r["packed_layer_count"] == 0
    assert r["layer_count"] == 2
    assert r["tracked_resident_compression"] == pytest.approx(1.0, rel=0.02)


def test_wrapping_reduces_measured_resident_bytes():
    fp32 = memory_report(_mlp()).to_dict()
    wrapped, _ = wrap_model(_mlp(), mode="binary_xnor", policy="all_large_linear")
    packed = memory_report(wrapped).to_dict()
    assert packed["tracked_resident_bytes"] < fp32["tracked_resident_bytes"]
    assert packed["packed_layer_count"] == 2


def test_resident_never_beats_theoretical():
    """Measured savings cannot exceed what the encoding allows — that would be a bug."""
    wrapped, _ = wrap_model(_mlp(), mode="binary_xnor", policy="all_large_linear")
    r = memory_report(wrapped).to_dict()
    assert r["tracked_resident_compression"] <= r["tracked_theoretical_compression"] + 1e-9
    # And the gap is real: alpha/bias stay FP32, so resident < 32x.
    assert r["tracked_theoretical_compression"] == pytest.approx(32.0, rel=0.01)
    assert r["tracked_resident_compression"] < 32.0


def test_whole_model_ratio_includes_unwrapped_parts():
    """An embedding kept in FP32 must drag the end-to-end number down."""

    class WithEmbed(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(4000, 512)  # large, never wrapped
            self.ffn_fc1 = nn.Linear(512, 2048)
            self.ffn_fc2 = nn.Linear(2048, 512)

        def forward(self, x):
            return self.ffn_fc2(torch.relu(self.ffn_fc1(x)))

    wrapped, _ = wrap_model(WithEmbed(), mode="binary_xnor", policy="hybrid_ffn")
    r = memory_report(wrapped).to_dict()
    assert r["other_param_bytes"] > 0, "embedding must be counted as unwrapped"
    assert r["whole_model_resident_compression"] < r["tracked_resident_compression"]


def test_ternary_reports_theoretical_2bit_not_int8_storage():
    """TernaryWeightOnlyLinear stores int8; the report must not claim 2-bit resident."""
    lin = nn.Linear(512, 256)
    mod = TernaryWeightOnlyLinear(lin.weight.data, lin.bias.data)
    r = memory_report(mod).to_dict()
    layer = r["layers"][0]
    assert layer["theoretical_bytes"] < layer["resident_bytes"]
    assert layer["theoretical_compression"] > layer["resident_compression"]


def test_per_layer_entries_are_complete():
    wrapped, _ = wrap_model(_mlp(), mode="binary_xnor", policy="all_large_linear")
    for layer in memory_report(wrapped).to_dict()["layers"]:
        assert layer["name"]
        assert layer["kind"]
        assert layer["resident_bytes"] > 0
        assert layer["fp32_equivalent_bytes"] > 0
        assert layer["packed"] is True


def test_totals_reconcile_with_torch_parameter_accounting():
    """tracked + other must equal what torch says the model weighs."""
    wrapped, _ = wrap_model(_mlp(), mode="binary_xnor", policy="all_large_linear")
    r = memory_report(wrapped)
    torch_total = sum(p.numel() * p.element_size() for p in wrapped.parameters()) + sum(
        b.numel() * b.element_size() for b in wrapped.buffers()
    )
    assert r.totals()["model_resident_bytes"] == torch_total


def test_report_carries_the_dual_metric_note():
    note = memory_report(_mlp()).to_dict()["thesis_note"].lower()
    assert "measured" in note
    assert "latency" in note


def test_conv_footprint_is_tracked():
    conv = nn.Conv2d(8, 16, kernel_size=3)
    r = memory_report(conv).to_dict()
    assert r["layer_count"] == 1
    assert r["layers"][0]["fp32_equivalent_bytes"] == 8 * 16 * 3 * 3 * 4


# --------------------------------------------------------------------------
# transient buffers
# --------------------------------------------------------------------------

def test_transient_activation_pack_is_32x():
    t = forward_transient_bytes(64, 4096, 4096)
    assert t["activation_pack_compression"] == pytest.approx(32.0)
    assert t["packed_activation_bytes"] == 64 * (4096 // 64) * 8


def test_transient_padding_for_unaligned_in_features():
    """100 features still costs 2 uint64 words per row."""
    t = forward_transient_bytes(4, 100, 8)
    assert t["packed_activation_bytes"] == 4 * 2 * 8
    assert t["activation_pack_compression"] < 32.0


def test_transient_total_is_dominated_by_the_fp32_output():
    """Honest sizing note: packing activations is cheap, the output is not."""
    t = forward_transient_bytes(64, 4096, 4096)
    assert t["output_bytes"] > t["packed_activation_bytes"]
    assert t["total_transient_bytes"] == t["output_bytes"] + t["packed_activation_bytes"]


def test_packed_linear_module_alone_reports_sane_numbers():
    mod = PackedBinaryXNORLinear(torch.randn(256, 512), None)
    layer = memory_report(mod).to_dict()["layers"][0]
    assert layer["packed"] is True
    assert layer["fp32_equivalent_bytes"] == 256 * 512 * 4
    assert layer["theoretical_bytes"] == 256 * (512 // 64) * 8
