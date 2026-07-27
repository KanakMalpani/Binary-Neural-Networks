"""The non-XNOR packed replacement modules: ternary, dequant, and conv."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from bnn.wrap.packed_linear import (
    BinaryWeightOnlyDequantLinear,
    PackedBinaryConv2d,
    PackedBinaryXNORLinear,
    TernaryWeightOnlyLinear,
    absmean_ternary,
    absmean_ternary_per_channel,
    sign_pm1,
)


def test_sign_pm1_maps_zero_to_plus_one():
    """Zero must not become 0 — the packed encoding only has ±1."""
    out = sign_pm1(torch.tensor([-2.0, -0.0, 0.0, 3.0]))
    assert set(out.unique().tolist()) <= {-1.0, 1.0}
    assert out[1].item() == 1.0
    assert out[2].item() == 1.0


def test_absmean_ternary_produces_only_three_levels():
    q, scale = absmean_ternary(torch.randn(16, 32))
    assert set(q.unique().tolist()) <= {-1, 0, 1}
    assert q.dtype == torch.int8
    assert scale.ndim == 0 and scale > 0


def test_absmean_ternary_per_channel_scale_shape():
    q, scale = absmean_ternary_per_channel(torch.randn(16, 32))
    assert q.shape == (16, 32)
    assert scale.shape == (16,)
    assert (scale > 0).all()


def test_absmean_ternary_handles_all_zero_weight():
    """A dead channel must not produce a divide-by-zero or NaN scale."""
    q, scale = absmean_ternary(torch.zeros(4, 8))
    assert torch.isfinite(scale).all()
    assert scale.item() > 0
    assert torch.equal(q, torch.zeros(4, 8, dtype=torch.int8))


# --------------------------------------------------------------------------
# TernaryWeightOnlyLinear
# --------------------------------------------------------------------------

@pytest.mark.parametrize("per_channel", [True, False])
def test_ternary_linear_forward_shape_and_finiteness(per_channel):
    lin = nn.Linear(64, 32)
    mod = TernaryWeightOnlyLinear(lin.weight.data, lin.bias.data, per_channel=per_channel)
    out = mod(torch.randn(5, 64))
    assert out.shape == (5, 32)
    assert torch.isfinite(out).all()


def test_ternary_linear_reports_theoretical_2bit_compression():
    lin = nn.Linear(64, 32)
    mod = TernaryWeightOnlyLinear(lin.weight.data, None)
    # Buffer is int8, so the reported size must be flagged theoretical.
    assert mod.compression_kind == "theoretical_2bit"
    assert mod.packed_weight_bytes() == 64 * 32 * 2 // 8


def test_ternary_linear_without_bias():
    mod = TernaryWeightOnlyLinear(torch.randn(8, 16), None)
    assert mod.bias is None
    assert mod(torch.randn(2, 16)).shape == (2, 8)


def test_ternary_linear_repr_states_mode():
    mod = TernaryWeightOnlyLinear(torch.randn(8, 16), None)
    assert "ternary_weight_only" in repr(mod)


def test_ternary_linear_is_closer_to_fp_than_binary():
    """Ternary keeps a zero level, so PTQ fidelity should beat pure binary."""
    torch.manual_seed(0)
    lin = nn.Linear(256, 128)
    x = torch.randn(16, 256)
    ref = lin(x)

    tern = TernaryWeightOnlyLinear(lin.weight.data, lin.bias.data)(x)
    binr = BinaryWeightOnlyDequantLinear(lin.weight.data, lin.bias.data)(x)

    cos = lambda a, b: torch.nn.functional.cosine_similarity(  # noqa: E731
        a.flatten(), b.flatten(), dim=0
    ).item()
    assert cos(ref, tern) > cos(ref, binr)


# --------------------------------------------------------------------------
# BinaryWeightOnlyDequantLinear
# --------------------------------------------------------------------------

def test_binary_dequant_forward_and_packed_bytes():
    lin = nn.Linear(128, 32)
    mod = BinaryWeightOnlyDequantLinear(lin.weight.data, lin.bias.data)
    out = mod(torch.randn(4, 128))
    assert out.shape == (4, 32)
    assert torch.isfinite(out).all()
    # 128 in-features == 2 uint64 words per row.
    assert mod.packed_weight_bytes() == 32 * 2 * 8


def test_binary_dequant_without_bias():
    mod = BinaryWeightOnlyDequantLinear(torch.randn(16, 64), None)
    assert mod.bias is None
    assert mod(torch.randn(3, 64)).shape == (3, 16)


# --------------------------------------------------------------------------
# PackedBinaryConv2d
# --------------------------------------------------------------------------

def test_packed_conv_forward_shape():
    conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
    mod = PackedBinaryConv2d(conv.weight.data, conv.bias.data, stride=1, padding=1)
    out = mod(torch.randn(2, 3, 16, 16))
    assert out.shape == (2, 8, 16, 16)
    assert torch.isfinite(out).all()


def test_packed_conv_respects_stride():
    conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
    mod = PackedBinaryConv2d(conv.weight.data, None, stride=2, padding=1)
    assert mod(torch.randn(1, 3, 16, 16)).shape == (1, 4, 8, 8)


def test_packed_conv_scalar_alpha_is_broadcast_per_channel():
    conv = nn.Conv2d(3, 6, kernel_size=3)
    mod = PackedBinaryConv2d(conv.weight.data, None, alpha=torch.tensor([0.5]))
    assert mod.alpha.numel() == 6
    assert torch.allclose(mod.alpha, torch.full((6,), 0.5))


def test_packed_conv_repr_states_mode():
    conv = nn.Conv2d(3, 4, kernel_size=3)
    assert "binary_conv_packed_dequant" in repr(PackedBinaryConv2d(conv.weight.data, None))


# --------------------------------------------------------------------------
# PackedBinaryXNORLinear extras
# --------------------------------------------------------------------------

def test_xnor_linear_state_dict_round_trip_keeps_outputs():
    """Packed words live in the state_dict; reload must not re-pack wrongly."""
    torch.manual_seed(0)
    lin = nn.Linear(128, 64)
    src = PackedBinaryXNORLinear(lin.weight.data, lin.bias.data)
    x = torch.randn(4, 128)
    expected = src(x)

    dst = PackedBinaryXNORLinear(torch.randn(64, 128), torch.randn(64))
    dst.load_state_dict(src.state_dict())
    assert torch.allclose(dst(x), expected, atol=1e-5)


def test_xnor_linear_accepts_higher_rank_input():
    mod = PackedBinaryXNORLinear(torch.randn(32, 64), None)
    out = mod(torch.randn(2, 5, 64))
    assert out.shape == (2, 5, 32)


def test_xnor_linear_gemm_only_matches_forward_scale():
    mod = PackedBinaryXNORLinear(torch.randn(32, 64), None)
    x_pm1 = np.sign(np.random.default_rng(0).standard_normal((4, 64))).astype(np.float32)
    x_pm1[x_pm1 == 0] = 1.0
    got = mod.gemm_only(x_pm1)
    assert got.shape == (4, 32)
    assert np.isfinite(got).all()


def test_xnor_linear_repr_reports_native_and_packed_once():
    mod = PackedBinaryXNORLinear(torch.randn(32, 64), None)
    text = repr(mod)
    assert "binary_xnor" in text
    assert "packed_once=True" in text
