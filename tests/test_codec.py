"""Codec encode/decode round-trip: compression 32×, GEMM err=0."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from bnn.cli import main as cli_main
from bnn.codec import (
    decode_file,
    encode_file,
    encode_linear_state,
    encode_model_linears,
    load_bnnpack,
    roundtrip_gemm_err,
)
from bnn.layers import BinaryLinear
from bnn.wrap.packed_linear import PackedBinaryXNORLinear


def test_encode_linear_compression_exact():
    w = torch.randn(128, 256)
    blob = encode_linear_state(w)
    assert abs(blob["compression"] - 32.0) < 1e-6
    assert blob["packed_bytes"] * 32 == blob["fp32_bytes"]


def test_roundtrip_gemm_err_zero():
    w = torch.randn(64, 128)
    stats = roundtrip_gemm_err(w, batch=8, seed=1)
    assert stats["max_abs_err"] == 0.0
    assert stats["compression"] == 32.0


def test_bnnpack_file_roundtrip(tmp_path: Path):
    model = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), BinaryLinear(64, 32))
    path = tmp_path / "toy.bnnpack"
    encode_file(model, path, meta={"test": True}, min_in_features=1)
    modules, meta = decode_file(path)
    assert meta.get("test") is True
    # Defaults: BinaryLinear only (not FP Linear stem)
    assert len(modules) == 1
    name, mod = next(iter(modules.items()))
    # nn.Sequential names children by index; BinaryLinear is child 2.
    assert name.endswith("2")
    x = torch.randn(4, mod.in_features)
    y = mod(x)
    assert y.shape == (4, mod.out_features)


def test_cli_encode_decode(tmp_path: Path):
    pack = tmp_path / "cli.bnnpack"
    assert (
        cli_main(
            [
                "encode",
                "--source",
                "random",
                "--in-features",
                "128",
                "--out-features",
                "64",
                "--out",
                str(pack),
            ]
        )
        == 0
    )
    assert pack.is_file()
    assert cli_main(["decode", "--pack", str(pack)]) == 0


def test_decode_rejects_bad_magic(tmp_path: Path):
    path = tmp_path / "bad.bnnpack"
    torch.save({"magic": "NOPE", "version": 1, "layers": {}}, path)
    with pytest.raises(ValueError) as excinfo:
        load_bnnpack(path)
    msg = str(excinfo.value)
    assert "magic" in msg.lower() or "BNNPACK" in msg


def test_encode_skips_attn_when_fp_linear_off():
    """Post-wrap style: only PackedBinaryXNORLinear / BinaryLinear encoded."""

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(64, 64)
            self.attn_qkv = nn.Linear(64, 192)
            self.ffn_fc1 = PackedBinaryXNORLinear(torch.randn(128, 64), None)
            self.lm_head = nn.Linear(64, 10)

    m = Toy()
    layers = encode_model_linears(m, include_fp_linear=False, include_packed=True)
    assert "ffn_fc1" in layers
    assert "embed" not in layers
    assert "attn_qkv" not in layers
    assert "lm_head" not in layers


def test_compression_padded_below_32():
    w = torch.randn(16, 100)  # 100 % 64 != 0
    blob = encode_linear_state(w)
    assert blob["compression"] < 32.0
