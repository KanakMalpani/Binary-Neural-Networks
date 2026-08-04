"""Codec encode/decode round-trip: compression 32×, GEMM err=0, v2 + safetensors."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from bnn.cli import main as cli_main
from bnn.codec import (
    BNNPACK_VERSION_V1,
    BNNPACK_VERSION_V2,
    decode_file,
    decode_to_packed_conv,
    decode_to_ternary_linear,
    encode_conv_state,
    encode_file,
    encode_linear_state,
    encode_model_linears,
    encode_ternary_state,
    export_bnnpack_safetensors,
    load_bnnpack,
    roundtrip_gemm_err,
    save_bnnpack,
    verify_layer_hashes,
)
from bnn.layers import BinaryLinear
from bnn.wrap.packed_linear import PackedBinaryConv2d, PackedBinaryXNORLinear


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
    payload = load_bnnpack(pack)
    assert int(payload["version"]) == BNNPACK_VERSION_V2
    assert "hashes" in payload


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


def test_v2_hashes_roundtrip_and_verify(tmp_path: Path):
    blob = encode_linear_state(torch.randn(32, 64), name="fc")
    path = tmp_path / "v2.bnnpack"
    save_bnnpack({"fc": blob}, path, meta={"schema": "v2"}, version=BNNPACK_VERSION_V2)
    payload = load_bnnpack(path)
    assert payload["version"] == BNNPACK_VERSION_V2
    assert payload["hashes"]["fc"] == blob["content_sha256"]
    assert verify_layer_hashes(payload) == []


def test_v1_still_loads(tmp_path: Path):
    blob = encode_linear_state(torch.randn(16, 64), name="fc", with_hash=False)
    path = tmp_path / "v1.bnnpack"
    save_bnnpack({"fc": blob}, path, version=BNNPACK_VERSION_V1)
    payload = load_bnnpack(path)
    assert payload["version"] == BNNPACK_VERSION_V1
    assert "hashes" not in payload
    modules, _ = decode_file(path)
    assert isinstance(modules["fc"], PackedBinaryXNORLinear)


def test_ternary_encode_decode_roundtrip():
    torch.manual_seed(0)
    w = torch.randn(24, 48)
    blob = encode_ternary_state(w, per_channel=True, name="t")
    assert blob["kind"] == "ternary_weight_only"
    assert blob["compression_kind"] == "theoretical_2bit"
    mod = decode_to_ternary_linear(blob)
    x = torch.randn(3, 48)
    y = mod(x)
    assert y.shape == (3, 24)
    assert torch.isfinite(y).all()


def test_conv_encode_decode_and_state_dict(tmp_path: Path):
    torch.manual_seed(1)
    conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
    blob = encode_conv_state(
        conv.weight.data, conv.bias.data, stride=1, padding=1, name="c0"
    )
    assert blob["kind"] == "binary_conv_packed"
    assert "content_sha256" in blob
    mod = decode_to_packed_conv(blob)
    x = torch.randn(2, 3, 12, 12)
    y0 = mod(x)
    assert y0.shape == (2, 8, 12, 12)

    # Packed buffer round-trip via state_dict (W5.T09 polish).
    dst = PackedBinaryConv2d(
        torch.randn(8, 3, 3, 3), torch.zeros(8), stride=1, padding=1
    )
    dst.load_state_dict(mod.state_dict())
    assert torch.allclose(dst(x), y0, atol=1e-5)

    path = tmp_path / "conv.bnnpack"
    save_bnnpack({"c0": blob}, path)
    modules, _ = decode_file(path)
    assert isinstance(modules["c0"], PackedBinaryConv2d)


def test_packed_conv_does_not_mutate_input():
    mod = PackedBinaryConv2d(torch.randn(4, 3, 3, 3), None, padding=1)
    x = torch.randn(1, 3, 8, 8)
    x_before = x.clone()
    _ = mod(x)
    assert torch.equal(x, x_before)


def test_safetensors_export_roundtrip(tmp_path: Path):
    pytest.importorskip("safetensors")
    from safetensors.torch import load_file

    blob = encode_linear_state(torch.randn(32, 64), name="linear")
    pack = tmp_path / "m.bnnpack"
    save_bnnpack({"linear": blob}, pack)
    st_path = tmp_path / "m.safetensors"
    out, meta_path = export_bnnpack_safetensors(pack, st_path)
    assert out.is_file() and meta_path.is_file()
    tensors = load_file(str(out))
    assert "linear.weight_packed_i64" in tensors
    assert tensors["linear.weight_packed_i64"].shape == blob["weight_packed_i64"].shape
    meta_txt = meta_path.read_text(encoding="utf-8")
    assert "bnnpack_safetensors_v1" in meta_txt
    assert "content_sha256" in meta_txt


def test_unsupported_version_rejected(tmp_path: Path):
    path = tmp_path / "badver.bnnpack"
    torch.save({"magic": "BNNPACK1", "version": 99, "layers": {}}, path)
    with pytest.raises(ValueError, match="unsupported version"):
        load_bnnpack(path)
