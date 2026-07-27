#!/usr/bin/env python3
"""Tiny Transformer wrap lane: hybrid_ffn + calib + light QAT; agreement + size/speed."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bnn.codec import encode_file, roundtrip_gemm_err  # noqa: E402
from bnn.layers import BinaryLinear  # noqa: E402
from bnn.paths import repo_relative  # noqa: E402
from bnn.ste import clip_weights_  # noqa: E402
from bnn.wrap.api import wrap_model  # noqa: E402
from bnn.wrap.calibrate import CalibConfig  # noqa: E402
from bnn.wrap.metrics import measure_agreement  # noqa: E402
from bnn.wrapper import model_param_bytes  # noqa: E402


class TinyTransformer(nn.Module):
    """Realistic tiny Transformer block stack (names enable hybrid_ffn policy)."""

    def __init__(self, d: int = 128, ff: int = 512, depth: int = 2, n_classes: int = 10):
        super().__init__()
        self.embed = nn.Linear(64, d)
        self.blocks = nn.ModuleList()
        for _i in range(depth):
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "attn_qkv": nn.Linear(d, d * 3),
                        "attn_proj": nn.Linear(d, d),
                        "ffn_fc1": nn.Linear(d, ff),
                        "ffn_fc2": nn.Linear(ff, d),
                        "n1": nn.LayerNorm(d),
                        "n2": nn.LayerNorm(d),
                    }
                )
            )
        self.lm_head = nn.Linear(d, n_classes)
        self.d = d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 64) → treat as single token for wrap demos
        h = self.embed(x)
        for blk in self.blocks:
            # FP attn sketch (single token → trivial)
            h_n = blk["n1"](h)
            qkv = blk["attn_qkv"](h_n)
            q, k, v = qkv.chunk(3, dim=-1)
            # Single-vector "attention": softmax over feature as identity-ish skip
            attn = torch.softmax(q * k, dim=-1)
            h = h + blk["attn_proj"](attn * v)
            h_n = blk["n2"](h)
            h = h + blk["ffn_fc2"](F.relu(blk["ffn_fc1"](h_n)))
        return self.lm_head(h)


def _qat_ffn_(model: TinyTransformer, steps: int, x: torch.Tensor, y: torch.Tensor) -> float:
    # Replace FFN linears with BinaryLinear for STE steps
    for blk in model.blocks:
        d, ff = blk["ffn_fc1"].in_features, blk["ffn_fc1"].out_features
        b1 = BinaryLinear(d, ff, bias=True)
        b2 = BinaryLinear(ff, d, bias=True)
        with torch.no_grad():
            b1.weight.copy_(blk["ffn_fc1"].weight)
            b2.weight.copy_(blk["ffn_fc2"].weight)
            if blk["ffn_fc1"].bias is not None:
                b1.bias.copy_(blk["ffn_fc1"].bias)
            if blk["ffn_fc2"].bias is not None:
                b2.bias.copy_(blk["ffn_fc2"].bias)
        blk["ffn_fc1"] = b1
        blk["ffn_fc2"] = b2
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    t0 = time.perf_counter()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        clip_weights_(model)
    return time.perf_counter() - t0


def _restore_linears_from_binary_(model: TinyTransformer) -> None:
    for blk in model.blocks:
        for key in ("ffn_fc1", "ffn_fc2"):
            mod = blk[key]
            if isinstance(mod, BinaryLinear):
                lin = nn.Linear(mod.in_features, mod.out_features, bias=mod.bias is not None)
                with torch.no_grad():
                    lin.weight.copy_(mod.weight)
                    if mod.bias is not None and lin.bias is not None:
                        lin.bias.copy_(mod.bias)
                blk[key] = lin


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--ff", type=int, default=512)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--qat-steps", type=int, default=40)
    p.add_argument("--policy", default="hybrid_ffn")
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "tiny_transformer_wrap.json",
    )
    p.add_argument(
        "--pack-out",
        type=Path,
        default=ROOT / "results" / "_tiny_transformer.bnnpack",
    )
    args = p.parse_args(argv)

    torch.manual_seed(0)
    model = TinyTransformer(args.d_model, args.ff, args.depth)
    x = torch.randn(args.batch, 64)
    y = torch.randint(0, 10, (args.batch,))

    qat_s = 0.0
    if args.qat_steps > 0:
        qat_s = _qat_ffn_(model, args.qat_steps, x, y)
        _restore_linears_from_binary_(model)

    before = model_param_bytes(model)
    with torch.no_grad():
        y_fp = model(x).clone()

    calib = CalibConfig(method="absmean", per_channel=True)
    wrapped, report = wrap_model(
        model,
        policy=args.policy,
        min_in_features=32,
        calib=calib,
        inplace=True,
    )
    with torch.no_grad():
        y_w = wrapped(x)
    agr = measure_agreement(y_fp, y_w)

    # Encode packed FFN layers only (not attn/embed/head FP Linears)
    pack_path = encode_file(
        wrapped,
        args.pack_out,
        meta={"policy": args.policy, "source": "tiny_transformer_wrap"},
        min_in_features=32,
        include_packed=True,
        include_fp_linear=False,
        include_binary_linear=False,
    )
    # Round-trip on a synthetic wide Linear for GEMM err gate
    rt = roundtrip_gemm_err(torch.randn(256, 256))

    after = model_param_bytes(wrapped)
    # Timing
    for _ in range(5):
        wrapped(x)
    t0 = time.perf_counter()
    for _ in range(20):
        wrapped(x)
    wrap_ms = (time.perf_counter() - t0) / 20 * 1e3

    payload = {
        "model": "TinyTransformer",
        "d_model": args.d_model,
        "ff": args.ff,
        "depth": args.depth,
        "policy": args.policy,
        "qat_steps": args.qat_steps,
        "qat_seconds": qat_s,
        "replaced": report.replaced,
        "skipped": report.skipped,
        "compression_replaced": report.compression,
        "fp32_weight_bytes_replaced": report.fp32_weight_bytes_replaced,
        "packed_weight_bytes": report.packed_weight_bytes,
        "native_kernel": report.native_kernel,
        "agreement": agr.to_dict() if hasattr(agr, "to_dict") else dict(agr),
        "bytes_before": before,
        "bytes_after": after,
        "wrapped_forward_ms": wrap_ms,
        "bnnpack": repo_relative(pack_path),
        "codec_roundtrip": rt,
        "thesis": "hybrid_ffn wrap; attn FP; packed CPU binary — not fake GPU 32x",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
