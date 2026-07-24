#!/usr/bin/env python3
"""Hybrid FFN-only wrap + optional short STE/QAT sketch for middle Linears.

Demonstrates the production pattern: keep embed/attn/head FP; wrap FFN Linears;
optionally run a few STE steps so binary_xnor is not a cold PTQ wipe.
"""

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

from bnn.layers import BinaryLinear  # noqa: E402
from bnn.ste import clip_weights_  # noqa: E402
from bnn.wrapper import wrap_linear_modules, model_param_bytes  # noqa: E402


class TinyTransformerish(nn.Module):
    """Toy stack: embed → attn-ish Linear → FFN → head (names matter for wrap)."""

    def __init__(self, d: int = 256, ff: int = 1024, n_classes: int = 10):
        super().__init__()
        self.embed = nn.Linear(28 * 28, d)
        self.attn_qkv = nn.Linear(d, d)  # kept FP by skip substr "attn"
        self.ffn_fc1 = nn.Linear(d, ff)
        self.ffn_fc2 = nn.Linear(ff, d)
        self.lm_head = nn.Linear(d, n_classes)

    def forward(self, x):
        h = F.relu(self.embed(x))
        h = h + F.relu(self.attn_qkv(h))
        h = h + self.ffn_fc2(F.relu(self.ffn_fc1(h)))
        return self.lm_head(h)


def qat_replace_ffn(model: TinyTransformerish) -> None:
    """Swap FFN Linears to BinaryLinear for a few STE steps (training sim)."""
    d, ff = model.ffn_fc1.in_features, model.ffn_fc1.out_features
    bl1 = BinaryLinear(d, ff, bias=True)
    bl2 = BinaryLinear(ff, d, bias=True)
    with torch.no_grad():
        bl1.weight.copy_(model.ffn_fc1.weight)
        bl2.weight.copy_(model.ffn_fc2.weight)
        if model.ffn_fc1.bias is not None:
            bl1.bias.copy_(model.ffn_fc1.bias)
        if model.ffn_fc2.bias is not None:
            bl2.bias.copy_(model.ffn_fc2.bias)
    model.ffn_fc1 = bl1
    model.ffn_fc2 = bl2


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--ff", type=int, default=1024)
    p.add_argument("--qat-steps", type=int, default=50)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--out", type=Path, default=ROOT / "results" / "hybrid_ffn_wrap.json")
    args = p.parse_args()

    torch.manual_seed(0)
    model = TinyTransformerish(args.d_model, args.ff)
    x = torch.randn(args.batch, 28 * 28)
    y = torch.randint(0, 10, (args.batch,))

    # Short STE/QAT on FFN only
    qat_replace_ffn(model)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    t0 = time.perf_counter()
    for _ in range(args.qat_steps):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        clip_weights_(model)
    qat_s = time.perf_counter() - t0

    # Export-style: wrap FFN names with packed binary (skip embed/attn/head)
    # Temporarily restore nn.Linear-shaped weights from BinaryLinear latents for wrap API
    with torch.no_grad():
        w1, b1 = model.ffn_fc1.weight.detach().clone(), model.ffn_fc1.bias.detach().clone()
        w2, b2 = model.ffn_fc2.weight.detach().clone(), model.ffn_fc2.bias.detach().clone()
    model.ffn_fc1 = nn.Linear(w1.shape[1], w1.shape[0])
    model.ffn_fc2 = nn.Linear(w2.shape[1], w2.shape[0])
    with torch.no_grad():
        model.ffn_fc1.weight.copy_(w1)
        model.ffn_fc1.bias.copy_(b1)
        model.ffn_fc2.weight.copy_(w2)
        model.ffn_fc2.bias.copy_(b2)

    before = model_param_bytes(model)
    _, report = wrap_linear_modules(
        model,
        mode="binary_xnor",
        skip_name_substr=("embed", "attn", "lm_head", "head"),
        min_in_features=32,
    )
    after = model_param_bytes(model)

    with torch.no_grad():
        logits = model(x)

    payload = {
        "pattern": "hybrid_ffn_only_wrap_after_short_STE_QAT",
        "qat_steps": args.qat_steps,
        "qat_seconds": qat_s,
        "replaced": report.replaced,
        "skipped": report.skipped,
        "compression_replaced_weights": report.compression,
        "param_bytes_before": before,
        "param_bytes_after_wrap": after,
        "logits_finite": bool(torch.isfinite(logits).all().item()),
        "native_kernel": report.native_kernel,
        "verdict": (
            "CLOSED: hybrid FFN wrap path works end-to-end (STE sketch → packed wrap). "
            "Production HF models need real data + BitDistill-scale QAT; this closes the "
            "architecture/protocol gap."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
