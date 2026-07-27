#!/usr/bin/env python3
"""Train / eval binary Encoder–Decoder on synthetic reverse seq2seq."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bnn.seq import (  # noqa: E402
    BinaryAutoEncoder,
    BinarySeq2Seq,
    make_reverse_batch,
    seq2seq_token_accuracy,
)


def train_seq2seq(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    model = BinarySeq2Seq(
        vocab=args.vocab,
        dim=args.dim,
        depth=args.depth,
        n_heads=args.heads,
        ff=args.ff,
        max_len=args.seq_len + 2,
        ffn_kind=args.ffn,
    )
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    t0 = time.perf_counter()
    last_acc = 0.0
    last_loss = 0.0
    for step in range(args.steps):
        src, tgt_in, tgt = make_reverse_batch(
            args.batch, args.seq_len, args.vocab, seed=args.seed + step
        )
        opt.zero_grad(set_to_none=True)
        logits = model(src, tgt_in)
        loss = F.cross_entropy(logits.reshape(-1, args.vocab), tgt.reshape(-1))
        loss.backward()
        opt.step()
        model.clip_weights()
        last_loss = float(loss.item())
        last_acc = seq2seq_token_accuracy(logits.detach(), tgt)
    train_s = time.perf_counter() - t0

    # Eval holdout
    model.eval()
    with torch.no_grad():
        src, tgt_in, tgt = make_reverse_batch(
            args.batch * 2, args.seq_len, args.vocab, seed=args.seed + 9999
        )
        logits = model(src, tgt_in)
        eval_acc = seq2seq_token_accuracy(logits, tgt)
        eval_loss = float(
            F.cross_entropy(logits.reshape(-1, args.vocab), tgt.reshape(-1)).item()
        )

    return {
        "task": "reverse_seq2seq",
        "ffn": args.ffn,
        "vocab": args.vocab,
        "seq_len": args.seq_len,
        "dim": args.dim,
        "depth": args.depth,
        "steps": args.steps,
        "train_token_acc": last_acc,
        "train_loss": last_loss,
        "eval_token_acc": eval_acc,
        "eval_loss": eval_loss,
        "train_seconds": train_s,
        "thesis": "attn/softmax/LN FP; FFN binary/ternary STE — not a GPU 32x claim",
    }


def train_ae(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    model = BinaryAutoEncoder(
        n_in=args.ae_in, latent=args.latent, hidden=args.dim, ffn_kind=args.ffn
    )
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    t0 = time.perf_counter()
    last = 0.0
    for _step in range(args.steps):
        x = torch.randn(args.batch, args.ae_in)
        opt.zero_grad(set_to_none=True)
        recon = model(x)
        loss = F.mse_loss(recon, x)
        loss.backward()
        opt.step()
        model.clip_weights()
        last = float(loss.item())
    train_s = time.perf_counter() - t0
    with torch.no_grad():
        x = torch.randn(args.batch * 2, args.ae_in)
        mse = float(F.mse_loss(model(x), x).item())
    return {
        "task": "binary_autoencoder",
        "ffn": args.ffn,
        "n_in": args.ae_in,
        "latent": args.latent,
        "steps": args.steps,
        "train_mse": last,
        "eval_mse": mse,
        "train_seconds": train_s,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Binary Encoder/Decoder demos")
    p.add_argument("--task", choices=("seq2seq", "ae", "both"), default="seq2seq")
    p.add_argument("--ffn", choices=("binary", "ternary", "fp"), default="binary")
    p.add_argument("--vocab", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--ff", type=int, default=128)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ae-in", type=int, default=64)
    p.add_argument("--latent", type=int, default=16)
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "seq2seq_encoder_decoder.json",
    )
    args = p.parse_args(argv)

    payload: dict = {"demos": []}
    if args.task in ("seq2seq", "both"):
        payload["demos"].append(train_seq2seq(args))
    if args.task in ("ae", "both"):
        payload["demos"].append(train_ae(args))
    # Flatten primary demo for quick glance
    if payload["demos"]:
        payload.update(payload["demos"][0])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
