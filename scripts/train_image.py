#!/usr/bin/env python3
"""Image lane: CIFAR-10 FP vs Bi-Real (+ optional tiny ViT) with latency/size report."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bnn.cifar import get_cifar10_loaders  # noqa: E402
from bnn.determinism import set_repro_seed  # noqa: E402
from bnn.models import count_parameters  # noqa: E402
from bnn.ste import clip_weights_, set_approx_sign  # noqa: E402
from bnn.vision.models import (  # noqa: E402
    BinaryCIFARCNN,
    FP32CIFARCNN,
    TinyBinaryViT,
)


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.numel()
    return 100.0 * correct / max(total, 1)


@torch.no_grad()
def latency_ms(model, loader, device, batches: int = 10) -> float:
    model.eval()
    it = iter(loader)
    xs = []
    for _ in range(min(batches, len(loader))):
        x, _ = next(it)
        xs.append(x.to(device))
    for _ in range(3):
        model(xs[0])
    t0 = time.perf_counter()
    n = 0
    for x in xs:
        model(x)
        n += 1
    return 1e3 * (time.perf_counter() - t0) / max(n, 1)


def train_one(name, model, train_loader, test_loader, epochs, lr, device):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    hist = []
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        model.train()
        running = n = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            if hasattr(model, "clip_weights"):
                model.clip_weights()
            else:
                clip_weights_(model)
            running += loss.item() * y.size(0)
            n += y.size(0)
        acc = evaluate(model, test_loader, device)
        hist.append({"epoch": ep, "loss": running / max(n, 1), "test_acc": acc})
        print(f"[{name}] epoch {ep}/{epochs} loss={hist[-1]['loss']:.4f} acc={acc:.2f}%", flush=True)
    return {
        "model": name,
        "test_acc": hist[-1]["test_acc"],
        "train_seconds": time.perf_counter() - t0,
        "latency_ms_per_batch": latency_ms(model, test_loader, device),
        "params": count_parameters(model),
        "history": hist,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--train-subset", type=int, default=30000, help="0 = full 50k")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--approx-sign", action="store_true", help="Use Bi-Real ApproxSign STE")
    p.add_argument("--include-vit", action="store_true", help="Also train tiny binary ViT")
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "image_cifar.json")
    args = p.parse_args()

    set_repro_seed(args.seed, deterministic=True, force_cpu=True)
    set_approx_sign(args.approx_sign)
    device = torch.device("cpu")
    subset = None if args.train_subset <= 0 else args.train_subset
    train_loader, test_loader = get_cifar10_loaders(
        args.data_dir, args.batch_size, train_subset=subset, seed=args.seed
    )
    print(
        f"Image CIFAR-10 subset={subset} epochs={args.epochs} approx_sign={args.approx_sign}",
        flush=True,
    )

    jobs = [
        ("fp32_cifar_cnn", lambda: FP32CIFARCNN(args.channels)),
        ("binary_cifar_bireal", lambda: BinaryCIFARCNN(args.channels)),
    ]
    if args.include_vit:
        jobs.append(("tiny_vit_binary", lambda: TinyBinaryViT(dim=64, depth=2)))

    results = [
        train_one(n, ctor().to(device), train_loader, test_loader, args.epochs, args.lr, device)
        for n, ctor in jobs
    ]
    gap = results[0]["test_acc"] - results[1]["test_acc"]
    payload = {
        "modality": "image",
        "dataset": "CIFAR-10",
        "train_subset": subset,
        "epochs": args.epochs,
        "approx_sign": args.approx_sign,
        "results": results,
        "acc_gap_pp_fp_vs_binary_cnn": gap,
        "verdict": (
            f"Bi-Real binary within {gap:.2f} pp of FP CNN twin. "
            "Packed Linear kernels apply to ViT FFN / MLP heads; Conv path is STE-sim "
            "(BinaryConv wrap available for size — see wrapper)."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md = [
        "# Image — CIFAR-10 Bi-Real",
        "",
        f"- Subset: {subset if subset else 'full'} | epochs: {args.epochs} | approx_sign: {args.approx_sign}",
        f"- FP32 CNN: **{results[0]['test_acc']:.2f}%** ({results[0]['latency_ms_per_batch']:.1f} ms/batch)",
        f"- Binary Bi-Real: **{results[1]['test_acc']:.2f}%** ({results[1]['latency_ms_per_batch']:.1f} ms/batch)",
        f"- Gap: **{gap:.2f} pp**",
        f"- {payload['verdict']}",
        "",
    ]
    args.out.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    # Also refresh legacy proxy path for eval-suite compatibility
    legacy = ROOT / "results" / "cifar10_proxy.json"
    legacy.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
