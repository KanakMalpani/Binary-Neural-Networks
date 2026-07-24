#!/usr/bin/env python3
"""Audio lane: synthetic tone spectrograms — FP vs binary CNN."""

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

from bnn.audio.data import get_audio_loaders  # noqa: E402
from bnn.audio.models import build_audio_model  # noqa: E402
from bnn.models import count_parameters  # noqa: E402
from bnn.ste import clip_weights_, set_approx_sign  # noqa: E402


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.numel()
    return 100.0 * correct / max(total, 1)


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
        "params": count_parameters(model),
        "history": hist,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-train", type=int, default=800)
    p.add_argument("--n-test", type=int, default=200)
    p.add_argument("--n-classes", type=int, default=8)
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--approx-sign", action="store_true")
    p.add_argument("--cache-dir", type=Path, default=ROOT / "data" / "audio_cache")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "audio_synth.json")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    set_approx_sign(args.approx_sign)
    device = torch.device("cpu")
    train_loader, test_loader, meta = get_audio_loaders(
        batch_size=args.batch_size,
        n_train=args.n_train,
        n_test=args.n_test,
        n_classes=args.n_classes,
        seed=args.seed,
        cache_dir=args.cache_dir,
    )
    print(f"Audio synthetic meta={meta}", flush=True)

    results = []
    for name in ("fp32_cnn", "binary_cnn"):
        model = build_audio_model(name, n_classes=args.n_classes, channels=args.channels).to(device)
        results.append(
            train_one(name, model, train_loader, test_loader, args.epochs, args.lr, device)
        )

    gap = results[0]["test_acc"] - results[1]["test_acc"]
    payload = {
        "modality": "audio",
        "meta": meta,
        "epochs": args.epochs,
        "approx_sign": args.approx_sign,
        "results": results,
        "acc_gap_pp": gap,
        "verdict": (
            f"Binary audio-CNN within {gap:.2f} pp of FP on synthetic tone spectrograms. "
            "Classic BNN is NOT production ASR — use INT8 Whisper/ORT for real speech; "
            "this demo proves STE + packing pattern on audio features."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.out.with_suffix(".md").write_text(
        "\n".join(
            [
                "# Audio — synthetic tone spectrograms",
                "",
                f"- Classes: {args.n_classes} | train: {args.n_train} | epochs: {args.epochs}",
                f"- FP32 CNN: **{results[0]['test_acc']:.2f}%**",
                f"- Binary CNN: **{results[1]['test_acc']:.2f}%**",
                f"- Gap: **{gap:.2f} pp**",
                f"- {payload['verdict']}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
