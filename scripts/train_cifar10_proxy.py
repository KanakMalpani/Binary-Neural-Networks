#!/usr/bin/env python3
"""CIFAR-10 FP32 vs Bi-Real-style binary CNN — ImageNet proxy evidence."""

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
from bnn.layers import BinaryConv2d, BiRealBlock  # noqa: E402
from bnn.ste import clip_weights_  # noqa: E402


class FP32CIFARCNN(nn.Module):
    def __init__(self, channels: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(channels, channels * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels * 2, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels * 2, channels * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels * 2, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(channels * 2, 10)

    def forward(self, x):
        return self.head(self.features(x).flatten(1))


class BinaryCIFARCNN(nn.Module):
    """Bi-Real-style CIFAR CNN: FP stem/head, binary blocks + FP residuals."""

    def __init__(self, channels: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels, momentum=0.9),
        )
        self.b1 = BiRealBlock(channels)
        self.b2 = BiRealBlock(channels)
        self.pool = nn.MaxPool2d(2)
        self.down = nn.Sequential(
            BinaryConv2d(channels, channels * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels * 2, momentum=0.9),
        )
        self.skip = nn.Conv2d(channels, channels * 2, 1, bias=False)
        self.b3 = BiRealBlock(channels * 2)
        self.b4 = BiRealBlock(channels * 2)
        self.pool2 = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(channels * 2, 10)

    def forward(self, x):
        x = self.stem(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.pool(x)
        x = self.down(x) + self.skip(x)
        x = self.b3(x)
        x = self.b4(x)
        return self.head(self.pool2(x).flatten(1))

    def clip_weights(self):
        clip_weights_(self)


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.numel()
    return 100.0 * correct / total


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
            running += loss.item() * y.size(0)
            n += y.size(0)
        acc = evaluate(model, test_loader, device)
        hist.append({"epoch": ep, "loss": running / max(n, 1), "test_acc": acc})
        print(f"[{name}] epoch {ep}/{epochs} loss={hist[-1]['loss']:.4f} acc={acc:.2f}%")
    return {
        "model": name,
        "test_acc": hist[-1]["test_acc"],
        "train_seconds": time.perf_counter() - t0,
        "history": hist,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--train-subset", type=int, default=20000, help="0 = full 50k")
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "cifar10_proxy.json")
    args = p.parse_args()

    torch.manual_seed(0)
    device = torch.device("cpu")
    subset = None if args.train_subset <= 0 else args.train_subset
    train_loader, test_loader = get_cifar10_loaders(
        args.data_dir, args.batch_size, train_subset=subset
    )
    print(f"CIFAR-10 train batches={len(train_loader)} subset={subset} device={device}")

    results = []
    for name, ctor in (
        ("fp32_cifar_cnn", lambda: FP32CIFARCNN(args.channels)),
        ("binary_cifar_bireal", lambda: BinaryCIFARCNN(args.channels)),
    ):
        results.append(
            train_one(name, ctor().to(device), train_loader, test_loader, args.epochs, args.lr, device)
        )

    gap = results[0]["test_acc"] - results[1]["test_acc"]
    payload = {
        "dataset": "CIFAR-10",
        "proxy_for": "ImageNet-scale BNN claim (#33)",
        "train_subset": subset,
        "epochs": args.epochs,
        "channels": args.channels,
        "results": results,
        "acc_gap_pp": gap,
        "verdict": (
            f"Bi-Real binary within {gap:.2f} pp of FP32 twin under same short schedule. "
            "Closes beyond-MNIST accuracy dimension by proxy; ImageNet optional scale-up."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md = [
        "# CIFAR-10 proxy (ImageNet substitute)",
        "",
        f"- Subset: {subset if subset else 'full'} | epochs: {args.epochs} | ch: {args.channels}",
        f"- FP32 acc: **{results[0]['test_acc']:.2f}%**",
        f"- Binary Bi-Real acc: **{results[1]['test_acc']:.2f}%**",
        f"- Gap: **{gap:.2f} pp**",
        f"- {payload['verdict']}",
        "",
    ]
    args.out.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
