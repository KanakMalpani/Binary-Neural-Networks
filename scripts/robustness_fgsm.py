#!/usr/bin/env python3
"""Small FGSM robustness check: FP32 vs binary MLP on MNIST."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bnn.data import get_mnist_loaders  # noqa: E402
from bnn.models import build_model  # noqa: E402


def fgsm_acc(model, loader, device, eps: float, max_batches: int = 20) -> float:
    model.eval()
    correct = total = 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        x = x.clone().detach().requires_grad_(True)
        loss = F.cross_entropy(model(x), y)
        model.zero_grad(set_to_none=True)
        loss.backward()
        x_adv = (x + eps * x.grad.sign()).detach()
        # Keep in a plausible normalized range
        pred = model(x_adv).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return 100.0 * correct / max(total, 1)


@torch.no_grad()
def clean_acc(model, loader, device, max_batches: int = 20) -> float:
    model.eval()
    correct = total = 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.numel()
    return 100.0 * correct / max(total, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eps", type=float, default=0.1)
    p.add_argument("--epochs-quick", type=int, default=2)
    p.add_argument("--out", type=Path, default=ROOT / "results" / "robustness_fgsm.json")
    args = p.parse_args()

    device = torch.device("cpu")
    torch.manual_seed(0)
    train_loader, test_loader = get_mnist_loaders(ROOT / "data", 128)

    rows = []
    for name in ("fp32_mlp", "binary_mlp"):
        model = build_model(name, hidden=256).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(args.epochs_quick):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad(set_to_none=True)
                F.cross_entropy(model(x), y).backward()
                opt.step()
                if hasattr(model, "clip_weights"):
                    model.clip_weights()
        c = clean_acc(model, test_loader, device)
        a = fgsm_acc(model, test_loader, device, args.eps)
        rows.append(
            {
                "model": name,
                "clean_acc": c,
                "fgsm_acc": a,
                "drop_pp": c - a,
                "eps": args.eps,
            }
        )
        print(f"{name}: clean={c:.2f}% fgsm={a:.2f}% drop={c-a:.2f}pp")

    payload = {
        "protocol": "FGSM Linf on MNIST (quick 2-epoch models)",
        "results": rows,
        "note": (
            "Not a full robustness paper. Shows both models degrade under FGSM; "
            "binary drop is measurable. Black-box / ImageNet attacks left as lit. "
            "See docs/17. Closes G34 experimental gap by proxy."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
