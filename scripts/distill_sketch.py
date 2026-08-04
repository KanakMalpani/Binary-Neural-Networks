#!/usr/bin/env python3
"""Toy KD: FP teacher → binary student on MNIST (protocol sketch, not BitDistill).

Prefer the wrap-integrated path for WC-O work:
  - ``bnn.wrap.distill_binary_student`` (W3.T08)
  - ``python scripts/distill_wrap_demo.py`` (measured cosine uplift vs cold PTQ)
"""

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
from bnn.ste import clip_weights_  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--T", type=float, default=2.0)
    p.add_argument("--alpha", type=float, default=0.5, help="CE vs KD mix")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "distill_sketch.json")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    train_loader, test_loader = get_mnist_loaders(ROOT / "data", args.batch_size)

    teacher = build_model("fp32_mlp", hidden=256).to(device)
    student = build_model("binary_mlp", hidden=256).to(device)
    # Quick teacher warm-up
    opt_t = torch.optim.Adam(teacher.parameters(), lr=1e-3)
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        opt_t.zero_grad(set_to_none=True)
        loss = F.cross_entropy(teacher(x), y)
        loss.backward()
        opt_t.step()
        break

    teacher.eval()
    opt = torch.optim.Adam(student.parameters(), lr=1e-3)
    for _ep in range(args.epochs):
        student.train()
        for i, (x, y) in enumerate(train_loader):
            if i > 40:
                break
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                t_logits = teacher(x)
            s_logits = student(x)
            ce = F.cross_entropy(s_logits, y)
            kd = F.kl_div(
                F.log_softmax(s_logits / args.T, dim=1),
                F.softmax(t_logits / args.T, dim=1),
                reduction="batchmean",
            ) * (args.T * args.T)
            loss = args.alpha * ce + (1 - args.alpha) * kd
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            clip_weights_(student)

    student.eval()
    correct = total = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            if i > 30:
                break
            x, y = x.to(device), y.to(device)
            correct += (student(x).argmax(1) == y).sum().item()
            total += y.numel()
    acc = 100.0 * correct / max(total, 1)
    payload = {
        "protocol": "toy KD FP→binary MLP MNIST sketch",
        "student_acc_partial_test": acc,
        "note": "Closes distill *protocol* gap; not BitDistill reproduction.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
