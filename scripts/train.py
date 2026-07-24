#!/usr/bin/env python3
"""Train FP32 / Binary / Ternary models on MNIST."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bnn.data import get_mnist_loaders  # noqa: E402
from bnn.determinism import set_repro_seed  # noqa: E402
from bnn.models import build_model, count_parameters  # noqa: E402


def set_seed(seed: int) -> None:
    set_repro_seed(seed, deterministic=True, force_cpu=True)


def get_loaders(data_dir: Path, batch_size: int) -> tuple[DataLoader, DataLoader]:
    return get_mnist_loaders(data_dir, batch_size)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return 100.0 * correct / total


def train_one(
    model_name: str,
    epochs: int,
    batch_size: int,
    lr: float,
    hidden: int,
    device: torch.device,
    data_dir: Path,
    ckpt_dir: Path,
    seed: int,
) -> dict:
    set_seed(seed)
    model = build_model(model_name, hidden=hidden).to(device)
    train_loader, test_loader = get_loaders(data_dir, batch_size)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    history = []
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            if hasattr(model, "clip_weights"):
                model.clip_weights()
            running += loss.item() * y.size(0)
            n += y.size(0)
        acc = evaluate(model, test_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": running / max(n, 1),
            "test_acc": acc,
        }
        history.append(row)
        print(
            f"[{model_name}] epoch {epoch}/{epochs}  "
            f"loss={row['train_loss']:.4f}  test_acc={acc:.2f}%"
        )

    train_s = time.perf_counter() - t0
    # Inference throughput (sim mode)
    model.eval()
    x_bench, _ = next(iter(test_loader))
    x_bench = x_bench.to(device)
    for _ in range(5):
        model(x_bench)
    t1 = time.perf_counter()
    reps = 20
    with torch.no_grad():
        for _ in range(reps):
            model(x_bench)
    infer_s = (time.perf_counter() - t1) / reps
    ips = x_bench.size(0) / infer_s

    ckpt_path = ckpt_dir / f"{model_name}.pt"
    torch.save({"model": model.state_dict(), "meta": {"name": model_name}}, ckpt_path)

    result = {
        "model": model_name,
        "device": str(device),
        "epochs": epochs,
        "test_acc": history[-1]["test_acc"],
        "train_seconds": train_s,
        "sim_infer_batch_seconds": infer_s,
        "sim_images_per_sec": ips,
        "params": count_parameters(model),
        "history": history,
        "checkpoint": str(ckpt_path),
    }
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--models",
        nargs="+",
        default=["fp32_mlp", "binary_mlp", "ternary_mlp", "fp32_cnn", "binary_cnn"],
    )
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=None, help="torch.set_num_threads")
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--ckpt-dir", type=Path, default=ROOT / "checkpoints")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "train_results.json")
    args = p.parse_args()

    if args.threads is not None:
        torch.set_num_threads(args.threads)
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} seed={args.seed} threads={torch.get_num_threads()}")
    print("NOTE: STE training is not an inference throughput win; use packed kernels for speed.")

    all_results = []
    for name in args.models:
        all_results.append(
            train_one(
                name,
                args.epochs,
                args.batch_size,
                args.lr,
                args.hidden,
                device,
                args.data_dir,
                args.ckpt_dir,
                args.seed,
            )
        )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"Wrote {args.out}")

    # Quick markdown summary
    md = args.out.with_suffix(".md")
    lines = [
        "# Training results",
        "",
        "| Model | Test acc % | Train s | Sim img/s | Binary-ish params |",
        "|-------|------------|---------|-----------|-------------------|",
    ]
    for r in all_results:
        lines.append(
            f"| {r['model']} | {r['test_acc']:.2f} | {r['train_seconds']:.1f} | "
            f"{r['sim_images_per_sec']:.0f} | {r['params']['binary_or_ternary_weight_params']} |"
        )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {md}")


if __name__ == "__main__":
    main()
