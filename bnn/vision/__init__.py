"""Optional ImageNet-style folder loader stub (not a full ImageNet train).

Expects ``root/train/<class>/*.jpg`` and ``root/val/<class>/*.jpg``.
If missing, raises with instructions — CIFAR remains the in-repo image path.
"""

from __future__ import annotations

from pathlib import Path


def describe_imagenet_folder_layout(root: Path | str) -> str:
    root = Path(root)
    return (
        f"ImageNet-style layout expected under {root}:\n"
        "  train/<classname>/*.jpg|png\n"
        "  val/<classname>/*.jpg|png\n"
        "Full ImageNet Bi-Real train is an ADR ACCEPTED-NON-GOAL.\n"
        "Use CIFAR-10 via `bnn train-image` / `bnn train-cifar` for in-repo evidence."
    )


def check_imagenet_folder(root: Path | str) -> dict:
    root = Path(root)
    train = root / "train"
    val = root / "val"
    ok = train.is_dir() and val.is_dir()
    n_train_cls = len([p for p in train.iterdir() if p.is_dir()]) if train.is_dir() else 0
    n_val_cls = len([p for p in val.iterdir() if p.is_dir()]) if val.is_dir() else 0
    return {
        "ok": ok and n_train_cls > 0 and n_val_cls > 0,
        "train_classes": n_train_cls,
        "val_classes": n_val_cls,
        "hint": describe_imagenet_folder_layout(root),
    }
