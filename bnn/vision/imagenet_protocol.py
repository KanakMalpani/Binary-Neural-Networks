"""ImageNet folder dataset contract + protocol helpers (M6 / W6.T07).

Full ImageNet Bi-Real *training* remains an ADR ACCEPTED-NON-GOAL and is never
a CI / ``bnn repro`` gate. This module ships the runnable protocol: layout check,
proxy scaffold, and smoke train step on tiny tensors.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Minimal valid 1×1 RGB PNG (stdlib — no Pillow required for proxy scaffold).
_MINIMAL_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753de"
    "0000000c4944415478da63f8cfc0000003010100f70341430000000049454e44ae426082"
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".ppm", ".bmp"}

# Published contract — do not invent alternate golden shapes for ImageNet SOTA.
IMAGENET_DATASET_CONTRACT: dict[str, Any] = {
    "schema": "bnn_imagenet_folder_contract_v1",
    "name": "imagenet_imagefolder",
    "layout": {
        "train": "train/<classname>/*.{jpg,jpeg,png,...}",
        "val": "val/<classname>/*.{jpg,jpeg,png,...}",
    },
    "full_imagenet_reference": {
        "num_classes": 1000,
        "train_images_approx": 1_281_167,
        "val_images": 50_000,
        "canonical_input_size": 224,
        "note": "Reference sizes only — not required for protocol smoke.",
    },
    "proxy_minimum": {
        "num_classes": 2,
        "images_per_class_per_split": 1,
        "input_sizes_ok": [32, 64, 224],
        "sufficient_for": ["layout check", "forward smoke", "1-step STE train"],
    },
    "pass_gates": {
        "layout_ok": True,
        "smoke_forward_finite": True,
        "sota_top1": False,
        "full_train_required": False,
        "invented_goldens": False,
    },
    "non_goals": [
        "Full ImageNet SOTA as CI / repro gate",
        "Committing ImageNet under data/",
        "Claiming GPU 32× from sign()",
    ],
    "in_repo_evidence": "CIFAR-10 Bi-Real via bnn train-image (results/image_cifar.json floors)",
}


def describe_imagenet_folder_layout(root: Path | str) -> str:
    root = Path(root)
    return (
        f"ImageNet-style layout expected under {root}:\n"
        "  train/<classname>/*.jpg|png|jpeg\n"
        "  val/<classname>/*.jpg|png|jpeg\n"
        "Full ImageNet Bi-Real train is an ADR ACCEPTED-NON-GOAL.\n"
        "Use CIFAR-10 via `bnn train-image` / `bnn train-cifar` for in-repo evidence.\n"
        "Protocol runner: `python scripts/imagenet_protocol.py --mode smoke`."
    )


def _count_images(class_dir: Path) -> int:
    return sum(
        1
        for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def check_imagenet_folder(root: Path | str, *, require_images: bool = False) -> dict:
    """Validate ImageFolder-style train/val layout.

    When ``require_images`` is False (default), non-empty class directories are
    enough (legacy stub behaviour). When True, each class must contain ≥1 image.
    """
    root = Path(root)
    train = root / "train"
    val = root / "val"
    train_classes = (
        sorted([p for p in train.iterdir() if p.is_dir()]) if train.is_dir() else []
    )
    val_classes = (
        sorted([p for p in val.iterdir() if p.is_dir()]) if val.is_dir() else []
    )
    n_train_cls = len(train_classes)
    n_val_cls = len(val_classes)
    train_images = sum(_count_images(p) for p in train_classes)
    val_images = sum(_count_images(p) for p in val_classes)
    layout_ok = train.is_dir() and val.is_dir() and n_train_cls > 0 and n_val_cls > 0
    images_ok = train_images > 0 and val_images > 0
    ok = layout_ok and (images_ok if require_images else True)
    proxy_min = IMAGENET_DATASET_CONTRACT["proxy_minimum"]
    meets_proxy = (
        layout_ok
        and n_train_cls >= proxy_min["num_classes"]
        and n_val_cls >= proxy_min["num_classes"]
        and (
            not require_images
            or (
                train_images >= n_train_cls * proxy_min["images_per_class_per_split"]
                and val_images >= n_val_cls * proxy_min["images_per_class_per_split"]
            )
        )
    )
    return {
        "ok": ok,
        "layout_ok": layout_ok,
        "images_ok": images_ok,
        "meets_proxy_minimum": meets_proxy,
        "train_classes": n_train_cls,
        "val_classes": n_val_cls,
        "train_images": train_images,
        "val_images": val_images,
        "contract_schema": IMAGENET_DATASET_CONTRACT["schema"],
        "hint": describe_imagenet_folder_layout(root),
    }


def write_dataset_contract(path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(IMAGENET_DATASET_CONTRACT, indent=2) + "\n", encoding="utf-8")
    return path


def make_proxy_imagenet(
    root: Path | str,
    *,
    n_classes: int = 4,
    images_per_class: int = 2,
    class_prefix: str = "n",
) -> dict:
    """Scaffold a tiny ImageFolder tree with minimal PNG bytes (no network)."""
    root = Path(root)
    if n_classes < 2:
        raise ValueError("proxy requires n_classes >= 2")
    names = [f"{class_prefix}{i:04d}" for i in range(n_classes)]
    for split in ("train", "val"):
        for name in names:
            d = root / split / name
            d.mkdir(parents=True, exist_ok=True)
            for j in range(images_per_class):
                (d / f"img_{j:03d}.png").write_bytes(_MINIMAL_PNG)
    report = check_imagenet_folder(root, require_images=True)
    report["proxy"] = True
    report["class_names"] = names
    report["root"] = str(root.resolve())
    return report


def dataset_contract_summary() -> str:
    c = IMAGENET_DATASET_CONTRACT
    lines = [
        f"schema: {c['schema']}",
        f"layout: {c['layout']['train']} | {c['layout']['val']}",
        f"proxy_min classes: {c['proxy_minimum']['num_classes']}",
        f"SOTA gate: {c['pass_gates']['sota_top1']}",
        f"in-repo evidence: {c['in_repo_evidence']}",
    ]
    return "\n".join(lines)
