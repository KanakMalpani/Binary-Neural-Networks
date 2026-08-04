#!/usr/bin/env python3
"""ImageNet protocol runner (M6 / W6.T07) — check, proxy, smoke (not SOTA).

Examples::

  python scripts/imagenet_protocol.py --mode contract
  python scripts/imagenet_protocol.py --mode check --root path/to/ImageNet
  python scripts/imagenet_protocol.py --mode smoke --out results/imagenet_protocol_smoke.json

Full ImageNet training is an ADR ACCEPTED-NON-GOAL. Smoke uses a tiny proxy
folder + synthetic tensors so CI needs no dataset download.
"""

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

from bnn.determinism import set_repro_seed  # noqa: E402
from bnn.models import count_parameters  # noqa: E402
from bnn.ste import clip_weights_, set_approx_sign  # noqa: E402
from bnn.vision.imagenet_protocol import (  # noqa: E402
    IMAGENET_DATASET_CONTRACT,
    check_imagenet_folder,
    dataset_contract_summary,
    make_proxy_imagenet,
    write_dataset_contract,
)
from bnn.vision.models import ResNetBiReal18, ResNetBiRealCIFAR  # noqa: E402

_DEFAULT_OUT = {
    "smoke": "imagenet_protocol_smoke.json",
    "check": "imagenet_protocol_check.json",
    "proxy": "imagenet_protocol_proxy.json",
    "contract": "imagenet_dataset_contract.json",
}


def _portable_path(path: Path | str) -> str:
    """Repo-relative posix path when under ROOT; else redact machine-local abs paths."""
    p = Path(path).resolve()
    try:
        return p.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return f"<external>/{p.name}"


def _sanitize_folder_report(report: dict) -> dict:
    out = dict(report)
    if "root" in out:
        out["root"] = _portable_path(out["root"])
    if "hint" in out:
        from bnn.vision.imagenet_protocol import describe_imagenet_folder_layout

        out["hint"] = describe_imagenet_folder_layout(out.get("root", "<root>"))
    return out


def _smoke_train_step(
    *,
    num_classes: int,
    image_size: int,
    width: int,
    cifar_stem: bool,
    batch_size: int,
    seed: int,
) -> dict:
    set_repro_seed(seed, deterministic=True, force_cpu=True)
    device = torch.device("cpu")
    if cifar_stem:
        model = ResNetBiRealCIFAR(num_classes=num_classes, width=width).to(device)
        arch = "resnet_bireal_cifar"
    else:
        model = ResNetBiReal18(num_classes=num_classes, width=width).to(device)
        arch = "resnet_bireal18"
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch_size, 3, image_size, image_size, generator=g)
    y = torch.randint(0, num_classes, (batch_size,), generator=g)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    t0 = time.perf_counter()
    opt.zero_grad(set_to_none=True)
    logits = model(x)
    loss = loss_fn(logits, y)
    loss.backward()
    opt.step()
    clip_weights_(model)
    train_s = time.perf_counter() - t0
    model.eval()
    with torch.no_grad():
        logits_e = model(x)
    return {
        "arch": arch,
        "num_classes": num_classes,
        "image_size": image_size,
        "width": width,
        "batch_size": batch_size,
        "loss": float(loss.item()),
        "logits_finite": bool(torch.isfinite(logits_e).all()),
        "logits_shape": list(logits_e.shape),
        "train_step_seconds": train_s,
        "params": count_parameters(model),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode",
        choices=("contract", "check", "proxy", "smoke"),
        default="smoke",
        help="contract=print/write schema; check=validate root; "
        "proxy=scaffold tiny folder; smoke=proxy+1 STE step",
    )
    p.add_argument("--root", type=Path, default=None, help="ImageNet / proxy root")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON (default: results/imagenet_protocol_<mode>.json; "
        "smoke alone writes imagenet_protocol_smoke.json)",
    )
    p.add_argument("--n-classes", type=int, default=4)
    p.add_argument("--images-per-class", type=int, default=2)
    p.add_argument("--image-size", type=int, default=64, help="Smoke tensor HxW")
    p.add_argument("--width", type=int, default=16, help="ResNet Bi-Real base width")
    p.add_argument(
        "--imagenet-stem",
        action="store_true",
        help="Use 7×7 ImageNet stem (default: CIFAR stem for small smoke sizes)",
    )
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--approx-sign", action="store_true")
    p.add_argument(
        "--write-contract",
        type=Path,
        default=None,
        help="Also write contract JSON to this path",
    )
    args = p.parse_args()
    if args.out is None:
        args.out = ROOT / "results" / _DEFAULT_OUT[args.mode]

    set_approx_sign(args.approx_sign)
    payload: dict = {
        "protocol": "imagenet_folder_v1",
        "mode": args.mode,
        "contract": IMAGENET_DATASET_CONTRACT,
        "sota_gate": False,
        "thesis_note": (
            "Packed CPU/edge XNOR-popcount; never claim GPU 32× from sign(). "
            "ImageNet SOTA is not a pass gate."
        ),
    }

    if args.write_contract is not None:
        write_dataset_contract(args.write_contract)
        payload["contract_written"] = _portable_path(args.write_contract)

    if args.mode == "contract":
        print(dataset_contract_summary())
        print(json.dumps(IMAGENET_DATASET_CONTRACT, indent=2))
        if args.write_contract is None:
            default_c = ROOT / "results" / "imagenet_dataset_contract.json"
            write_dataset_contract(default_c)
            print(f"Wrote {default_c}")
        return 0

    root = args.root
    if args.mode in ("proxy", "smoke"):
        if root is None:
            root = ROOT / "data" / "_imagenet_proxy_smoke"
        report = make_proxy_imagenet(
            root,
            n_classes=args.n_classes,
            images_per_class=args.images_per_class,
        )
        payload["folder"] = _sanitize_folder_report(report)
        print(f"Proxy ImageNet at {payload['folder']['root']} ok={report['ok']}", flush=True)
    elif args.mode == "check":
        if root is None:
            print("ERROR: --root required for --mode check", file=sys.stderr)
            return 2
        report = check_imagenet_folder(root, require_images=True)
        report = dict(report)
        report["root"] = _portable_path(root)
        payload["folder"] = _sanitize_folder_report(report)
        print(json.dumps(payload["folder"], indent=2))
        if not report["ok"]:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return 1

    if args.mode == "smoke":
        use_imagenet_stem = args.imagenet_stem and args.image_size >= 112
        smoke = _smoke_train_step(
            num_classes=args.n_classes,
            image_size=args.image_size,
            width=args.width,
            cifar_stem=not use_imagenet_stem,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        payload["smoke"] = smoke
        payload["verdict"] = (
            "PASS protocol smoke"
            if smoke["logits_finite"]
            else "FAIL non-finite logits"
        )
        print(
            f"Smoke {smoke['arch']} loss={smoke['loss']:.4f} "
            f"finite={smoke['logits_finite']} ({smoke['train_step_seconds']:.3f}s)",
            flush=True,
        )
        if not smoke["logits_finite"]:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
