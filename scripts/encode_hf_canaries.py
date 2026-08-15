#!/usr/bin/env python3
"""Encode tiny Hub canary ``.bnnpack`` files (Wave P1 / W5).

Lab canaries, not ImageNet SOTA. 32× is uint64 pack compression, not GPU from
``sign()``. Packs are gitignored; cards live under ``cards/``.

Examples::

    python scripts/encode_hf_canaries.py --out-dir results/hf_canaries
    python scripts/encode_hf_canaries.py --only codec --out-dir results/hf_canaries
    python scripts/encode_hf_canaries.py --out-dir results/hf_canaries --upload
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bnn.codec import (  # noqa: E402
    decode_file,
    encode_file,
    encode_from_packed_module,
    encode_linear_state,
    load_bnnpack,
    packed_module_fp_err,
    save_bnnpack,
)
from bnn.determinism import set_repro_seed  # noqa: E402
from bnn.models import build_model  # noqa: E402
from bnn.wrap.packed_linear import PackedBinaryXNORLinear  # noqa: E402

NAMESPACE = "KanakMalpani"
CANARIES: dict[str, dict[str, Any]] = {
    "wrap-demo": {
        "repo_id": f"{NAMESPACE}/bnn-lab-wrap-demo",
        "filename": "model.bnnpack",
        "card": ROOT / "cards" / "wrap-demo" / "README.md",
    },
    "mnist-mlp": {
        "repo_id": f"{NAMESPACE}/bnn-lab-mnist-mlp-canary",
        "filename": "model.bnnpack",
        "card": ROOT / "cards" / "mnist-mlp" / "README.md",
    },
    "codec": {
        "repo_id": f"{NAMESPACE}/bnn-lab-codec-canary",
        "filename": "model.bnnpack",
        "card": ROOT / "cards" / "codec" / "README.md",
    },
}


def _summarize(path: Path) -> dict[str, Any]:
    payload = load_bnnpack(path)
    layers = payload["layers"]
    fp = sum(int(b["fp32_bytes"]) for b in layers.values())
    pk = sum(int(b["packed_bytes"]) for b in layers.values())
    modules, meta = decode_file(path)
    errs = {name: packed_module_fp_err(mod) for name, mod in modules.items()}
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "n_layers": len(layers),
        "layer_names": sorted(layers),
        "fp32_bytes": fp,
        "packed_bytes": pk,
        "compression": float(fp / max(pk, 1)),
        "gemm_err": errs,
        "meta": meta if isinstance(meta, dict) else {},
    }


def make_wide_mlp(hidden: int) -> nn.Sequential:
    """Same Sequential as ``scripts/wrap_existing_demo.py`` (committed wrap_demo shape)."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, 10),
    )


def encode_wrap_demo(out: Path, *, hidden: int = 4096, seed: int = 0) -> dict[str, Any]:
    """PTQ pack of Sequential middles 3/5 — shape/codec canary, not the QAT JSON."""
    set_repro_seed(seed, deterministic=True, force_cpu=True)
    model = make_wide_mlp(hidden)
    layers: dict[str, Any] = {}
    for idx in (3, 5):
        lin = model[idx]
        assert isinstance(lin, nn.Linear)
        packed = PackedBinaryXNORLinear(lin.weight.data, lin.bias.data)
        layers[str(idx)] = encode_from_packed_module(packed, name=str(idx))
    meta = {
        "canary": True,
        "not_sota": True,
        "source": "wrap_demo_shape",
        "hidden": hidden,
        "replaced_layers": ["3", "5"],
        "seed": seed,
        "qat": False,
        "note": (
            "Shape/codec canary for wrap_demo hidden=4096. Dual-metric AND-gate "
            "(cosine ≥ 0.85 and e2e ≥ 1.5× after MSE+fold_α QAT) is in "
            "results/wrap_demo.json — not these PTQ bytes. Ultra TinyBlock hybrid "
            "still REFUSE (~0.70). 32× is uint64 pack compression, not GPU from sign()."
        ),
        "floors": "tests/golden_floors.json wrap_demo",
        "cli": "scripts/encode_hf_canaries.py",
    }
    save_bnnpack(layers, out, meta=meta)
    return _summarize(out)


def encode_mnist_mlp(
    out: Path,
    *,
    hidden: int = 512,
    seed: int = 0,
    checkpoint: Path | None = None,
) -> dict[str, Any]:
    set_repro_seed(seed, deterministic=True, force_cpu=True)
    model = build_model("binary_mlp", hidden=hidden)
    from_ckpt = False
    if checkpoint is not None:
        blob = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
        model.load_state_dict(state)
        from_ckpt = True
    meta = {
        "canary": True,
        "not_sota": True,
        "source": "binary_mlp",
        "hidden": hidden,
        "seed": seed,
        "from_checkpoint": from_ckpt,
        "note": (
            "Codec canary of BinaryMLP hidden BinaryLinear layers. Published MNIST "
            "floors (binary_mlp_min_acc 95.0, recorded 96.36) are results/train_results.json "
            "+ tests/golden_floors.json, not this pack unless --from-checkpoint was used. "
            "32× is uint64 pack compression, not GPU from sign()."
        ),
        "floors": "tests/golden_floors.json mnist",
        "cli": "scripts/encode_hf_canaries.py",
    }
    encode_file(
        model,
        out,
        meta=meta,
        include_binary_linear=True,
        include_fp_linear=False,
        include_packed=True,
    )
    return _summarize(out)


def encode_codec(out: Path, *, dim: int = 256, seed: int = 0) -> dict[str, Any]:
    set_repro_seed(seed, deterministic=True, force_cpu=True)
    w = torch.randn(dim, dim)
    blob = encode_linear_state(w, name="linear")
    meta = {
        "canary": True,
        "not_sota": True,
        "source": "random_linear",
        "in_features": dim,
        "out_features": dim,
        "seed": seed,
        "note": (
            "Tiny random Linear encode for codec round-trip. GEMM err=0; 32× is "
            "uint64 pack compression, not GPU from sign()."
        ),
        "floors": "tests/golden_floors.json compression_exact_when_uint64_pack",
        "cli": "scripts/encode_hf_canaries.py",
    }
    save_bnnpack({"linear": blob}, out, meta=meta)
    return _summarize(out)


def encode_one(
    kind: str,
    out_dir: Path,
    *,
    wrap_hidden: int = 4096,
    mnist_hidden: int = 512,
    codec_dim: int = 256,
    checkpoint: Path | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = CANARIES[kind]
    pack = out_dir / f"{kind}.bnnpack"
    if kind == "wrap-demo":
        summary = encode_wrap_demo(pack, hidden=wrap_hidden)
    elif kind == "mnist-mlp":
        summary = encode_mnist_mlp(pack, hidden=mnist_hidden, checkpoint=checkpoint)
    elif kind == "codec":
        summary = encode_codec(pack, dim=codec_dim)
    else:
        raise ValueError(kind)
    summary["id"] = kind
    summary["repo_id"] = spec["repo_id"]
    summary["card"] = str(spec["card"])
    return summary


def upload_one(kind: str, pack: Path, *, commit_message: str) -> str:
    spec = CANARIES[kind]
    card: Path = spec["card"]
    if not pack.is_file():
        raise FileNotFoundError(pack)
    if not card.is_file():
        raise FileNotFoundError(card)
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError as exc:
        raise ImportError(
            'huggingface_hub required for --upload (pip install -e ".[hf]")'
        ) from exc
    repo_id = spec["repo_id"]
    api = HfApi()
    create_repo(repo_id, repo_type="model", exist_ok=True, private=False)
    api.upload_file(
        path_or_fileobj=str(card),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )
    api.upload_file(
        path_or_fileobj=str(pack),
        path_in_repo=spec["filename"],
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )
    return f"https://huggingface.co/{repo_id}"


def ensure_collection(item_repo_ids: list[str]) -> str | None:
    """Create or update the public collection. Returns URL or None if API missing."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return None
    api = HfApi()
    # Hub collection description max is 150 characters.
    description = (
        "Tiny bnn-lab .bnnpack canaries (not SOTA). 32x is uint64 pack, not GPU from sign()."
    )
    collection = None
    create = getattr(api, "create_collection", None)
    if create is None:
        print("WARN huggingface_hub has no create_collection; skip collection", file=sys.stderr)
        return None
    try:
        collection = create(
            title="bnn-lab .bnnpack canaries",
            namespace=NAMESPACE,
            description=description,
            private=False,
            exists_ok=True,
        )
    except TypeError:
        try:
            collection = create(
                title="bnn-lab .bnnpack canaries",
                namespace=NAMESPACE,
                description=description,
                private=False,
            )
        except Exception as exc:
            print(f"WARN create_collection failed: {exc}", file=sys.stderr)
            return None
    except Exception as exc:
        print(f"WARN create_collection failed: {exc}", file=sys.stderr)
        return None
    slug = getattr(collection, "slug", None) or getattr(collection, "id", None)
    if slug is None and isinstance(collection, dict):
        slug = collection.get("slug") or collection.get("id")
    add = getattr(api, "add_collection_item", None)
    if add is not None and slug:
        for repo_id in item_repo_ids:
            try:
                add(slug, item_id=repo_id, item_type="model", exists_ok=True)
            except TypeError:
                try:
                    add(collection_slug=slug, item_id=repo_id, item_type="model")
                except Exception as exc:
                    print(f"WARN add_collection_item {repo_id}: {exc}", file=sys.stderr)
            except Exception as exc:
                print(f"WARN add_collection_item {repo_id}: {exc}", file=sys.stderr)
    if slug:
        return f"https://huggingface.co/collections/{slug}"
    return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=ROOT / "results" / "hf_canaries")
    p.add_argument(
        "--only",
        choices=sorted(CANARIES),
        action="append",
        help="Encode a subset (repeatable). Default: all three canaries.",
    )
    p.add_argument("--wrap-hidden", type=int, default=4096)
    p.add_argument("--mnist-hidden", type=int, default=512)
    p.add_argument("--codec-dim", type=int, default=256)
    p.add_argument(
        "--from-checkpoint",
        type=Path,
        default=None,
        help="Optional BinaryMLP .pt for mnist-mlp (still a canary, not SOTA).",
    )
    p.add_argument(
        "--upload",
        action="store_true",
        help="Create public Hub model repos and upload pack + card.",
    )
    p.add_argument(
        "--collection",
        action="store_true",
        help="Also create/update the Hub collection (implies --upload).",
    )
    args = p.parse_args(argv)

    kinds = args.only or list(CANARIES)
    reports: list[dict[str, Any]] = []
    for kind in kinds:
        summary = encode_one(
            kind,
            args.out_dir,
            wrap_hidden=args.wrap_hidden,
            mnist_hidden=args.mnist_hidden,
            codec_dim=args.codec_dim,
            checkpoint=args.from_checkpoint if kind == "mnist-mlp" else None,
        )
        reports.append(summary)
        err_ok = all(v == 0.0 for v in summary["gemm_err"].values())
        print(
            f"{kind}: layers={summary['n_layers']} "
            f"packed_bytes={summary['packed_bytes']} "
            f"compression={summary['compression']:.2f}x "
            f"gemm_err_ok={err_ok} -> {summary['path']}"
        )
        if not err_ok:
            print(f"ERROR GEMM err not 0: {summary['gemm_err']}", file=sys.stderr)
            return 2

    hub_urls: list[str] = []
    collection_url: str | None = None
    if args.upload or args.collection:
        msg = "feat(W5): tiny .bnnpack canary (lab demo, not SOTA)"
        try:
            for kind, summary in zip(kinds, reports, strict=True):
                url = upload_one(kind, Path(summary["path"]), commit_message=msg)
                hub_urls.append(url)
                print(f"uploaded {url}")
            if args.collection or args.upload:
                collection_url = ensure_collection(
                    [CANARIES[k]["repo_id"] for k in kinds]
                )
                if collection_url:
                    print(f"collection {collection_url}")
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "upload_ok": False,
                        "blocker": str(exc),
                        "human": (
                            "hf auth login with a write-scoped token, then "
                            "python scripts/encode_hf_canaries.py --out-dir "
                            "results/hf_canaries --upload --collection"
                        ),
                    },
                    indent=2,
                )
            )
            # Still write local summary; in-repo cards already exist.
            (args.out_dir / "encode_summary.json").write_text(
                json.dumps({"reports": reports, "upload_ok": False, "blocker": str(exc)}, indent=2)
                + "\n",
                encoding="utf-8",
            )
            return 0

    payload = {
        "schema": "bnn_hub_canaries_encode_v1",
        "honesty": "canary / lab demo, not ImageNet SOTA; 32× is pack compression",
        "reports": reports,
        "hub_urls": hub_urls,
        "collection_url": collection_url,
        "upload_ok": bool(hub_urls),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "encode_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"n": len(reports), "hub_urls": hub_urls, "collection_url": collection_url}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
