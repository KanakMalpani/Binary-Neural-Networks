"""Export ``.bnnpack`` packed tensors to safetensors (W5.T06).

Soft-depends on the ``safetensors`` package. Does not replace ``.bnnpack`` —
it is a HF-friendly side-car of the same packed buffers + JSON meta.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .packfile import load_bnnpack


def _require_safetensors():
    try:
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover - env dependent
        raise ImportError(
            "safetensors is required for export_bnnpack_safetensors. "
            "Install with: pip install safetensors"
        ) from exc
    return save_file


def _tensor_from_blob_field(value: Any) -> torch.Tensor | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous()
    return torch.as_tensor(value).cpu().contiguous()


def bnnpack_tensors_for_safetensors(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Flatten layer packed fields into a safetensors-friendly name → tensor map."""
    tensors: dict[str, torch.Tensor] = {}
    layers = payload.get("layers") or {}
    if not isinstance(layers, dict):
        raise ValueError("payload.layers must be a dict")
    for name, blob in layers.items():
        if not isinstance(blob, dict):
            raise ValueError(f"layer {name!r} is not a dict")
        prefix = name.replace("/", ".")
        kind = str(blob.get("kind", "unknown"))
        for key in (
            "weight_packed_i64",
            "weight_packed_u8",
            "alpha",
            "scale",
            "bias",
        ):
            t = _tensor_from_blob_field(blob.get(key))
            if t is None:
                continue
            tensors[f"{prefix}.{key}"] = t
        # Tiny tag tensor so kind survives even if a consumer drops JSON meta.
        tensors[f"{prefix}.__kind_tag"] = torch.tensor(
            [hash(kind) & 0xFFFFFFFF], dtype=torch.int64
        )
    return tensors


def export_bnnpack_safetensors(
    pack_path: Path | str,
    out_path: Path | str,
    *,
    meta_json_path: Path | str | None = None,
) -> tuple[Path, Path]:
    """Load ``.bnnpack`` → write ``.safetensors`` + JSON meta sidecar.

    Returns ``(safetensors_path, meta_json_path)``.
    """
    save_file = _require_safetensors()
    pack_path = Path(pack_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if meta_json_path is None:
        meta_json_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    else:
        meta_json_path = Path(meta_json_path)

    payload = load_bnnpack(pack_path)
    tensors = bnnpack_tensors_for_safetensors(payload)
    if not tensors:
        raise ValueError(f"{pack_path}: no exportable packed tensors")

    save_file(tensors, str(out_path))

    layer_meta: dict[str, Any] = {}
    for name, blob in (payload.get("layers") or {}).items():
        if not isinstance(blob, dict):
            continue
        layer_meta[name] = {
            k: blob[k]
            for k in (
                "kind",
                "name",
                "in_features",
                "out_features",
                "in_channels",
                "out_channels",
                "kernel_size",
                "stride",
                "padding",
                "n",
                "fp32_bytes",
                "packed_bytes",
                "compression",
                "per_channel",
                "content_sha256",
            )
            if k in blob
        }

    meta_doc = {
        "format": "bnnpack_safetensors_v1",
        "source_pack": str(pack_path),
        "bnnpack_version": int(payload.get("version", 0)),
        "magic": payload.get("magic"),
        "pack_meta": payload.get("meta") or {},
        "hashes": payload.get("hashes") or {},
        "layers": layer_meta,
        "tensor_keys": sorted(tensors.keys()),
    }
    meta_json_path.parent.mkdir(parents=True, exist_ok=True)
    meta_json_path.write_text(json.dumps(meta_doc, indent=2) + "\n", encoding="utf-8")
    return out_path, meta_json_path


__all__ = [
    "bnnpack_tensors_for_safetensors",
    "export_bnnpack_safetensors",
]
