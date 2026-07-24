"""Checkpoint save/load for latent STE weights and packed inference blobs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .kernels.packed import pack_binary_pm1
from .ste import binary_sign


def save_checkpoint(
    model: nn.Module,
    path: Path | str,
    *,
    meta: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta or {}}, path)
    return path


def load_checkpoint(
    model: nn.Module,
    path: Path | str,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    model.load_state_dict(payload["state_dict"])
    return payload.get("meta", {})


def pack_linear_weight(weight: torch.Tensor) -> dict[str, Any]:
    """Pack ±1 signs of a Linear weight into uint64 + scale alpha."""
    w = weight.detach().float().cpu()
    alpha = float(w.abs().mean().clamp(min=1e-4).item())
    pm1 = binary_sign(w).numpy().astype(np.float32)
    packed, n = pack_binary_pm1(pm1, axis=1)
    return {
        "packed": packed,
        "n": int(n),
        "out_features": int(w.shape[0]),
        "in_features": int(w.shape[1]),
        "alpha": alpha,
        "fp32_bytes": int(w.numel() * 4),
        "packed_bytes": int(packed.nbytes),
        "compression": (w.numel() * 4) / max(packed.nbytes, 1),
    }


def save_packed_linears(model: nn.Module, path: Path | str) -> Path:
    """Export all nn.Linear modules as packed binary blobs (no FP weight copy)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable: dict[str, Any] = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        blob = pack_linear_weight(mod.weight)
        if mod.bias is not None:
            blob["bias"] = mod.bias.detach().float().cpu().numpy()
        else:
            blob["bias"] = None
        serializable[name] = blob
    torch.save(serializable, path)
    return path


def load_packed_linears(path: Path | str) -> dict[str, Any]:
    return torch.load(Path(path), map_location="cpu", weights_only=False)
