"""``.bnnpack`` format: packed binary Linear blobs + scales + metadata.

Format (torch.save dict, magic-gated):
  magic: \"BNNPACK1\"
  version: 1
  layers: { name -> layer_blob }
  meta: optional dict

Layer blob (binary_xnor):
  kind, in_features, out_features, n, weight_packed_i64 (int64 view of uint64),
  alpha (float32 tensor), bias (optional float32), fp32_bytes, packed_bytes

Security: ``load_bnnpack`` requires ``torch.load(..., weights_only=True)`` — no
unsafe pickle fallback (untrusted ``.bnnpack`` files must not execute code).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ..kernels.packed import pack_binary_pm1
from ..layers import BinaryLinear
from ..ste import binary_sign
from ..wrap.packed_linear import PackedBinaryXNORLinear, sign_pm1
from ..wrap.policy import HYBRID_FFN_SKIP

BNNPACK_MAGIC = "BNNPACK1"
BNNPACK_VERSION = 1


def unpack_binary_pm1(packed: np.ndarray, n: int) -> np.ndarray:
    """Inverse of ``pack_binary_pm1`` for 2D (rows, words) uint64 → (rows, n) ±1."""
    packed = np.ascontiguousarray(packed, dtype=np.uint64)
    if packed.ndim != 2:
        raise ValueError(f"expected 2D packed, got ndim={packed.ndim}")
    rows, words = packed.shape
    u8 = packed.astype("<u8", copy=False).view(np.uint8).reshape(rows, words, 8)
    bits = np.unpackbits(u8, axis=-1, bitorder="little")  # (rows, words, 64)
    bits = bits.reshape(rows, words * 64)[:, :n]
    return np.where(bits.astype(bool), np.float32(-1.0), np.float32(1.0))


def encode_linear_state(
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    alpha: torch.Tensor | None = None,
    name: str = "linear",
) -> dict[str, Any]:
    """Encode one Linear weight into a portable packed blob (binary XNOR path).

    Accepts FP ``nn.Linear`` weights or ``BinaryLinear`` latents (signed via STE).
    Compression is **exact 32×** when ``in_features % 64 == 0`` (no pad words);
    otherwise slightly lower due to uint64 padding.
    """
    w = weight.detach().float().cpu()
    out_f, in_f = int(w.shape[0]), int(w.shape[1])
    if alpha is None:
        alpha_t = w.abs().mean(dim=1).clamp(min=1e-4)
    else:
        alpha_t = alpha.detach().float().reshape(-1).cpu()
        if alpha_t.numel() == 1:
            alpha_t = alpha_t.expand(out_f)
        if alpha_t.numel() != out_f:
            raise ValueError(f"alpha length {alpha_t.numel()} != out_features {out_f}")
    w_pm1 = sign_pm1(w).numpy().astype(np.float32)
    packed, n = pack_binary_pm1(w_pm1, axis=1)
    assert n == in_f
    wp_i64 = torch.from_numpy(np.ascontiguousarray(packed).view(np.int64).copy())
    blob: dict[str, Any] = {
        "kind": "binary_xnor",
        "name": name,
        "in_features": in_f,
        "out_features": out_f,
        "n": int(n),
        "weight_packed_i64": wp_i64,
        "alpha": alpha_t.contiguous().clone(),
        "fp32_bytes": int(w.numel() * 4),
        "packed_bytes": int(packed.nbytes),
        "compression": float((w.numel() * 4) / max(packed.nbytes, 1)),
    }
    if bias is not None:
        blob["bias"] = bias.detach().float().cpu().contiguous().clone()
    else:
        blob["bias"] = None
    return blob


def decode_to_packed_linear(blob: dict[str, Any]) -> PackedBinaryXNORLinear:
    """Rebuild a ``PackedBinaryXNORLinear`` from an encoded blob (no FP weights)."""
    if blob.get("kind") != "binary_xnor":
        raise ValueError(f"unsupported layer kind: {blob.get('kind')}")
    in_f = int(blob["in_features"])
    out_f = int(blob["out_features"])
    # Minimal construct: ones weight, then overwrite packed buffers (no large random).
    fake_w = torch.ones(out_f, in_f)
    bias = blob.get("bias")
    alpha = blob["alpha"]
    if isinstance(alpha, torch.Tensor) and alpha.numel() not in (1, out_f):
        raise ValueError(f"alpha numel {alpha.numel()} incompatible with out={out_f}")
    mod = PackedBinaryXNORLinear(fake_w, bias, alpha=alpha)
    wp = blob["weight_packed_i64"]
    if not isinstance(wp, torch.Tensor):
        wp = torch.as_tensor(wp)
    expected_words = (in_f + 63) // 64
    if wp.numel() != out_f * expected_words:
        raise ValueError(
            f"packed size {wp.numel()} != out*words {out_f * expected_words}"
        )
    mod.weight_packed_i64.copy_(wp.to(dtype=torch.int64).cpu().reshape_as(mod.weight_packed_i64))
    mod._sync_numpy_views()
    return mod


def encode_from_packed_module(mod: PackedBinaryXNORLinear, *, name: str = "linear") -> dict[str, Any]:
    """Serialize an already-packed module into a bnnpack layer blob."""
    blob: dict[str, Any] = {
        "kind": "binary_xnor",
        "name": name,
        "in_features": int(mod.in_features),
        "out_features": int(mod.out_features),
        "n": int(mod._n),
        "weight_packed_i64": mod.weight_packed_i64.detach().cpu().contiguous().clone(),
        "alpha": mod.alpha.detach().float().cpu().contiguous().clone(),
        "fp32_bytes": int(mod.in_features * mod.out_features * 4),
        "packed_bytes": int(mod.packed_weight_bytes()),
        "compression": float(
            (mod.in_features * mod.out_features * 4) / max(mod.packed_weight_bytes(), 1)
        ),
    }
    if mod.bias is not None:
        blob["bias"] = mod.bias.detach().float().cpu().contiguous().clone()
    else:
        blob["bias"] = None
    return blob


def encode_model_linears(
    model: nn.Module,
    *,
    skip_name_substr: tuple[str, ...] | None = None,
    min_in_features: int = 1,
    include_packed: bool = True,
    include_binary_linear: bool = True,
    include_fp_linear: bool = False,
) -> dict[str, Any]:
    """Encode modules into a layers dict.

    Defaults favor the thesis wrap story:
    - Already-packed ``PackedBinaryXNORLinear`` (post-wrap FFN)
    - ``BinaryLinear`` STE modules
    - **Not** arbitrary FP ``nn.Linear`` (avoids silently binary-packing attn/embed/head)

    Set ``include_fp_linear=True`` only when you intentionally want cold PTQ of FP
    Linears; then ``skip_name_substr`` defaults to ``HYBRID_FFN_SKIP``.
    """
    if skip_name_substr is None:
        skip_name_substr = HYBRID_FFN_SKIP if include_fp_linear else ()
    layers: dict[str, Any] = {}
    for name, mod in model.named_modules():
        lname = name.lower()
        if any(s.lower() in lname for s in skip_name_substr):
            continue
        if include_packed and isinstance(mod, PackedBinaryXNORLinear):
            if mod.in_features < min_in_features:
                continue
            layers[name] = encode_from_packed_module(mod, name=name)
            continue
        if include_binary_linear and isinstance(mod, BinaryLinear):
            if mod.weight.shape[1] < min_in_features:
                continue
            a = mod.alpha.detach() if hasattr(mod, "alpha") else None
            bias = mod.bias if getattr(mod, "bias", None) is not None else None
            layers[name] = encode_linear_state(mod.weight, bias, alpha=a, name=name)
            continue
        if include_fp_linear and isinstance(mod, nn.Linear):
            if mod.weight.shape[1] < min_in_features:
                continue
            bias = mod.bias if getattr(mod, "bias", None) is not None else None
            layers[name] = encode_linear_state(mod.weight, bias, name=name)
    return layers


def save_bnnpack(
    layers: dict[str, Any],
    path: Path | str,
    *,
    meta: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "magic": BNNPACK_MAGIC,
        "version": BNNPACK_VERSION,
        "layers": layers,
        "meta": meta or {},
    }
    torch.save(payload, path)
    return path


def load_bnnpack(path: Path | str) -> dict[str, Any]:
    """Load ``.bnnpack`` with ``weights_only=True`` only (no unsafe pickle fallback)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(
            f"{path}: failed weights_only load ({exc}). "
            "Refusing unsafe pickle fallback for .bnnpack — regenerate the pack "
            "with a current bnn encode, or use only trusted sources."
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a bnnpack dict")
    if payload.get("magic") != BNNPACK_MAGIC:
        raise ValueError(
            f"{path}: bad magic {payload.get('magic')!r}; expected {BNNPACK_MAGIC}"
        )
    if int(payload.get("version", 0)) != BNNPACK_VERSION:
        raise ValueError(f"{path}: unsupported version {payload.get('version')}")
    if "layers" not in payload or not isinstance(payload["layers"], dict):
        raise ValueError(f"{path}: missing layers")
    return payload


def encode_file(
    model: nn.Module,
    path: Path | str,
    *,
    meta: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Path:
    layers = encode_model_linears(model, **kwargs)
    return save_bnnpack(layers, path, meta=meta)


def decode_file(
    path: Path | str,
) -> tuple[dict[str, PackedBinaryXNORLinear], dict[str, Any]]:
    """Load ``.bnnpack`` → mapping name → PackedBinaryXNORLinear + meta."""
    payload = load_bnnpack(path)
    modules = {
        name: decode_to_packed_linear(blob) for name, blob in payload["layers"].items()
    }
    meta = payload.get("meta") or {}
    return modules, meta if isinstance(meta, dict) else {}


def packed_module_fp_err(
    mod: PackedBinaryXNORLinear,
    *,
    batch: int = 4,
    seed: int = 0,
) -> float:
    """Max |packed GEMM − (±1 FP matmul × alpha)|; expect 0 for aligned packs."""
    rng = np.random.default_rng(seed)
    n = int(mod.in_features)
    x_pm1 = rng.choice([-1.0, 1.0], size=(batch, n)).astype(np.float32)
    w_pm1 = unpack_binary_pm1(mod._wp_np, n)
    ref = (x_pm1 @ w_pm1.T) * mod._alpha_np
    y = mod.gemm_only(x_pm1)
    return float(np.max(np.abs(y - ref)))


def roundtrip_gemm_err(
    weight: torch.Tensor,
    *,
    batch: int = 4,
    seed: int = 0,
) -> dict[str, float]:
    """Encode → decode → compare packed GEMM vs ±1 FP reference; expect err=0."""
    from ..kernels.packed import native_kernel_available

    blob = encode_linear_state(weight)
    mod = decode_to_packed_linear(blob)
    err = packed_module_fp_err(mod, batch=batch, seed=seed)
    return {
        "max_abs_err": err,
        "compression": float(blob["compression"]),
        "native": float(native_kernel_available()),
        "packed_bytes": float(blob["packed_bytes"]),
        "fp32_bytes": float(blob["fp32_bytes"]),
    }
