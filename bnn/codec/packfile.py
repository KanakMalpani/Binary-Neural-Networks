"""``.bnnpack`` format: packed binary / ternary / Conv2d blobs + metadata.

Format (torch.save dict, magic-gated):
  magic: \"BNNPACK1\"
  version: 1 | 2   (writers default to 2)
  layers: { name -> layer_blob }
  meta: optional dict
  hashes: optional { name -> sha256 }  (v2)

Layer kinds:
  binary_xnor — Linear uint64 pack (v1 compatible)
  ternary_weight_only — 2-bit packed {-1,0,+1} + scale
  binary_conv_packed — flattened Conv2d row-pack + per-out alpha

Security: ``load_bnnpack`` requires ``torch.load(..., weights_only=True)`` — no
unsafe pickle fallback (untrusted ``.bnnpack`` files must not execute code).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ..kernels.packed import pack_binary_pm1
from ..kernels.ternary_pack import pack_ternary_2bit, unpack_ternary_2bit
from ..layers import BinaryLinear
from ..wrap.packed_linear import (
    PackedBinaryConv2d,
    PackedBinaryXNORLinear,
    TernaryWeightOnlyLinear,
    absmean_ternary,
    absmean_ternary_per_channel,
    sign_pm1,
)
from ..wrap.policy import HYBRID_FFN_SKIP

BNNPACK_MAGIC = "BNNPACK1"
BNNPACK_VERSION_V1 = 1
BNNPACK_VERSION_V2 = 2
BNNPACK_VERSION = BNNPACK_VERSION_V2
SUPPORTED_VERSIONS = frozenset({BNNPACK_VERSION_V1, BNNPACK_VERSION_V2})

KIND_BINARY_XNOR = "binary_xnor"
KIND_TERNARY = "ternary_weight_only"
KIND_BINARY_CONV = "binary_conv_packed"

# Stable int codes for safetensors side-car tags (not Python hash()).
KIND_CODE: dict[str, int] = {
    KIND_BINARY_XNOR: 1,
    KIND_TERNARY: 2,
    KIND_BINARY_CONV: 3,
}


def _optional_bias_tensor(bias: Any) -> torch.Tensor | None:
    """Normalize ``nn.Parameter | Tensor | None`` → ``Tensor | None`` for mypy."""
    if bias is None:
        return None
    if isinstance(bias, torch.Tensor):
        return bias.detach()
    return torch.as_tensor(bias).detach()


def _symmetric_hw_int(value: Any, *, name: str) -> int:
    """Accept scalar or equal HxW pair; reject asymmetric / string Conv2d args."""
    if isinstance(value, str):
        raise ValueError(
            f"string Conv2d {name}={value!r} not supported (use integer padding/stride)"
        )
    if isinstance(value, (tuple, list)):
        if len(value) == 0:
            raise ValueError(f"empty Conv2d {name}")
        if len(value) == 1:
            return int(value[0])
        if len(value) != 2 or int(value[0]) != int(value[1]):
            raise ValueError(
                f"asymmetric Conv2d {name}={tuple(value)!r} not supported "
                "(encode requires equal H/W)"
            )
        return int(value[0])
    return int(value)


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


def content_sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def content_sha256_tensor(t: torch.Tensor) -> str:
    arr = np.ascontiguousarray(t.detach().cpu().numpy())
    return content_sha256_bytes(arr.tobytes())


def _attach_hash(blob: dict[str, Any], packed: torch.Tensor | None = None) -> dict[str, Any]:
    if packed is None:
        for key in ("weight_packed_i64", "weight_packed_u8"):
            if key in blob and blob[key] is not None:
                packed = blob[key]
                break
    if packed is not None and isinstance(packed, torch.Tensor):
        blob["content_sha256"] = content_sha256_tensor(packed)
    return blob


def encode_linear_state(
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    alpha: torch.Tensor | None = None,
    name: str = "linear",
    with_hash: bool = True,
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
        "kind": KIND_BINARY_XNOR,
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
    return _attach_hash(blob, wp_i64) if with_hash else blob


def encode_ternary_state(
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    scale: torch.Tensor | None = None,
    per_channel: bool = True,
    name: str = "ternary",
    with_hash: bool = True,
) -> dict[str, Any]:
    """Encode Linear weights as absmean ternary + 2-bit pack (size pedagogy)."""
    w = weight.detach().float().cpu()
    out_f, in_f = int(w.shape[0]), int(w.shape[1])
    if scale is None:
        if per_channel:
            q, scale_t = absmean_ternary_per_channel(w)
        else:
            q, scale_t = absmean_ternary(w)
            scale_t = scale_t.reshape(())
    else:
        scale_t = scale.detach().float().cpu()
        if per_channel:
            if scale_t.numel() == 1:
                scale_t = scale_t.expand(out_f)
            q = (w / scale_t.reshape(-1, 1)).round().clamp(-1, 1).to(torch.int8)
        else:
            q = (w / scale_t.reshape(())).round().clamp(-1, 1).to(torch.int8)
    q_np = q.detach().cpu().numpy().astype(np.int8)
    packed_u8 = pack_ternary_2bit(q_np)
    wp_u8 = torch.from_numpy(np.ascontiguousarray(packed_u8).copy())
    theoretical_bytes = (out_f * in_f * 2 + 7) // 8
    blob: dict[str, Any] = {
        "kind": KIND_TERNARY,
        "name": name,
        "in_features": in_f,
        "out_features": out_f,
        "per_channel": bool(per_channel),
        "weight_packed_u8": wp_u8,
        "scale": scale_t.detach().float().cpu().contiguous().clone()
        if isinstance(scale_t, torch.Tensor)
        else torch.tensor(float(scale_t), dtype=torch.float32),
        "fp32_bytes": int(w.numel() * 4),
        "packed_bytes": int(theoretical_bytes),
        "compression": float((w.numel() * 4) / max(theoretical_bytes, 1)),
        "compression_kind": "theoretical_2bit",
    }
    if bias is not None:
        blob["bias"] = bias.detach().float().cpu().contiguous().clone()
    else:
        blob["bias"] = None
    return _attach_hash(blob, wp_u8) if with_hash else blob


def encode_conv_state(
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    stride: int = 1,
    padding: int = 0,
    alpha: torch.Tensor | None = None,
    name: str = "conv",
    with_hash: bool = True,
) -> dict[str, Any]:
    """Encode Conv2d ``(out, in, kh, kw)`` weights as packed ±1 rows (size path)."""
    w = weight.detach().float().cpu()
    if w.ndim != 4:
        raise ValueError(f"expected 4D Conv2d weight, got shape {tuple(w.shape)}")
    out_c, in_c, kh, kw = (int(x) for x in w.shape)
    if alpha is None:
        alpha_t = w.abs().mean(dim=(1, 2, 3)).clamp(min=1e-4)
    else:
        alpha_t = alpha.detach().float().reshape(-1).cpu()
        if alpha_t.numel() == 1:
            alpha_t = alpha_t.expand(out_c)
        if alpha_t.numel() != out_c:
            raise ValueError(f"alpha length {alpha_t.numel()} != out_channels {out_c}")
    w_pm1 = sign_pm1(w).numpy().astype(np.float32)
    flat = w_pm1.reshape(out_c, -1)
    packed, n = pack_binary_pm1(flat, axis=1)
    wp_i64 = torch.from_numpy(np.ascontiguousarray(packed).view(np.int64).copy())
    blob: dict[str, Any] = {
        "kind": KIND_BINARY_CONV,
        "name": name,
        "in_channels": in_c,
        "out_channels": out_c,
        "kernel_size": (kh, kw),
        "stride": int(stride),
        "padding": int(padding),
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
    return _attach_hash(blob, wp_i64) if with_hash else blob


def decode_to_packed_linear(blob: dict[str, Any]) -> PackedBinaryXNORLinear:
    """Rebuild a ``PackedBinaryXNORLinear`` from an encoded blob (no FP weights)."""
    if blob.get("kind") != KIND_BINARY_XNOR:
        raise ValueError(f"unsupported layer kind: {blob.get('kind')}")
    in_f = int(blob["in_features"])
    out_f = int(blob["out_features"])
    fake_w = torch.ones(out_f, in_f)
    bias_raw = blob.get("bias")
    bias: torch.Tensor | None
    if bias_raw is None:
        bias = None
    elif isinstance(bias_raw, torch.Tensor):
        bias = bias_raw
    else:
        bias = torch.as_tensor(bias_raw)
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


def decode_to_ternary_linear(blob: dict[str, Any]) -> TernaryWeightOnlyLinear:
    """Rebuild ``TernaryWeightOnlyLinear`` from a v2 ternary blob."""
    if blob.get("kind") != KIND_TERNARY:
        raise ValueError(f"unsupported layer kind: {blob.get('kind')}")
    in_f = int(blob["in_features"])
    out_f = int(blob["out_features"])
    per_channel = bool(blob.get("per_channel", True))
    packed = blob["weight_packed_u8"]
    if not isinstance(packed, torch.Tensor):
        packed = torch.as_tensor(packed)
    q = unpack_ternary_2bit(packed.detach().cpu().numpy(), out_f, in_f)
    scale = blob["scale"]
    if not isinstance(scale, torch.Tensor):
        scale = torch.as_tensor(scale, dtype=torch.float32)
    fake = torch.ones(out_f, in_f)
    bias_raw = blob.get("bias")
    bias_t: torch.Tensor | None
    if bias_raw is None:
        bias_t = None
    elif isinstance(bias_raw, torch.Tensor):
        bias_t = bias_raw
    else:
        bias_t = torch.as_tensor(bias_raw)
    mod = TernaryWeightOnlyLinear(fake, bias_t, per_channel=per_channel)
    mod.weight_q.copy_(torch.from_numpy(np.ascontiguousarray(q)))
    scale_t = scale.detach().float().cpu().reshape(-1)
    if per_channel:
        if scale_t.numel() == 1:
            scale_t = scale_t.expand(out_f)
        mod.scale.copy_(scale_t.reshape_as(mod.scale))
    else:
        mod.scale.copy_(scale_t.reshape(()))
    return mod


def decode_to_packed_conv(blob: dict[str, Any]) -> PackedBinaryConv2d:
    """Rebuild ``PackedBinaryConv2d`` from a packed conv blob."""
    if blob.get("kind") != KIND_BINARY_CONV:
        raise ValueError(f"unsupported layer kind: {blob.get('kind')}")
    in_c = int(blob["in_channels"])
    out_c = int(blob["out_channels"])
    ks = blob["kernel_size"]
    if isinstance(ks, torch.Tensor):
        kh, kw = int(ks[0]), int(ks[1])
    else:
        kh, kw = int(ks[0]), int(ks[1])
    stride = int(blob.get("stride", 1))
    padding = int(blob.get("padding", 0))
    n = int(blob["n"])
    wp = blob["weight_packed_i64"]
    if not isinstance(wp, torch.Tensor):
        wp = torch.as_tensor(wp)
    packed = np.ascontiguousarray(wp.detach().cpu().numpy().view(np.uint64))
    if packed.ndim == 1:
        words = (n + 63) // 64
        packed = packed.reshape(out_c, words)
    w_pm1 = unpack_binary_pm1(packed, n).reshape(out_c, in_c, kh, kw)
    alpha = blob["alpha"]
    bias_raw = blob.get("bias")
    bias_t: torch.Tensor | None
    if bias_raw is None:
        bias_t = None
    elif isinstance(bias_raw, torch.Tensor):
        bias_t = bias_raw
    else:
        bias_t = torch.as_tensor(bias_raw)
    mod = PackedBinaryConv2d(
        torch.from_numpy(w_pm1),
        bias_t,
        stride=stride,
        padding=padding,
        alpha=alpha if isinstance(alpha, torch.Tensor) else torch.as_tensor(alpha),
    )
    mod.weight_packed_i64.copy_(
        wp.to(dtype=torch.int64).cpu().reshape_as(mod.weight_packed_i64)
    )
    mod._sync_numpy_views()
    return mod


def decode_layer(blob: dict[str, Any]) -> nn.Module:
    """Dispatch blob ``kind`` → packed module."""
    kind = blob.get("kind")
    if kind == KIND_BINARY_XNOR:
        return decode_to_packed_linear(blob)
    if kind == KIND_TERNARY:
        return decode_to_ternary_linear(blob)
    if kind == KIND_BINARY_CONV:
        return decode_to_packed_conv(blob)
    raise ValueError(f"unsupported layer kind: {kind!r}")


def encode_from_packed_module(mod: PackedBinaryXNORLinear, *, name: str = "linear") -> dict[str, Any]:
    """Serialize an already-packed module into a bnnpack layer blob."""
    blob: dict[str, Any] = {
        "kind": KIND_BINARY_XNOR,
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
    return _attach_hash(blob)


def encode_from_ternary_module(
    mod: TernaryWeightOnlyLinear, *, name: str = "ternary"
) -> dict[str, Any]:
    q = mod.weight_q.detach().cpu().numpy().astype(np.int8)
    packed_u8 = pack_ternary_2bit(q)
    wp_u8 = torch.from_numpy(np.ascontiguousarray(packed_u8).copy())
    out_f, in_f = int(mod.out_features), int(mod.in_features)
    theoretical_bytes = int(mod.packed_weight_bytes())
    blob: dict[str, Any] = {
        "kind": KIND_TERNARY,
        "name": name,
        "in_features": in_f,
        "out_features": out_f,
        "per_channel": bool(mod._per_channel),
        "weight_packed_u8": wp_u8,
        "scale": mod.scale.detach().float().cpu().contiguous().clone(),
        "fp32_bytes": int(in_f * out_f * 4),
        "packed_bytes": theoretical_bytes,
        "compression": float((in_f * out_f * 4) / max(theoretical_bytes, 1)),
        "compression_kind": "theoretical_2bit",
    }
    if mod.bias is not None:
        blob["bias"] = mod.bias.detach().float().cpu().contiguous().clone()
    else:
        blob["bias"] = None
    return _attach_hash(blob, wp_u8)


def encode_from_packed_conv(mod: PackedBinaryConv2d, *, name: str = "conv") -> dict[str, Any]:
    kh, kw = mod.kernel_size
    blob: dict[str, Any] = {
        "kind": KIND_BINARY_CONV,
        "name": name,
        "in_channels": int(mod.in_channels),
        "out_channels": int(mod.out_channels),
        "kernel_size": (int(kh), int(kw)),
        "stride": int(mod.stride),
        "padding": int(mod.padding),
        "n": int(mod._n),
        "weight_packed_i64": mod.weight_packed_i64.detach().cpu().contiguous().clone(),
        "alpha": mod.alpha.detach().float().cpu().contiguous().clone(),
        "fp32_bytes": int(mod.in_channels * mod.out_channels * kh * kw * 4),
        "packed_bytes": int(mod.packed_weight_bytes()),
        "compression": float(
            (mod.in_channels * mod.out_channels * kh * kw * 4)
            / max(mod.packed_weight_bytes(), 1)
        ),
    }
    if mod.bias is not None:
        blob["bias"] = mod.bias.detach().float().cpu().contiguous().clone()
    else:
        blob["bias"] = None
    return _attach_hash(blob)


def encode_model_linears(
    model: nn.Module,
    *,
    skip_name_substr: tuple[str, ...] | None = None,
    min_in_features: int = 1,
    include_packed: bool = True,
    include_binary_linear: bool = True,
    include_fp_linear: bool = False,
    include_ternary: bool = False,
    include_conv: bool = False,
) -> dict[str, Any]:
    """Encode modules into a layers dict.

    Defaults favor the thesis wrap story:
    - Already-packed ``PackedBinaryXNORLinear`` (post-wrap FFN)
    - ``BinaryLinear`` STE modules
    - **Not** arbitrary FP ``nn.Linear`` (avoids silently binary-packing attn/embed/head)

    Set ``include_fp_linear=True`` only when you intentionally want cold PTQ of FP
    Linears; then ``skip_name_substr`` defaults to ``HYBRID_FFN_SKIP``.
    Opt in to ternary / Conv2d with ``include_ternary`` / ``include_conv``.
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
        if include_ternary and isinstance(mod, TernaryWeightOnlyLinear):
            if mod.in_features < min_in_features:
                continue
            layers[name] = encode_from_ternary_module(mod, name=name)
            continue
        if include_conv and isinstance(mod, PackedBinaryConv2d):
            layers[name] = encode_from_packed_conv(mod, name=name)
            continue
        if include_binary_linear and isinstance(mod, BinaryLinear):
            if mod.weight.shape[1] < min_in_features:
                continue
            a = mod.alpha.detach() if hasattr(mod, "alpha") else None
            layers[name] = encode_linear_state(
                mod.weight,
                _optional_bias_tensor(mod.bias),
                alpha=a,
                name=name,
            )
            continue
        if include_fp_linear and isinstance(mod, nn.Linear):
            if mod.weight.shape[1] < min_in_features:
                continue
            layers[name] = encode_linear_state(
                mod.weight,
                _optional_bias_tensor(mod.bias),
                name=name,
            )
            continue
        if include_conv and isinstance(mod, nn.Conv2d) and mod.groups == 1:
            layers[name] = encode_conv_state(
                mod.weight,
                _optional_bias_tensor(mod.bias),
                stride=_symmetric_hw_int(mod.stride, name="stride"),
                padding=_symmetric_hw_int(mod.padding, name="padding"),
                name=name,
            )
    return layers


def _layer_hashes(layers: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for name, blob in layers.items():
        if isinstance(blob, dict) and "content_sha256" in blob:
            out[name] = str(blob["content_sha256"])
    return out


def save_bnnpack(
    layers: dict[str, Any],
    path: Path | str,
    *,
    meta: dict[str, Any] | None = None,
    version: int = BNNPACK_VERSION,
) -> Path:
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(
            f"unsupported bnnpack version {version}; want {sorted(SUPPORTED_VERSIONS)}"
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if version >= BNNPACK_VERSION_V2:
        for blob in layers.values():
            if isinstance(blob, dict) and "content_sha256" not in blob:
                _attach_hash(blob)
    payload: dict[str, Any] = {
        "magic": BNNPACK_MAGIC,
        "version": int(version),
        "layers": layers,
        "meta": meta or {},
    }
    if version >= BNNPACK_VERSION_V2:
        payload["hashes"] = _layer_hashes(layers)
    torch.save(payload, path)
    return path


def load_bnnpack(
    path: Path | str,
    *,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    """Load ``.bnnpack`` with ``weights_only=True`` only (no unsafe pickle fallback).

    When ``verify_hashes`` is True (default) and the file is v2+, recompute
    per-layer ``content_sha256`` and raise if any mismatch.
    """
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
    ver = int(payload.get("version", 0))
    if ver not in SUPPORTED_VERSIONS:
        raise ValueError(
            f"{path}: unsupported version {payload.get('version')}; "
            f"supported={sorted(SUPPORTED_VERSIONS)}"
        )
    if "layers" not in payload or not isinstance(payload["layers"], dict):
        raise ValueError(f"{path}: missing layers")
    if verify_hashes and ver >= BNNPACK_VERSION_V2:
        bad = verify_layer_hashes(payload)
        if bad:
            raise ValueError(
                f"{path}: content_sha256 mismatch for layers {bad}; "
                "file may be corrupted or tampered"
            )
    return payload


def verify_layer_hashes(payload: dict[str, Any]) -> list[str]:
    """Return list of layer names whose ``content_sha256`` mismatches recomputation."""
    mismatches: list[str] = []
    layers = payload.get("layers") or {}
    if not isinstance(layers, dict):
        return mismatches
    for name, blob in layers.items():
        if not isinstance(blob, dict):
            continue
        expected = blob.get("content_sha256")
        if not expected:
            continue
        packed = None
        for key in ("weight_packed_i64", "weight_packed_u8"):
            if blob.get(key) is not None:
                packed = blob[key]
                break
        if packed is None:
            continue
        if not isinstance(packed, torch.Tensor):
            packed = torch.as_tensor(packed)
        if content_sha256_tensor(packed) != expected:
            mismatches.append(name)
    return mismatches


def encode_file(
    model: nn.Module,
    path: Path | str,
    *,
    meta: dict[str, Any] | None = None,
    version: int = BNNPACK_VERSION,
    **kwargs: Any,
) -> Path:
    layers = encode_model_linears(model, **kwargs)
    return save_bnnpack(layers, path, meta=meta, version=version)


def decode_file(
    path: Path | str,
) -> tuple[dict[str, nn.Module], dict[str, Any]]:
    """Load ``.bnnpack`` → mapping name → packed module + meta."""
    payload = load_bnnpack(path)
    modules = {name: decode_layer(blob) for name, blob in payload["layers"].items()}
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
