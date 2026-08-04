"""High-level wrap API (policies, calib, report schema)."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field

import torch.nn as nn

from ..kernels.packed import native_kernel_available
from .calibrate import CalibConfig
from .metrics import EffectivenessReport, unmeasured_effectiveness
from .packed_linear import (
    BinaryWeightOnlyDequantLinear,
    PackedBinaryConv2d,
    PackedBinaryXNORLinear,
    PackedLinearLike,
    TernaryWeightOnlyLinear,
)
from .policy import (
    DEFAULT_SKIP,
    WrapMode,
    WrapPolicy,
    detect_hardware,
    recommend_wrap_policy,
    select_linears,
)


@dataclass
class WrapReport:
    mode: WrapMode
    policy: str = "default"
    replaced: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    fp32_weight_bytes_replaced: int = 0
    packed_weight_bytes: int = 0
    native_kernel: bool = False
    calib_method: str | None = None
    effectiveness: dict | None = None
    policy_reason: str | None = None
    qat: dict | None = None
    distill: dict | None = None
    fuse: dict | None = None
    drop_in_ok: bool | None = None
    forced: bool = False

    @property
    def compression(self) -> float:
        if self.packed_weight_bytes <= 0:
            return 0.0
        return self.fp32_weight_bytes_replaced / self.packed_weight_bytes

    def to_dict(self) -> dict:
        d = asdict(self)
        d["compression"] = self.compression
        return d


def _set_module(root: nn.Module, path: str, new: nn.Module) -> None:
    parts = path.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new)


def _build_wrapped(
    lin: nn.Linear,
    mode: WrapMode,
    *,
    calib: CalibConfig | None,
) -> tuple[nn.Module, int]:
    w = lin.weight.data
    b = lin.bias.data if lin.bias is not None else None
    new: PackedLinearLike
    if mode == "binary_xnor":
        new = PackedBinaryXNORLinear(w, b, calib=calib)
        packed_b = new.packed_weight_bytes()
    elif mode == "ternary_weight_only":
        per_ch = True if calib is None else calib.per_channel
        new = TernaryWeightOnlyLinear(w, b, per_channel=per_ch, calib=calib)
        packed_b = new.packed_weight_bytes()
    elif mode == "binary_weight_only_dequant":
        new = BinaryWeightOnlyDequantLinear(w, b)
        packed_b = new.packed_weight_bytes()
    else:
        raise ValueError(mode)
    return new, packed_b


def wrap_model(
    model: nn.Module,
    mode: WrapMode | str | None = None,
    *,
    policy: WrapPolicy = "hybrid_ffn",
    skip_name_substr: Iterable[str] | None = None,
    min_in_features: int = 64,
    min_out_features: int = 0,
    skip_attn: bool = True,
    calib: CalibConfig | None = None,
    inplace: bool = True,
    accuracy_first: bool = False,
    exclude_exact: Iterable[str] | None = None,
    force_narrow: bool = False,
    fuse_bn: bool = False,
    drop_in_threshold: float = 0.85,
) -> tuple[nn.Module, WrapReport]:
    """Product wrap API with hybrid / aggressive / ternary_wo / auto policies.

    ``mode=None`` means unspecified (default binary_xnor, or recommender when
    ``policy='auto'`` / ``mode='auto'``).

    ``exclude_exact``: full dotted module names to never wrap (sensitivity).
    ``force_narrow``: allow binary_xnor on shapes guardrails would refuse.
    ``fuse_bn``: fold Linear+BN1d / BiReal BN before packing (W3.T09).
    """
    import copy as _copy

    from .guardrails import check_linear_wrap_guardrails

    hw = detect_hardware()
    resolved_mode: WrapMode
    # W3.T03 — policy_reason is always a non-empty string
    policy_reason: str

    if policy == "auto" or mode == "auto":
        decision = recommend_wrap_policy(None, hw, accuracy_first=accuracy_first)
        if policy == "auto":
            policy = decision.policy
        if mode is None or mode == "auto":
            resolved_mode = decision.mode
        else:
            resolved_mode = mode  # type: ignore[assignment]
        policy_reason = decision.reason or (
            f"auto → policy={policy} mode={resolved_mode}"
        )
        if min_in_features == 64:
            min_in_features = decision.min_in_features
        if min_out_features == 0:
            min_out_features = decision.min_out_features
        skip_attn = decision.skip_attn
    elif policy == "ternary_wo":
        resolved_mode = "ternary_weight_only"
        policy_reason = "policy=ternary_wo (accurate-first weight-only)"
    else:
        resolved_mode = (mode or "binary_xnor")  # type: ignore[assignment]
        policy_reason = f"policy={policy} mode={resolved_mode}"

    if not inplace:
        model = _copy.deepcopy(model)

    fuse_payload: dict | None = None
    if fuse_bn:
        from .fuse import fuse_bn_for_wrap_

        fuse_payload = fuse_bn_for_wrap_(model).to_dict()

    to_replace, skipped = select_linears(
        model,
        policy=policy,
        skip_name_substr=skip_name_substr,
        min_in_features=min_in_features,
        min_out_features=min_out_features,
        skip_attn=skip_attn,
        exclude_exact=exclude_exact,
    )

    report = WrapReport(
        mode=resolved_mode,
        policy=policy,
        skipped=skipped,
        native_kernel=native_kernel_available(),
        calib_method=(calib.method if calib else "absmean"),
        policy_reason=policy_reason,
        fuse=fuse_payload,
        # W3.T02 — effectiveness always present (stub until attach_effectiveness)
        effectiveness=unmeasured_effectiveness(
            drop_in_threshold=drop_in_threshold
        ).to_dict(),
        drop_in_ok=False,
    )

    for name, lin in to_replace:
        verdict = check_linear_wrap_guardrails(
            lin, mode=str(resolved_mode), force=force_narrow
        )
        if not verdict.ok:
            report.skipped.append(f"{name} ({verdict.code}: {verdict.message})")
            continue
        fp_bytes = int(lin.weight.numel() * 4)
        new, packed_b = _build_wrapped(lin, resolved_mode, calib=calib)
        report.replaced.append(name)
        report.fp32_weight_bytes_replaced += fp_bytes
        report.packed_weight_bytes += packed_b
        if resolved_mode == "binary_xnor":
            report.native_kernel = getattr(new, "uses_native", False)
        _set_module(model, name, new)

    return model, report


def wrap_linear_modules(
    model: nn.Module,
    mode: WrapMode = "binary_xnor",
    *,
    skip_name_substr: Iterable[str] = DEFAULT_SKIP,
    min_in_features: int = 64,
    min_out_features: int = 0,
    calib: CalibConfig | None = None,
    inplace: bool = True,
) -> tuple[nn.Module, WrapReport]:
    """Legacy API: skip-list based wrap (still used by demos/tests)."""
    return wrap_model(
        model,
        mode,
        policy="default",
        skip_name_substr=skip_name_substr,
        min_in_features=min_in_features,
        min_out_features=min_out_features,
        calib=calib,
        inplace=inplace,
    )


def attach_effectiveness(
    report: WrapReport,
    eff: EffectivenessReport,
    *,
    force: bool = False,
) -> WrapReport:
    """Attach measured effectiveness; always keeps the field populated (W3.T02)."""
    report.effectiveness = eff.to_dict()
    if force:
        report.drop_in_ok = True
        report.forced = True
    else:
        from .metrics import drop_in_ok as _drop_in_ok

        report.drop_in_ok = _drop_in_ok(eff, force=False)
        report.forced = False
    if report.policy_reason is None:
        report.policy_reason = f"policy={report.policy} mode={report.mode}"
    return report


def model_param_bytes(model: nn.Module) -> dict:
    p_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    b_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return {"param_bytes": p_bytes, "buffer_bytes": b_bytes, "total_bytes": p_bytes + b_bytes}


def wrap_conv_modules(
    model: nn.Module,
    *,
    skip_name_substr: Iterable[str] = ("stem", "head", "skip"),
    min_weight_elems: int = 256,
    inplace: bool = True,
) -> tuple[nn.Module, WrapReport]:
    from ..layers import BinaryConv2d

    report = WrapReport(mode="binary_xnor", policy="conv", native_kernel=False)
    to_replace: list[tuple[str, BinaryConv2d | nn.Conv2d]] = []
    for name, mod in model.named_modules():
        if isinstance(mod, BinaryConv2d) or isinstance(mod, nn.Conv2d) and mod.groups == 1:
            pass
        else:
            continue
        lname = name.lower()
        if any(s.lower() in lname for s in skip_name_substr):
            report.skipped.append(f"{name} (skip list)")
            continue
        w = mod.weight
        if w.numel() < min_weight_elems:
            report.skipped.append(f"{name} (too small)")
            continue
        to_replace.append((name, mod))

    for name, mod in to_replace:
        w = mod.weight.data
        mod_bias = getattr(mod, "bias", None)
        b = mod_bias.data if mod_bias is not None else None
        stride = getattr(mod, "stride", 1)
        padding = getattr(mod, "padding", 0)
        if isinstance(stride, tuple):
            stride = stride[0]
        if isinstance(padding, tuple):
            padding = padding[0]
        alpha = getattr(mod, "alpha", None)
        if alpha is not None:
            alpha = alpha.detach().reshape(-1)
        new = PackedBinaryConv2d(w, b, stride=stride, padding=padding, alpha=alpha)
        report.replaced.append(name)
        report.fp32_weight_bytes_replaced += int(w.numel() * 4)
        report.packed_weight_bytes += new.packed_weight_bytes()
        if inplace:
            _set_module(model, name, new)
    return model, report
