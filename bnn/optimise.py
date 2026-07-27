"""Public optimiser entrypoint (W1 product API).

Prefer::

    from bnn.optimise import optimise_model

over ad-hoc script wiring. See ``docs/adr/0001_public_optimiser_api.md``.
"""

from __future__ import annotations

import copy
import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from .wrap.api import WrapReport, attach_effectiveness, model_param_bytes, wrap_model
from .wrap.calibrate import CalibConfig
from .wrap.metrics import measure_agreement
from .wrap.policy import WrapMode, WrapPolicy
from .wrap.qat import light_qat_recover
from .wrap.schema import SCHEMA_ID, SCHEMA_VERSION, envelope, validate_optimise_report

__all__ = [
    "OptimiseConfig",
    "OptimiseResult",
    "optimise_model",
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "validate_optimise_report",
]


@dataclass
class OptimiseConfig:
    """Knob set for ``optimise_model`` (stable defaults)."""

    policy: WrapPolicy | str = "auto"
    mode: WrapMode | str | None = None
    min_in_features: int = 64
    min_out_features: int = 0
    skip_attn: bool = True
    skip_name_substr: Iterable[str] | None = None
    calib: CalibConfig | None = field(default_factory=CalibConfig)
    qat_steps: int = 0
    qat_layer_names: list[str] | None = None
    drop_in_threshold: float = 0.85
    force: bool = False
    accuracy_first: bool = False
    inplace: bool = False
    encode_path: Path | str | None = None
    encode_min_width: int = 64
    # W3.T05 — optional layer-wise sensitivity before wrap
    sensitivity: bool = False
    sensitivity_fragile_drop: float = 0.05


@dataclass
class OptimiseResult:
    """Product result: wrapped model + versioned report (+ optional pack path)."""

    model: nn.Module
    report: WrapReport
    payload: dict[str, Any]
    pack_path: Path | None = None
    bytes_before: dict[str, int] | None = None
    bytes_after: dict[str, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)


def optimise_model(
    model: nn.Module,
    calib_inputs: Tensor | None = None,
    config: OptimiseConfig | None = None,
    *,
    teacher: nn.Module | None = None,
    **kwargs: Any,
) -> OptimiseResult:
    """Calibrate → (optional QAT) → wrap → effectiveness → optional ``.bnnpack``.

    Parameters
    ----------
    model:
        FP (or mixed) ``nn.Module`` to optimise for CPU/edge packed inference.
    calib_inputs:
        Optional batch used for agreement metrics and light QAT. Shape must match
        the model's forward.
    config:
        ``OptimiseConfig``; keyword overrides also accepted via ``kwargs``.
    teacher:
        Optional FP teacher for agreement; defaults to a deepcopy taken *before*
        wrap when ``calib_inputs`` is provided.

    Returns
    -------
    OptimiseResult
        Wrapped model, ``WrapReport``, and a ``bnn_optimise_report_v1`` payload.
    """
    cfg = config or OptimiseConfig()
    # Allow kwargs overrides for ergonomics
    for k, v in kwargs.items():
        if not hasattr(cfg, k):
            raise TypeError(f"unknown OptimiseConfig field: {k}")
        setattr(cfg, k, v)

    work = model if cfg.inplace else copy.deepcopy(model)
    teacher_mod = teacher
    if calib_inputs is not None and teacher_mod is None:
        teacher_mod = copy.deepcopy(work)
        teacher_mod.eval()

    qat_info: dict[str, Any] | None = None
    mode = cfg.mode
    policy = cfg.policy
    use_binary_qat = (
        cfg.qat_steps > 0
        and policy not in ("ternary_wo",)
        and mode not in ("ternary_weight_only",)
    )
    if use_binary_qat and calib_inputs is not None:
        layer_names = cfg.qat_layer_names
        qat_info = light_qat_recover(
            work,
            calib_inputs,
            teacher=teacher_mod,
            steps=cfg.qat_steps,
            lr=1e-3,
            layer_names=layer_names,
        )

    sensitivity_payload: dict[str, Any] | None = None
    exclude_exact: list[str] | None = None
    if cfg.sensitivity:
        if calib_inputs is None:
            warnings.warn(
                "OptimiseConfig.sensitivity=True requires calib_inputs; "
                "sensitivity scoring skipped.",
                stacklevel=2,
            )
        else:
            from .wrap.sensitivity import score_layer_sensitivity

            sens_mode = "ternary_weight_only" if cfg.accuracy_first else "binary_xnor"
            if mode in ("ternary_weight_only", "binary_xnor"):
                sens_mode = mode
            sens = score_layer_sensitivity(
                work,
                calib_inputs,
                mode=sens_mode,  # type: ignore[arg-type]
                policy="all_large_linear",
                min_in_features=cfg.min_in_features,
                min_out_features=cfg.min_out_features,
                drop_in_threshold=cfg.drop_in_threshold,
                fragile_drop=cfg.sensitivity_fragile_drop,
                calib=cfg.calib,
            )
            sensitivity_payload = sens.to_dict()
            exclude_exact = list(sens.skip_suggested)

    before = model_param_bytes(work)
    wrapped, report = wrap_model(
        work,
        mode=mode,
        policy=policy,  # type: ignore[arg-type]
        skip_name_substr=cfg.skip_name_substr,
        min_in_features=cfg.min_in_features,
        min_out_features=cfg.min_out_features,
        skip_attn=cfg.skip_attn,
        calib=cfg.calib,
        inplace=True,
        accuracy_first=cfg.accuracy_first,
        exclude_exact=exclude_exact,
        force_narrow=cfg.force,
    )
    after = model_param_bytes(wrapped)

    if qat_info is not None:
        report.qat = qat_info

    status = "OK"
    if calib_inputs is not None and teacher_mod is not None:
        with torch.no_grad():
            t_logits = teacher_mod(calib_inputs)
            s_logits = wrapped(calib_inputs)
        eff = measure_agreement(
            t_logits, s_logits, drop_in_threshold=cfg.drop_in_threshold
        )
        attach_effectiveness(report, eff, force=cfg.force)
        if not report.drop_in_ok and not cfg.force:
            status = "REFUSE_DROP_IN_CLAIM"
        elif report.forced:
            status = "FORCED"
    elif cfg.force:
        status = "FORCED_NO_CALIB"
        report.forced = True

    pack_path: Path | None = None
    if cfg.encode_path is not None:
        from .codec import encode_file

        pack_path = Path(cfg.encode_path)
        encode_file(
            wrapped,
            pack_path,
            meta={
                "source": "bnn.optimise",
                "policy": report.policy,
                "mode": report.mode,
            },
            min_in_features=cfg.encode_min_width,
            include_binary_linear=True,
            include_fp_linear=False,
            include_packed=True,
        )

    payload = envelope(
        policy=str(report.policy),
        mode=str(report.mode),
        replaced=report.replaced,
        skipped=report.skipped,
        compression_replaced_weights=report.compression,
        fp32_weight_bytes_replaced=report.fp32_weight_bytes_replaced,
        packed_weight_bytes=report.packed_weight_bytes,
        native_kernel=bool(report.native_kernel),
        drop_in_ok=report.drop_in_ok,
        forced=bool(report.forced),
        status=status,
        policy_reason=report.policy_reason,
        calib_method=report.calib_method,
        effectiveness=report.effectiveness,
        qat=report.qat,
        sensitivity=sensitivity_payload,
        param_bytes_before=before,
        param_bytes_after=after,
        pack_path=str(pack_path) if pack_path else None,
    )
    errs = validate_optimise_report(payload)
    if errs:
        warnings.warn(f"optimise report schema issues: {errs}", stacklevel=2)

    return OptimiseResult(
        model=wrapped,
        report=report,
        payload=payload,
        pack_path=pack_path,
        bytes_before=before,
        bytes_after=after,
    )
