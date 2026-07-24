"""Versioned optimiser / wrap report schema (W1.T06).

Schema id: ``bnn_optimise_report_v1``

Dual-metric rule
----------------
``compression_*`` fields are **theoretical** pack ratios (e.g. ~32× for aligned
uint64 binary). Latency / samples_per_s fields are **wall-clock**. Never conflate.
"""

from __future__ import annotations

from typing import Any

SCHEMA_ID = "bnn_optimise_report_v1"
SCHEMA_VERSION = 1

# Required top-level keys for a single-run optimise report.
REQUIRED_KEYS = frozenset(
    {
        "schema",
        "schema_version",
        "policy",
        "mode",
        "replaced",
        "skipped",
        "compression_replaced_weights",
        "fp32_weight_bytes_replaced",
        "packed_weight_bytes",
        "native_kernel",
        "drop_in_ok",
        "forced",
        "status",
    }
)

# Optional but recommended dual-metric / honesty fields.
RECOMMENDED_KEYS = frozenset(
    {
        "policy_reason",
        "effectiveness",
        "e2e_latency_ms_fp",
        "e2e_latency_ms_wrapped",
        "e2e_speedup",
        "thesis_note",
    }
)


def validate_optimise_report(payload: dict[str, Any], *, strict: bool = False) -> list[str]:
    """Return a list of validation errors (empty ⇒ OK).

    ``strict=True`` also requires recommended dual-metric keys when latency was measured.
    """
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["report must be a dict"]

    schema = payload.get("schema")
    if schema not in (SCHEMA_ID, "ultra_wrap_report_v1"):
        # Accept legacy ultra_wrap_report_v1 as alias during transition
        errors.append(f"schema must be {SCHEMA_ID!r} (or legacy ultra_wrap_report_v1), got {schema!r}")

    if payload.get("schema") == SCHEMA_ID:
        ver = payload.get("schema_version")
        if ver != SCHEMA_VERSION:
            errors.append(f"schema_version must be {SCHEMA_VERSION}, got {ver!r}")
        for key in REQUIRED_KEYS:
            if key not in payload:
                errors.append(f"missing required key: {key}")

    if strict:
        for key in ("effectiveness", "policy_reason"):
            if key not in payload:
                errors.append(f"strict: missing recommended key: {key}")

    # Honesty: compression alone must not be labeled as e2e speedup
    if "e2e_speedup" in payload and payload.get("e2e_speedup") is not None:
        try:
            if float(payload["e2e_speedup"]) > 20 and float(payload.get("compression_replaced_weights") or 0) > 20:
                # Soft warning only — store as note, not hard fail
                note = str(payload.get("thesis_note") or "")
                if "dual-metric" not in note.lower() and "theory" not in note.lower():
                    errors.append(
                        "strict honesty: high e2e_speedup + high compression requires "
                        "thesis_note mentioning dual-metric / theory vs wall-clock"
                    )
        except (TypeError, ValueError):
            pass

    return errors


def is_valid_optimise_report(payload: dict[str, Any], *, strict: bool = False) -> bool:
    return not validate_optimise_report(payload, strict=strict)


def envelope(
    *,
    policy: str,
    mode: str,
    replaced: list[str],
    skipped: list[str],
    compression_replaced_weights: float,
    fp32_weight_bytes_replaced: int,
    packed_weight_bytes: int,
    native_kernel: bool,
    drop_in_ok: bool | None,
    forced: bool,
    status: str,
    **extra: Any,
) -> dict[str, Any]:
    """Build a minimal valid ``bnn_optimise_report_v1`` dict."""
    out: dict[str, Any] = {
        "schema": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "policy": policy,
        "mode": mode,
        "replaced": list(replaced),
        "skipped": list(skipped),
        "compression_replaced_weights": float(compression_replaced_weights),
        "fp32_weight_bytes_replaced": int(fp32_weight_bytes_replaced),
        "packed_weight_bytes": int(packed_weight_bytes),
        "native_kernel": bool(native_kernel),
        "drop_in_ok": drop_in_ok,
        "forced": bool(forced),
        "status": status,
        "thesis_note": (
            "Compression is theoretical pack ratio; latency fields are wall-clock. "
            "Never claim GPU 32× from sign()/STE."
        ),
    }
    out.update(extra)
    return out
