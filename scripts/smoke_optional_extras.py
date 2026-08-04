#!/usr/bin/env python3
"""Probe optional extras (transformers / torchao) — W14.T06.

Never fails the core lab. Exit 0 when probes are skipped or pass; exit 2 only
when ``--require`` asks for an extra that is missing / broken.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _version(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def probe_hf() -> dict:
    row: dict = {"extra": "hf", "dist": "transformers"}
    ver = _version("transformers")
    if ver is None:
        row["status"] = "skipped"
        row["reason"] = "transformers not installed (pip install -e '.[hf]')"
        return row
    row["version"] = ver
    try:
        import transformers  # noqa: F401

        row["status"] = "ok"
        row["import"] = "transformers"
    except Exception as exc:  # pragma: no cover
        row["status"] = "error"
        row["error"] = str(exc)
    return row


def probe_torchao() -> dict:
    row: dict = {"extra": "torchao", "dist": "torchao"}
    ver = _version("torchao")
    if ver is None:
        row["status"] = "skipped"
        row["reason"] = "torchao not installed (GPU host recipe only)"
        return row
    row["version"] = ver
    try:
        import torchao  # noqa: F401

        row["status"] = "ok"
        row["import"] = "torchao"
        row["note"] = "Commodity GPU INT4/FP8 path — not classic BNN 32× CUDA"
    except Exception as exc:  # pragma: no cover
        row["status"] = "error"
        row["error"] = str(exc)
    return row


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probe", choices=("all", "hf", "torchao"), default="all")
    p.add_argument(
        "--require",
        action="append",
        default=[],
        choices=("hf", "torchao"),
        help="Fail (exit 2) if the named extra is missing or broken",
    )
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    probes = []
    if args.probe in ("all", "hf"):
        probes.append(probe_hf())
    if args.probe in ("all", "torchao"):
        probes.append(probe_torchao())

    payload = {
        "schema": "bnn_optional_extras_smoke_v1",
        "generated_utc": datetime.now(UTC).isoformat(),
        "probes": probes,
        "policy": "docs/OPTIONAL_EXTRAS_MATRIX.md",
    }
    out = args.out or (ROOT / "results" / "optional_extras_smoke.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))

    required = set(args.require)
    for row in probes:
        if row["extra"] in required and row.get("status") != "ok":
            print(f"REQUIRED extra {row['extra']} not ok: {row}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
