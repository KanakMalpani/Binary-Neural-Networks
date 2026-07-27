#!/usr/bin/env python3
"""Generate a lightweight CycloneDX-ish SBOM for the installed environment (W8.T06).

Does not require cyclonedx-bom. Emits JSON listing direct + installed packages
from ``pip freeze`` / importlib.metadata. For a stricter CycloneDX document,
install ``cyclonedx-bom`` and run ``cyclonedx-py environment`` (documented in
``docs/SBOM.md``).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _pip_freeze() -> list[dict[str, str]]:
    r = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True,
        text=True,
        check=False,
    )
    comps: list[dict[str, str]] = []
    for line in (r.stdout or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if " @ " in line:
            name, loc = line.split(" @ ", 1)
            comps.append({"name": name.strip(), "version": "editable/url", "purl_hint": loc.strip()})
        elif "==" in line:
            name, ver = line.split("==", 1)
            comps.append({"name": name.strip(), "version": ver.strip()})
        else:
            comps.append({"name": line, "version": "unknown"})
    return comps


def main() -> int:
    p = argparse.ArgumentParser(description="Generate SBOM JSON for bnn env")
    p.add_argument("--out", type=Path, default=ROOT / "sbom.json")
    args = p.parse_args()

    # Package version
    try:
        from bnn._version import __version__ as bnn_ver
    except Exception:
        bnn_ver = "unknown"

    doc = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "metadata": {
            "timestamp": datetime.now(UTC).isoformat(),
            "component": {
                "type": "library",
                "name": "bnn",
                "version": bnn_ver,
                "bom-ref": f"pkg:pypi/bnn@{bnn_ver}",
            },
            "tools": [{"name": "scripts/generate_sbom.py", "vendor": "Binary Neural Network Lab"}],
            "note": (
                "Lightweight SBOM from pip freeze. For full CycloneDX 1.5, "
                "see docs/SBOM.md (cyclonedx-py)."
            ),
        },
        "components": _pip_freeze(),
    }
    args.out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"Wrote {args.out} ({len(doc['components'])} components)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
