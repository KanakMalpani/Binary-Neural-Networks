"""Optional transformers / torchao version probes (W14.T06)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.optional_extras
def test_optional_extras_smoke_script(tmp_path: Path):
    out = tmp_path / "opt.json"
    code = subprocess.call(
        [sys.executable, str(ROOT / "scripts" / "smoke_optional_extras.py"), "--out", str(out)],
        cwd=str(ROOT),
    )
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema"] == "bnn_optional_extras_smoke_v1"
    assert payload["probes"]
    for row in payload["probes"]:
        assert row["status"] in {"ok", "skipped", "error"}


def test_optional_extras_doc_exists():
    assert (ROOT / "docs" / "OPTIONAL_EXTRAS_MATRIX.md").is_file()
    assert (ROOT / "docs" / "TORCH_PIN_POLICY.md").is_file()
