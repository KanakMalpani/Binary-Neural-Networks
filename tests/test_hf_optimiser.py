"""Optional HF optimiser smoke (W5.T04).

Marked ``hf`` + ``slow`` — skipped in default ``bnn repro`` / fast CI.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("transformers")

torch = pytest.importorskip("torch")
nn = torch.nn

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.hf
@pytest.mark.slow
def test_hf_tiny_wrap_script(tmp_path: Path):

    out = tmp_path / "hf_wrap.json"
    # Prefer in-process import of the demo main for clearer failures
    import runpy
    import sys

    script = ROOT / "scripts" / "hf_tiny_wrap_demo.py"
    argv = ["hf_tiny_wrap_demo.py", "--out", str(out), "--model", "hf-internal-testing/tiny-random-BertModel"]
    old = sys.argv
    try:
        sys.argv = argv
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as exc:
        # runpy raises SystemExit on the *success* path too, so pytest.raises
        # would be wrong here — we assert the exit code, not that it raised.
        assert int(exc.code or 0) == 0  # noqa: PT017
    finally:
        sys.argv = old

    assert out.is_file()
    data = json.loads(out.read_text(encoding="utf-8"))
    if data.get("skipped"):
        pytest.skip(data.get("reason", "hf skipped"))
    assert "replaced" in data
    assert "warning" in data


@pytest.mark.hf
@pytest.mark.slow
def test_hf_wrap_model_api():
    from transformers import AutoModelForSequenceClassification

    from bnn import wrap_model

    model = AutoModelForSequenceClassification.from_pretrained(
        "hf-internal-testing/tiny-random-BertModel",
        num_labels=2,
    )
    _, report = wrap_model(model, policy="hybrid_ffn", min_in_features=32)
    # Tiny random BERT may replace 0–N FFN layers depending on names/widths
    assert isinstance(report.replaced, list)
    assert report.compression >= 0.0
