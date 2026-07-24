#!/usr/bin/env python3
"""Optional HF tiny wrap demo (requires bnn[hf])."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    try:
        from transformers import AutoModelForSequenceClassification
    except ImportError:
        print(
            json.dumps(
                {
                    "skipped": True,
                    "reason": "transformers not installed — pip install -e \".[hf]\"",
                }
            )
        )
        return 0

    import torch
    from bnn.wrapper import wrap_model, model_param_bytes

    p = argparse.ArgumentParser()
    p.add_argument("--model", default="hf-internal-testing/tiny-random-BertModel")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "hf_tiny_wrap.json")
    args = p.parse_args()

    # Prefer a tiny classification head model if available; fall back gracefully
    try:
        model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
    except Exception as e:
        print(json.dumps({"skipped": True, "reason": str(e)}))
        return 0

    before = model_param_bytes(model)
    _, report = wrap_model(model, policy="hybrid_ffn", min_in_features=32)
    after = model_param_bytes(model)
    payload = {
        "model": args.model,
        "replaced": report.replaced,
        "skipped": report.skipped[:20],
        "compression_replaced": report.compression,
        "bytes_before": before,
        "bytes_after": after,
        "warning": (
            "PTQ wrap without QAT — quality not guaranteed. "
            "For GPU LLMs use INT4/FP8; for BitNet use bitnet.cpp."
        ),
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
