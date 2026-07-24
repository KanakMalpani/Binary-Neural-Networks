#!/usr/bin/env python3
"""Run verification gates and regenerate results/SUMMARY.md."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bnn.eval_report import write_summary  # noqa: E402


def run(cmd: list[str]) -> int:
    print(">", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(ROOT))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=ROOT / "results" / "SUMMARY.md")
    p.add_argument("--full", action="store_true", help="Also run CIFAR proxy (slow)")
    p.add_argument("--skip-pytest", action="store_true")
    args = p.parse_args()

    steps = [
        [sys.executable, "scripts/export_check.py"],
        [sys.executable, "scripts/validate_native.py"],
        [sys.executable, "scripts/benchmark.py", "--reps", "3"],
        [sys.executable, "scripts/energy_bound_measured.py"],
    ]
    if not args.skip_pytest:
        steps.insert(0, [sys.executable, "-m", "pytest", "-q"])
    if args.full:
        steps.append(
            [
                sys.executable,
                "scripts/train_image.py",
                "--epochs",
                "1",
                "--train-subset",
                "2000",
                "--channels",
                "32",
            ]
        )
        steps.append(
            [
                sys.executable,
                "scripts/train_audio.py",
                "--epochs",
                "1",
                "--n-train",
                "64",
                "--n-test",
                "32",
                "--n-classes",
                "4",
                "--channels",
                "16",
            ]
        )

    failed = []
    for cmd in steps:
        code = run(cmd)
        if code != 0:
            failed.append((cmd, code))

    out = write_summary(args.out)
    print(f"Wrote {out}")

    # Soft report of golden floors
    floors = json.loads((ROOT / "tests" / "golden_floors.json").read_text(encoding="utf-8"))
    print("golden_floors:", json.dumps(floors, indent=2))

    if failed:
        print("FAILED STEPS:", failed, file=sys.stderr)
        return 1
    print("eval-suite: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
