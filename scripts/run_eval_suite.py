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
from bnn.profile import (  # noqa: E402
    check_committed_bench_soft_floors,
    check_soft_budgets,
    profile_packed_linear,
)


def run(cmd: list[str]) -> int:
    print(">", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(ROOT))


def _codec_seq_smokes() -> list[str]:
    """Focused codec + seq gates (W7.T08) — always part of eval-suite."""
    return [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_codec.py::test_roundtrip_gemm_err_zero",
        "tests/test_codec.py::test_bnnpack_file_roundtrip",
        "tests/test_seq_encoder_decoder.py::test_encoder_forward_shape",
        "tests/test_seq_encoder_decoder.py::test_decoder_causal_forward",
    ]


def _profile_soft_budget_step() -> int:
    """Soft latency budget smoke (W13.T03) — warn-only unless --strict-budgets."""
    br = profile_packed_linear(
        m=8, n=256, k=256, reps=3, warmup=1, compare_baselines=True
    )
    violations = check_soft_budgets(br)
    print(
        "profile soft budgets:",
        "PASS" if not violations else f"WARN {violations}",
        flush=True,
    )
    bench_path = ROOT / "results" / "benchmark.json"
    if bench_path.is_file():
        bench = json.loads(bench_path.read_text(encoding="utf-8"))
        floor_v = check_committed_bench_soft_floors(bench)
        print(
            "committed bench soft floors:",
            "PASS" if not floor_v else f"WARN {floor_v}",
            flush=True,
        )
        if floor_v:
            violations = [*violations, *floor_v]
    return 1 if violations else 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=ROOT / "results" / "SUMMARY.md")
    p.add_argument("--full", action="store_true", help="Also run CIFAR proxy (slow)")
    p.add_argument("--skip-pytest", action="store_true")
    p.add_argument(
        "--strict-budgets",
        action="store_true",
        help="Fail eval-suite when soft latency budgets are exceeded (default: warn)",
    )
    args = p.parse_args()

    steps = [
        [sys.executable, "scripts/export_check.py"],
        [sys.executable, "scripts/validate_native.py"],
        [sys.executable, "scripts/benchmark.py", "--reps", "3"],
        [sys.executable, "scripts/energy_bound_measured.py"],
        _codec_seq_smokes(),
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

    budget_code = _profile_soft_budget_step()
    if budget_code != 0 and args.strict_budgets:
        failed.append((["profile-soft-budgets"], budget_code))
    elif budget_code != 0:
        print("soft budgets exceeded (non-fatal without --strict-budgets)", flush=True)

    out = write_summary(args.out)
    print(f"Wrote {out}")

    floors = json.loads((ROOT / "tests" / "golden_floors.json").read_text(encoding="utf-8"))
    print("golden_floors:", json.dumps(floors, indent=2))

    if failed:
        print("FAILED STEPS:", failed, file=sys.stderr)
        return 1
    print("eval-suite: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
