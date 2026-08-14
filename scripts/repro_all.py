#!/usr/bin/env python3
"""One-command reproducibility entrypoint (also exposed as ``bnn repro``).

Modes
-----
* ``verify`` (default, fast): compile-native (best-effort) → pytest →
  export-check → validate-native (skip if no DLL) → compare committed
  ``results/*.json`` against ``tests/golden_floors.json`` → regenerate SUMMARY.
* ``full`` (slow): same as verify, then optional short deterministic smoke
  trains that write under ``results/_repro_smoke_*.json`` (does **not**
  overwrite published goldens unless ``--overwrite-goldens``).
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bnn.eval_report import write_summary  # noqa: E402


def run(cmd: list[str], *, check: bool = False) -> int:
    print(">", " ".join(cmd), flush=True)
    code = subprocess.call(cmd, cwd=str(ROOT))
    if check and code != 0:
        raise SystemExit(code)
    return code


def compile_native_best_effort() -> tuple[str, int]:
    """Try MSVC native build; never hard-fail the verify path."""
    if platform.system() != "Windows":
        msg = "SKIP compile-native (non-Windows → NumPy fallback)"
        print(msg, flush=True)
        return msg, 0
    code = run([sys.executable, "-m", "bnn.kernels.compile_native"])
    if code != 0:
        msg = "WARN compile-native failed (MinGW/missing MSVC?). NumPy fallback OK."
        print(msg, flush=True)
        return msg, 0
    return "OK compile-native", 0


def validate_native_soft() -> tuple[str, int]:
    from bnn.kernels.packed import native_kernel_available

    if not native_kernel_available():
        msg = "SKIP validate-native (native DLL not loaded; NumPy path covered by pytest)"
        print(msg, flush=True)
        return msg, 0
    code = run([sys.executable, str(ROOT / "scripts" / "validate_native.py")])
    if code != 0:
        return "FAIL validate-native", code
    return "OK validate-native", 0


def compare_goldens() -> tuple[str, int]:
    """Assert committed result JSONs satisfy golden floors (same as pytest gates)."""
    floors_path = ROOT / "tests" / "golden_floors.json"
    floors = json.loads(floors_path.read_text(encoding="utf-8"))
    failures: list[str] = []

    # Compression / native err from benchmark
    bench = json.loads((ROOT / "results" / "benchmark.json").read_text(encoding="utf-8"))
    for r in bench.get("results") or []:
        if r.get("max_abs_error_vs_fp32", 1) > floors["native_err_max"]:
            failures.append(f"benchmark err {r.get('max_abs_error_vs_fp32')} > 0")
        comp = r["theoretical"]["weight_compression"]
        if abs(comp - floors["compression_exact_when_uint64_pack"]) > 1e-9:
            failures.append(f"compression {comp} != 32")

    # MNIST
    mnist_rows = json.loads((ROOT / "results" / "train_results.json").read_text(encoding="utf-8"))
    by = {r["model"]: r["test_acc"] for r in mnist_rows}
    g = floors["mnist"]
    if by.get("binary_mlp", 0) < g["binary_mlp_min_acc"]:
        failures.append(f"mnist binary_mlp {by.get('binary_mlp')} < {g['binary_mlp_min_acc']}")
    if (
        by.get("fp32_mlp", 0) >= g["fp_for_gap_gate"]
        and (by["fp32_mlp"] - by["binary_mlp"]) > g["gap_max_pp_fp_vs_binary"]
    ):
        failures.append("mnist gap too large")

    # Image
    img = json.loads((ROOT / "results" / "image_cifar.json").read_text(encoding="utf-8"))
    iby = {r["model"]: r["test_acc"] for r in img.get("results") or []}
    ig = floors["image_cifar"]
    if iby.get("binary_cifar_bireal", 0) < ig["binary_bireal_min_acc"]:
        failures.append("image binary below floor")
    gap = img.get("acc_gap_pp_fp_vs_binary_cnn", 99)
    if gap > ig["gap_max_pp_fp_vs_binary"]:
        failures.append(f"image gap {gap} > {ig['gap_max_pp_fp_vs_binary']}")

    # Audio
    aud = json.loads((ROOT / "results" / "audio_synth.json").read_text(encoding="utf-8"))
    aby = {r["model"]: r["test_acc"] for r in aud.get("results") or []}
    ag = floors["audio_synth"]
    if aby.get("binary_cnn", 0) < ag["binary_cnn_min_acc"]:
        failures.append("audio binary below floor")

    # Wrap
    wrap = json.loads((ROOT / "results" / "wrap_demo.json").read_text(encoding="utf-8"))
    wg = floors["wrap_demo"]
    if abs(wrap["weight_compression_replaced_layers"] - wg["weight_compression_exact"]) > 1e-9:
        failures.append("wrap compression != 32")
    if wrap.get("qat"):
        if float(wrap.get("output_cosine_vs_fp") or 0) < wg["cosine_min"]:
            failures.append("wrap_demo QAT cosine below 0.85")
        if float(wrap.get("e2e_speedup") or 0) < wg["e2e_speedup_min"]:
            failures.append("wrap_demo e2e below 1.5×")
        if wrap.get("forced"):
            failures.append("wrap_demo AND-gate used --force")
        if wrap.get("drop_in_ok") is not True:
            failures.append("wrap_demo drop_in_ok is not true")

    ultra_path = ROOT / "results" / "ultra_wrap.json"
    ug = floors.get("ultra_wrap")
    if ug and ultra_path.exists():
        ultra = json.loads(ultra_path.read_text(encoding="utf-8"))
        ba = ultra.get("before_after") or {}
        if abs(ba.get("binary_compression", 0) - ug["binary_compression_exact"]) > 1e-9:
            failures.append("ultra_wrap binary compression != 32")
        if ba.get("ternary_hybrid_calib_cosine", 0) < ug["ternary_cosine_min"]:
            failures.append("ultra_wrap ternary cosine below floor")
        if ba.get("binary_hybrid_calib_cosine", 0) < ug["binary_hybrid_cosine_min"]:
            failures.append("ultra_wrap binary hybrid cosine below floor")
        wide = ba.get("binary_gemm_only_speedup_wide")
        if wide is not None and wide < ug["gemm_only_speedup_wide_min"]:
            failures.append(f"ultra_wrap wide gemm speedup {wide} < {ug['gemm_only_speedup_wide_min']}")

    if failures:
        print("GOLDEN COMPARE FAIL:", *failures, sep="\n  ", flush=True)
        return "FAIL golden-compare", 1
    print("GOLDEN COMPARE: PASS (committed results within published floors)", flush=True)
    return "OK golden-compare", 0


def smoke_trains(overwrite: bool) -> list[tuple[str, int]]:
    """Short deterministic smokes; default writes to _repro_smoke_*.json."""
    from bnn.determinism import set_repro_seed

    status = set_repro_seed(0, deterministic=True, force_cpu=True)
    print("repro seed status:", json.dumps(status, indent=2), flush=True)

    img_out = ROOT / "results" / ("image_cifar.json" if overwrite else "_repro_smoke_image.json")
    aud_out = ROOT / "results" / ("audio_synth.json" if overwrite else "_repro_smoke_audio.json")
    steps = [
        (
            "smoke-audio",
            [
                sys.executable,
                str(ROOT / "scripts" / "train_audio.py"),
                "--epochs",
                "2",
                "--n-train",
                "128",
                "--n-test",
                "32",
                "--n-classes",
                "4",
                "--channels",
                "16",
                "--seed",
                "0",
                "--out",
                str(aud_out),
            ],
        ),
        (
            "smoke-image",
            [
                sys.executable,
                str(ROOT / "scripts" / "train_image.py"),
                "--epochs",
                "1",
                "--train-subset",
                "512",
                "--batch-size",
                "64",
                "--channels",
                "32",
                "--seed",
                "0",
                "--out",
                str(img_out),
            ],
        ),
    ]
    out: list[tuple[str, int]] = []
    for name, cmd in steps:
        code = run(cmd)
        out.append((name, code))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Reproduce / verify published BNN goldens")
    p.add_argument(
        "--mode",
        choices=("verify", "full"),
        default="verify",
        help="verify=fast gates; full=+short smoke trains",
    )
    p.add_argument(
        "--overwrite-goldens",
        action="store_true",
        help="With --mode full, write smoke outputs over published results (dangerous)",
    )
    p.add_argument("--skip-compile", action="store_true")
    p.add_argument("--skip-pytest", action="store_true")
    args = p.parse_args(argv)

    report: list[tuple[str, int]] = []

    if not args.skip_compile:
        report.append(compile_native_best_effort())

    if not args.skip_pytest:
        code = run([sys.executable, "-m", "pytest", "-q"])
        report.append(("pytest", code))

    code = run([sys.executable, str(ROOT / "scripts" / "export_check.py")])
    report.append(("export-check", code))

    report.append(validate_native_soft())
    report.append(compare_goldens())

    if args.mode == "full":
        for name, code in smoke_trains(overwrite=args.overwrite_goldens):
            report.append((name, code))

    out = write_summary(ROOT / "results" / "SUMMARY.md")
    print(f"Wrote {out}", flush=True)

    print("\n=== REPRO REPORT ===", flush=True)
    failed = False
    for name, code in report:
        status = "PASS" if code == 0 else "FAIL"
        if code != 0:
            failed = True
        print(f"  [{status}] {name}", flush=True)

    floors = json.loads((ROOT / "tests" / "golden_floors.json").read_text(encoding="utf-8"))
    print("\nPublished floors (schema", floors.get("schema_version"), "):", flush=True)
    print(
        json.dumps(
            {
                "native_err_max": floors["native_err_max"],
                "compression_target": floors["compression_target"],
                "mnist_binary_min": floors["mnist"]["binary_mlp_min_acc"],
                "image_binary_min": floors["image_cifar"]["binary_bireal_min_acc"],
                "audio_binary_min": floors["audio_synth"]["binary_cnn_min_acc"],
            },
            indent=2,
        ),
        flush=True,
    )

    if failed:
        print("\nREPRO: FAIL", flush=True)
        return 1
    print("\nREPRO: PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
