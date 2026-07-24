"""Tolerance gates vs committed results/*.json and tests/golden_floors.json.

These tests do **not** retrain. They assert that published goldens still support
the same conclusions (accuracy floors, exact compression, native err=0).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FLOORS = json.loads((ROOT / "tests" / "golden_floors.json").read_text(encoding="utf-8"))


def _load_results(name: str):
    path = ROOT / "results" / name
    if not path.exists():
        pytest.skip(f"missing {path.relative_to(ROOT)}")
    return json.loads(path.read_text(encoding="utf-8"))


def _by_model(rows: list[dict]) -> dict[str, dict]:
    return {r["model"]: r for r in rows if "model" in r}


def test_compression_and_native_err_from_benchmark():
    floors = FLOORS
    data = _load_results("benchmark.json")
    rows = data.get("results") or []
    assert rows, "benchmark.json has no results"
    for r in rows:
        err = r.get("max_abs_error_vs_fp32")
        assert err is not None
        assert err <= floors["native_err_max"]
        comp = r["theoretical"]["weight_compression"]
        assert comp >= floors["compression_min"]
        assert abs(comp - floors["compression_exact_when_uint64_pack"]) < 1e-9


def test_mnist_golden_gates():
    g = FLOORS["mnist"]
    rows = _load_results("train_results.json")
    by = _by_model(rows)
    assert "binary_mlp" in by and "fp32_mlp" in by

    fp = by["fp32_mlp"]["test_acc"]
    bn = by["binary_mlp"]["test_acc"]
    assert fp >= g["fp32_mlp_min_acc"]
    assert bn >= g["binary_mlp_min_acc"]
    if fp >= g["fp_for_gap_gate"]:
        assert (fp - bn) <= g["gap_max_pp_fp_vs_binary"]

    # Same conclusion as recorded golden (within tolerance)
    assert abs(bn - g["recorded"]["binary_mlp"]) <= g["acc_tolerance_pp"]
    assert abs(fp - g["recorded"]["fp32_mlp"]) <= g["acc_tolerance_pp"]

    if "ternary_mlp" in by:
        te = by["ternary_mlp"]["test_acc"]
        assert te >= g["ternary_mlp_min_acc"]
        assert abs(te - g["recorded"]["ternary_mlp"]) <= g["acc_tolerance_pp"]


def test_image_cifar_golden_gates():
    g = FLOORS["image_cifar"]
    data = _load_results("image_cifar.json")
    by = _by_model(data.get("results") or [])
    assert "fp32_cifar_cnn" in by and "binary_cifar_bireal" in by

    fp = by["fp32_cifar_cnn"]["test_acc"]
    bn = by["binary_cifar_bireal"]["test_acc"]
    assert fp >= g["fp32_cnn_min_acc"]
    assert bn >= g["binary_bireal_min_acc"]
    gap = data.get("acc_gap_pp_fp_vs_binary_cnn", fp - bn)
    assert gap <= g["gap_max_pp_fp_vs_binary"]

    assert abs(fp - g["recorded"]["fp32_cifar_cnn"]) <= g["acc_tolerance_pp"]
    assert abs(bn - g["recorded"]["binary_cifar_bireal"]) <= g["acc_tolerance_pp"]


def test_audio_synth_golden_gates():
    g = FLOORS["audio_synth"]
    data = _load_results("audio_synth.json")
    by = _by_model(data.get("results") or [])
    assert "fp32_cnn" in by and "binary_cnn" in by

    fp = by["fp32_cnn"]["test_acc"]
    bn = by["binary_cnn"]["test_acc"]
    assert fp >= g["fp32_cnn_min_acc"]
    assert bn >= g["binary_cnn_min_acc"]
    gap = abs(data.get("acc_gap_pp", fp - bn))
    assert gap <= g["gap_max_pp_abs"]

    assert abs(fp - g["recorded"]["fp32_cnn"]) <= g["acc_tolerance_pp"]
    assert abs(bn - g["recorded"]["binary_cnn"]) <= g["acc_tolerance_pp"]


def test_wrap_demo_compression_gate():
    g = FLOORS["wrap_demo"]
    data = _load_results("wrap_demo.json")
    comp = data["weight_compression_replaced_layers"]
    assert abs(comp - g["weight_compression_exact"]) < 1e-9
    # Without QAT, cosine is low; gate only bounds "not accidentally perfect"
    cos = data.get("output_cosine_vs_fp")
    if cos is not None:
        assert cos <= g["cosine_max_without_qat"]


def test_ultra_wrap_floors():
    g = FLOORS.get("ultra_wrap")
    if g is None:
        return
    path = ROOT / "results" / "ultra_wrap.json"
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    ba = data["before_after"]
    assert abs(ba["binary_compression"] - g["binary_compression_exact"]) < 1e-9
    assert abs(ba["ternary_compression"] - g["ternary_compression_exact"]) < 1e-9
    assert ba["ternary_hybrid_calib_cosine"] >= g["ternary_cosine_min"]
    assert ba["binary_hybrid_calib_cosine"] >= g["binary_hybrid_cosine_min"]
    wide = ba.get("binary_gemm_only_speedup_wide")
    if wide is not None:
        # Advisory efficiency floor — soft on CI noise (still assert minimum sanity)
        assert wide >= g["gemm_only_speedup_wide_min"], (
            f"wide gemm speedup {wide} < {g['gemm_only_speedup_wide_min']}"
        )


def test_export_check_compression_floor_matches_floors():
    """Live micro-check: packing still ~32× (independent of committed JSON)."""
    import numpy as np
    import torch

    from bnn.kernels.packed import pack_binary_pm1
    from bnn.ste import binary_sign

    torch.manual_seed(0)
    w = binary_sign(torch.randn(512, 1024)).cpu().numpy()
    packed, _ = pack_binary_pm1(w, axis=1)
    ratio = (w.size * 4) / packed.nbytes
    assert ratio >= FLOORS["compression_min"]
    assert abs(ratio - FLOORS["compression_exact_when_uint64_pack"]) < 1e-9
