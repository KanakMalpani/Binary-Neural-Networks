"""SUMMARY.md aggregation: tolerant of missing/partial results, no global leaks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import bnn.eval_report as er
from bnn.eval_report import machine_card, render_summary, write_summary


def _write(d: Path, name: str, payload) -> None:
    (d / name).write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def results(tmp_path: Path) -> Path:
    out = tmp_path / "results"
    out.mkdir()
    return out


def test_render_does_not_mutate_module_global(results: Path):
    """A scoped call must not redirect later default-directory calls.

    render_summary() used to rebind the module-level RESULTS, so any later
    no-arg call read the caller's (often deleted) directory instead.
    """
    before = er.RESULTS
    render_summary(results)
    assert before == er.RESULTS
    # And the default path still resolves to the real repo results dir.
    assert er.RESULTS.name == "results"


def test_render_empty_dir_reports_missing_sections(results: Path):
    text = render_summary(results)
    assert "No benchmark.json rows found" in text
    assert "No train_results.json" in text
    assert "No image results" in text
    assert "No audio_synth.json" in text


def test_render_benchmark_rows_and_compression(results: Path):
    _write(
        results,
        "benchmark.json",
        {
            "results": [
                {
                    "shape": {"batch": 64, "in_features": 4096, "out_features": 4096},
                    "speedup_compute_vs_numpy_fp32": 23.86,
                    "speedup_compute_vs_torch_fp32": 12.5,
                    "max_abs_error_vs_fp32": 0.0,
                    "theoretical": {"weight_compression": 32.0},
                }
            ]
        },
    )
    text = render_summary(results)
    assert "64×4096×4096" in text
    assert "23.86" in text
    assert "12.50" in text
    assert "32.0" in text


def test_render_handles_missing_speedups_without_crashing(results: Path):
    """Partial rows are common mid-run; they must render as em dashes."""
    _write(results, "benchmark.json", {"results": [{"shape": "8x8x8"}]})
    text = render_summary(results)
    assert "8x8x8" in text
    assert "—" in text


def test_render_accepts_all_three_bench_row_keys(results: Path):
    for key in ("rows", "results", "benchmarks"):
        _write(results, "benchmark.json", {key: [{"shape": "1x2x3", "err": 0}]})
        assert "1x2x3" in render_summary(results)


def test_render_mnist_from_list_and_dict(results: Path):
    _write(results, "train_results.json", [{"model": "binary_mlp", "test_acc": 96.36}])
    assert "96.36" in render_summary(results)

    _write(results, "train_results.json", {"results": [{"name": "ternary", "acc": 97.16}]})
    assert "97.16" in render_summary(results)


def test_render_image_and_audio_gap(results: Path):
    _write(
        results,
        "image_cifar.json",
        {"results": [
            {"model": "fp32_cifar_cnn", "test_acc": 71.14},
            {"model": "binary_cifar_bireal", "test_acc": 61.14},
        ]},
    )
    _write(
        results,
        "audio_synth.json",
        {"results": [
            {"model": "fp32_cnn", "test_acc": 94.5},
            {"model": "binary_cnn", "test_acc": 96.0},
        ], "acc_gap_pp": -1.5},
    )
    text = render_summary(results)
    assert "71.14" in text and "61.14" in text
    assert "10.00 pp" in text          # gap derived when not supplied
    assert "Not production ASR" in text  # audio honesty note is mandatory


def test_render_image_falls_back_to_cifar_proxy(results: Path):
    _write(
        results,
        "cifar10_proxy.json",
        {"results": [
            {"model": "fp32_cnn", "test_acc": 60.0},
            {"model": "binary_bireal", "test_acc": 55.0},
        ]},
    )
    text = render_summary(results)
    assert "cifar10_proxy.json" in text


def test_render_wrap_section_keeps_dual_metric_honesty(results: Path):
    _write(
        results,
        "wrap_demo.json",
        {
            "e2e_latency_ms_fp": 21.5,
            "e2e_latency_ms_wrapped": 18.6,
            "e2e_speedup": 1.15,
            "weight_compression_replaced_layers": 32.0,
            "output_cosine_vs_fp": 0.28,
            "layer_microbench": {"speedup_gemm_only_vs_torch_linear": 9.39},
        },
    )
    text = render_summary(results)
    assert "32.0" in text
    assert "9.39" in text
    # Compression must be labelled theoretical, never as an e2e win.
    assert "exact bit-pack" in text
    assert "low without QAT is expected" in text


def test_write_summary_creates_file_and_parent(tmp_path: Path, results: Path):
    out = tmp_path / "nested" / "SUMMARY.md"
    written = write_summary(out, results)
    assert written == out
    assert out.is_file()
    assert out.read_text(encoding="utf-8").startswith("# Results summary")


def test_write_summary_defaults_into_results_dir(results: Path):
    written = write_summary(None, results)
    assert written == results / "SUMMARY.md"
    assert written.is_file()


def test_machine_card_has_reproducibility_fields():
    card = machine_card()
    for key in ("platform", "python", "torch", "cuda", "generated_utc"):
        assert key in card
    assert isinstance(card["cuda"], bool)
    assert card["generated_utc"].endswith("+00:00")  # UTC, not naive local time
