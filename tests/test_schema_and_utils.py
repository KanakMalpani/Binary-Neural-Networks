"""Optimise-report schema, logging helpers, determinism, and build commands."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from bnn import logutil
from bnn.determinism import set_repro_seed
from bnn.kernels.compile_native import unix_compile_commands
from bnn.wrap.schema import (
    SCHEMA_ID,
    SCHEMA_VERSION,
    envelope,
    is_valid_optimise_report,
    validate_optimise_report,
)

# --------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------

def _minimal() -> dict:
    return envelope(
        policy="hybrid_ffn",
        mode="binary_xnor",
        replaced=["ffn_fc1"],
        skipped=["attn_qkv"],
        compression_replaced_weights=32.0,
        fp32_weight_bytes_replaced=4096,
        packed_weight_bytes=128,
        native_kernel=True,
        drop_in_ok=False,
        forced=True,
        status="ok",
    )


def test_envelope_is_valid_and_self_describing():
    report = _minimal()
    assert report["schema"] == SCHEMA_ID
    assert report["schema_version"] == SCHEMA_VERSION
    assert validate_optimise_report(report) == []
    assert is_valid_optimise_report(report)


def test_envelope_always_carries_dual_metric_note():
    """Compression is theoretical; the note is what stops it reading as speedup."""
    note = _minimal()["thesis_note"].lower()
    assert "theoretical" in note
    assert "wall-clock" in note


def test_envelope_passes_through_extras():
    report = envelope(**{**_dict_args(), "e2e_speedup": 1.2, "custom": "x"})
    assert report["e2e_speedup"] == 1.2
    assert report["custom"] == "x"


def _dict_args() -> dict:
    return {
        "policy": "p", "mode": "m", "replaced": [], "skipped": [],
        "compression_replaced_weights": 1.0, "fp32_weight_bytes_replaced": 0,
        "packed_weight_bytes": 0, "native_kernel": False, "drop_in_ok": None,
        "forced": False, "status": "ok",
    }


def test_non_dict_payload_rejected():
    assert validate_optimise_report(["not", "a", "dict"]) == ["report must be a dict"]


def test_unknown_schema_rejected():
    errs = validate_optimise_report({"schema": "something_else"})
    assert any("schema must be" in e for e in errs)


def test_legacy_schema_accepted_during_transition():
    assert validate_optimise_report({"schema": "ultra_wrap_report_v1"}) == []


def test_wrong_version_rejected():
    report = _minimal()
    report["schema_version"] = 999
    errs = validate_optimise_report(report)
    assert any("schema_version must be" in e for e in errs)


def test_missing_required_keys_are_each_reported():
    report = _minimal()
    del report["policy"]
    del report["status"]
    errs = validate_optimise_report(report)
    assert any("missing required key: policy" in e for e in errs)
    assert any("missing required key: status" in e for e in errs)


def test_strict_mode_requires_recommended_keys():
    report = _minimal()
    assert validate_optimise_report(report) == []
    errs = validate_optimise_report(report, strict=True)
    assert any("effectiveness" in e for e in errs)
    assert any("policy_reason" in e for e in errs)


def test_honesty_rule_flags_big_speedup_claimed_with_big_compression():
    """A >20x e2e claim alongside >20x compression must carry a dual-metric note."""
    report = _minimal()
    report["e2e_speedup"] = 25.0
    report["thesis_note"] = "very fast"      # no dual-metric wording
    errs = validate_optimise_report(report)
    assert any("strict honesty" in e for e in errs)


def test_honesty_rule_satisfied_by_dual_metric_note():
    report = _minimal()
    report["e2e_speedup"] = 25.0
    report["thesis_note"] = "dual-metric: compression theoretical, latency wall-clock"
    assert validate_optimise_report(report) == []


def test_honesty_rule_ignores_unparseable_speedup():
    report = _minimal()
    report["e2e_speedup"] = "fast"
    assert validate_optimise_report(report) == []


def test_honesty_rule_allows_modest_speedup():
    report = _minimal()
    report["e2e_speedup"] = 2.0
    assert validate_optimise_report(report) == []


# --------------------------------------------------------------------------
# logutil
# --------------------------------------------------------------------------

def test_info_goes_to_stdout_with_fields(capsys):
    logutil.info("packed", layers=3, ok=True)
    cap = capsys.readouterr()
    assert "INFO packed" in cap.out
    assert "layers=3" in cap.out
    assert cap.err == ""


@pytest.mark.parametrize("fn,label", [(logutil.warn, "WARN"), (logutil.error, "ERROR")])
def test_warn_and_error_go_to_stderr(capsys, fn, label):
    """Diagnostics must not pollute stdout — JSON reports are parsed from it."""
    fn("something")
    cap = capsys.readouterr()
    assert f"{label} something" in cap.err
    assert cap.out == ""


def test_log_repr_quotes_strings(capsys):
    logutil.info("m", path="a b")
    assert "path='a b'" in capsys.readouterr().out


def test_log_without_fields_has_no_trailing_space(capsys):
    logutil.info("bare")
    assert capsys.readouterr().out == "INFO bare\n"


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------

def test_set_repro_seed_makes_torch_and_numpy_reproducible():
    set_repro_seed(1234)
    a_t, a_n = torch.randn(5), np.random.rand(5)  # noqa: NPY002 — legacy global RNG is what set_repro_seed seeds
    set_repro_seed(1234)
    b_t, b_n = torch.randn(5), np.random.rand(5)  # noqa: NPY002 — legacy global RNG is what set_repro_seed seeds
    assert torch.equal(a_t, b_t)
    assert np.array_equal(a_n, b_n)


def test_different_seeds_differ():
    set_repro_seed(1)
    a = torch.randn(5)
    set_repro_seed(2)
    assert not torch.equal(a, torch.randn(5))


def test_set_repro_seed_reports_status_for_result_json():
    status = set_repro_seed(7)
    assert status["seed"] == 7
    assert status["deterministic_requested"] is True
    assert isinstance(status["cuda_available"], bool)
    assert isinstance(status["notes"], list)


def test_set_repro_seed_non_deterministic_mode():
    status = set_repro_seed(3, deterministic=False)
    assert status["deterministic_requested"] is False


# --------------------------------------------------------------------------
# native build command ladder
# --------------------------------------------------------------------------

def test_compile_commands_never_bake_in_host_isa():
    """-march=native would make a wheel SIGILL on older CPUs of the same arch."""
    for cmd in unix_compile_commands("gcc", Path("out.so"), Path("in.c"), openmp=True):
        joined = " ".join(cmd)
        assert "-march=native" not in joined
        assert "-mtune=native" not in joined
        assert "-O3" in joined
        assert "-shared" in joined and "-fPIC" in joined


def test_compile_commands_end_with_single_threaded_fallback():
    """Correctness must never depend on OpenMP being available."""
    cmds = unix_compile_commands("cc", Path("out.so"), Path("in.c"), openmp=True)
    assert len(cmds) >= 2
    assert not any("openmp" in a for a in cmds[-1]), "last resort still requires OpenMP"


def test_compile_commands_openmp_disabled_yields_only_plain_build():
    cmds = unix_compile_commands("gcc", Path("o.so"), Path("i.c"), openmp=False)
    assert len(cmds) == 1
    assert not any("openmp" in a for a in cmds[0])


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS libomp routing")
def test_macos_uses_xpreprocessor_for_libomp():
    cmds = unix_compile_commands("clang", Path("o.so"), Path("i.c"), openmp=True)
    assert any("-Xpreprocessor" in c for c in cmds)
