"""CLI handler coverage: flag wiring + cheap in-process commands.

Expensive trains/wraps are mocked at ``_run_script``. Cheap commands
(recommend, encode/decode, profile, export-check, pareto --demo) run for real.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from bnn import cli
from bnn.cli import (
    _exit_code,
    _run_script,
    _ultra_wrap_extra,
    build_parser,
    cmd_decode,
    cmd_encode,
    cmd_wrap,
    main,
)

# --------------------------------------------------------------------------
# script runner
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, 0),
        (True, 0),
        (False, 1),
        (0, 0),
        (2, 2),
        ("fail", 1),
        ("", 0),
    ],
)
def test_exit_code(value, expected):
    assert _exit_code(value) == expected


def test_run_script_export_check():
    assert _run_script("export_check.py") == 0


@pytest.mark.parametrize("bad", ["../setup.py", "foo/bar.py", "not_py.txt", "missing.py"])
def test_run_script_rejects_bad_names(bad: str):
    with pytest.raises(FileNotFoundError):
        _run_script(bad)


def test_run_script_argv_aware_main(monkeypatch):
    seen: list[list[str]] = []

    def main_fn(argv=None):
        seen.append(list(argv or []))
        return 0

    monkeypatch.setattr(cli, "_load_script_module", lambda _p: SimpleNamespace(main=main_fn))
    assert _run_script("repro_all.py", ["--mode", "verify"]) == 0
    assert seen == [["--mode", "verify"]]


def test_run_script_rewrites_sys_argv_for_parse_args_mains(monkeypatch):
    """Dominant path: ``def main():`` + ``parse_args()`` with no argv arg."""
    before = list(sys.argv)
    captured: list[str] = []

    def main_fn():
        captured.extend(sys.argv)
        return

    monkeypatch.setattr(cli, "_load_script_module", lambda _p: SimpleNamespace(main=main_fn))
    assert _run_script("export_check.py", ["--flag", "1"]) == 0
    assert captured[0].endswith("export_check.py")
    assert captured[1:] == ["--flag", "1"]
    assert sys.argv == before


def test_run_script_restores_argv_after_failure(monkeypatch):
    before = list(sys.argv)

    def main_fn():
        raise RuntimeError("boom")

    monkeypatch.setattr(cli, "_load_script_module", lambda _p: SimpleNamespace(main=main_fn))
    assert _run_script("export_check.py", ["x"]) == 1
    assert sys.argv == before


def test_run_script_systemexit_and_exceptions(monkeypatch):
    def exit_fail():
        raise SystemExit("fail")

    def assert_fail():
        raise AssertionError("x")

    monkeypatch.setattr(cli, "_load_script_module", lambda _p: SimpleNamespace(main=exit_fail))
    assert _run_script("export_check.py") == 1
    monkeypatch.setattr(cli, "_load_script_module", lambda _p: SimpleNamespace(main=assert_fail))
    assert _run_script("export_check.py") == 1


def test_run_script_no_main_returns_1(monkeypatch):
    monkeypatch.setattr(cli, "_load_script_module", lambda _p: SimpleNamespace())
    assert _run_script("export_check.py") == 1


def test_failed_import_does_not_poison_cache(tmp_path: Path, monkeypatch):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    target = scripts / "broken_tmp.py"
    target.write_text("raise RuntimeError('import boom')\n", encoding="utf-8")
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    mod_name = "bnn._scripts.broken_tmp"
    try:
        assert _run_script("broken_tmp.py") == 1
        assert mod_name not in sys.modules
        target.write_text("def main():\n    return 0\n", encoding="utf-8")
        assert _run_script("broken_tmp.py") == 0
    finally:
        sys.modules.pop(mod_name, None)


def test_main_maps_file_and_runtime_errors():
    with patch.object(cli, "_run_script", side_effect=FileNotFoundError("gone")):
        assert main(["export-check"]) == 2
    with patch.object(cli, "_run_script", side_effect=RuntimeError("boom")):
        assert main(["export-check"]) == 1


# --------------------------------------------------------------------------
# handler argv wiring
# --------------------------------------------------------------------------


@pytest.fixture
def capture_script():
    calls: list[tuple[str, list[str]]] = []

    def fake(script: str, extra: list[str] | None = None) -> int:
        calls.append((script, list(extra or [])))
        return 0

    with patch.object(cli, "_run_script", side_effect=fake):
        yield calls


@pytest.mark.parametrize(
    ("argv", "script", "exact_extra"),
    [
        (["bench"], "benchmark.py", []),
        (
            ["bench", "--reps", "3", "--warmup", "1", "--threads", "1,2"],
            "benchmark.py",
            ["--reps", "3", "--threads", "1,2", "--warmup", "1"],
        ),
        (
            ["train", "--epochs", "2", "--seed", "7", "--model", "binary_mlp", "--threads", "4"],
            "train.py",
            ["--epochs", "2", "--seed", "7", "--models", "binary_mlp", "--threads", "4"],
        ),
        (["train"], "train.py", ["--epochs", "3", "--seed", "0"]),
        (
            ["repro", "--mode", "full", "--overwrite-goldens", "--skip-compile", "--skip-pytest"],
            "repro_all.py",
            ["--mode", "full", "--overwrite-goldens", "--skip-compile", "--skip-pytest"],
        ),
        (["validate-native"], "validate_native.py", []),
        (["energy-bound"], "energy_bound_measured.py", []),
        (
            ["train-cifar", "--epochs", "1", "--subset", "100", "--batch-size", "16"],
            "train_cifar10_proxy.py",
            ["--epochs", "1", "--train-subset", "100", "--batch-size", "16"],
        ),
            (
                ["train-image", "--epochs", "1", "--subset", "50", "--approx-sign", "--include-vit",
                 "--out", "results/_t.json"],
                "train_image.py",
                [
                    "--epochs", "1", "--train-subset", "50", "--batch-size", "128",
                    "--channels", "64", "--seed", "0", "--approx-sign", "--include-vit",
                    "--out", str(Path("results/_t.json")),
                ],
            ),
            (
                ["train-audio", "--epochs", "1", "--n-train", "32", "--approx-sign"],
                "train_audio.py",
                [
                    "--epochs", "1", "--batch-size", "64", "--n-train", "32", "--n-test", "200",
                    "--n-classes", "8", "--channels", "32", "--seed", "0", "--approx-sign",
                ],
            ),
            (
                ["eval-suite", "--full", "--skip-pytest", "--out", "results/S.md"],
                "run_eval_suite.py",
                ["--out", str(Path("results/S.md")), "--full", "--skip-pytest"],
            ),
            (
                ["train-seq2seq", "--task", "ae", "--ffn", "ternary", "--steps", "5", "--out", "o.json"],
                "train_seq2seq.py",
                [
                    "--task", "ae", "--ffn", "ternary", "--steps", "5", "--batch", "32",
                    "--seq-len", "8", "--dim", "64", "--seed", "0", "--out", str(Path("o.json")),
                ],
            ),
            (
                ["wrap-transformer", "--d-model", "64", "--qat-steps", "2", "--out", "w.json"],
                "tiny_transformer_wrap_demo.py",
                [
                    "--d-model", "64", "--ff", "512", "--depth", "2", "--batch", "32",
                    "--qat-steps", "2", "--policy", "hybrid_ffn", "--out", str(Path("w.json")),
                ],
            ),
    ],
)
def test_handler_exact_argv(capture_script, argv, script, exact_extra):
    assert main(argv) == 0
    got_script, extra = capture_script[0]
    assert got_script == script
    assert extra == exact_extra


def test_pareto_from_optimise_argv(capture_script, tmp_path: Path):
    src = tmp_path / "ultra_wrap.json"
    src.write_text("{}", encoding="utf-8")
    out = tmp_path / "x.json"
    assert main(["pareto", "--from-optimise", str(src), "--out", str(out)]) == 0
    script, extra = capture_script[0]
    assert script == "pareto_report.py"
    assert extra[:4] == ["--out", str(out), "--from-optimise", str(src)]
    assert "--demo" not in extra


def test_wrap_ultra_forwards_explicit_mode(capture_script):
    assert main(["wrap", "--ultra", "--policy", "auto", "--mode", "ternary_weight_only"]) == 0
    script, extra = capture_script[0]
    assert script == "ultra_wrap_demo.py"
    assert extra[extra.index("--mode") + 1] == "ternary_weight_only"
    assert extra[extra.index("--policy") + 1] == "auto"


def test_wrap_legacy_warns(capture_script):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert main(["wrap", "--mode", "binary_xnor", "--hidden", "128"]) == 0
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert capture_script[0] == (
        "wrap_existing_demo.py",
        ["--mode", "binary_xnor", "--hidden", "128", "--batch", "32"],
    )


def test_optimise_pack_then_encode(tmp_path: Path, capture_script):
    pack = tmp_path / "o.bnnpack"
    with patch.object(cli, "cmd_encode", return_value=0) as enc:
        assert main(["optimise", "--pack", str(pack), "--min-width", "32"]) == 0
        assert Path(enc.call_args[0][0].out) == pack
    assert capture_script[0][0] == "ultra_wrap_demo.py"


def test_optimise_propagates_failure():
    with patch.object(cli, "_run_script", return_value=7):
        assert main(["optimise"]) == 7


@pytest.mark.parametrize(
    ("ns_kwargs", "mode"),
    [
        ({"policy": "auto", "mode": "auto"}, "auto"),
        ({"policy": "auto", "mode": "ternary_weight_only"}, "ternary_weight_only"),
        ({"policy": "hybrid_ffn", "mode": "binary_xnor"}, "binary_xnor"),
    ],
)
def test_ultra_wrap_extra_forwards_mode(ns_kwargs, mode):
    ns = argparse.Namespace(
        batch=8,
        d_model=64,
        ff=128,
        calib_batches=1,
        min_width=16,
        qat_steps=0,
        drop_in_threshold=0.8,
        force=False,
        report=None,
        compare_baseline=False,
        **ns_kwargs,
    )
    extra = _ultra_wrap_extra(ns)
    assert extra[extra.index("--mode") + 1] == mode


def test_compile_native_force():
    with patch("bnn.kernels.compile_native.main", return_value=0) as m:
        assert main(["compile-native", "--force"]) == 0
        assert m.call_args[0][0] == ["--force"]


def test_cmd_wrap_direct():
    with patch.object(cli, "_run_script", return_value=0) as m:
        ns = build_parser().parse_args(["wrap", "--mode", "ternary_weight_only"])
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            assert cmd_wrap(ns) == 0
        assert m.call_args[0][0] == "wrap_existing_demo.py"


# --------------------------------------------------------------------------
# cheap real commands
# --------------------------------------------------------------------------


def test_recommend(capsys):
    assert main(["recommend", "--goal", "research-xnor"]) == 0
    out = capsys.readouterr().out
    assert "research-xnor" in out


def test_encode_decode_roundtrip(tmp_path: Path):
    pack = tmp_path / "mlp.bnnpack"
    assert main(["encode", "--source", "mlp", "--hidden", "64", "--out", str(pack)]) == 0
    assert main(["decode", "--pack", str(pack)]) == 0


def test_encode_unknown_source(tmp_path: Path):
    ns = argparse.Namespace(
        source="nope",
        out=tmp_path / "x.bnnpack",
        hidden=64,
        min_width=1,
        in_features=32,
        out_features=16,
    )
    assert cmd_encode(ns) == 2


def test_decode_nonzero_err(tmp_path: Path):
    pack = tmp_path / "r.bnnpack"
    assert main([
        "encode", "--source", "random",
        "--in-features", "64", "--out-features", "32", "--out", str(pack),
    ]) == 0
    with patch("bnn.codec.packed_module_fp_err", return_value=1.0):
        assert cmd_decode(argparse.Namespace(pack=pack)) == 1


def test_profile_json(capsys):
    assert main([
        "profile", "--batch", "2", "--in-features", "64",
        "--out-features", "32", "--reps", "1", "--warmup", "0",
    ]) == 0
    assert json.loads(capsys.readouterr().out)["m"] == 2


def test_pareto_demo(tmp_path: Path):
    out = tmp_path / "pareto.json"
    assert main(["pareto", "--demo", "--out", str(out)]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)


def test_energy_bound(tmp_path: Path):
    if not Path("results/wrap_demo.json").is_file():
        pytest.skip("wrap_demo.json missing")
    out = tmp_path / "energy.json"
    assert _run_script("energy_bound_measured.py", ["--out", str(out)]) == 0
    assert out.is_file()
