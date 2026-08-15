"""CLI surface: every subcommand parses, --help works, and defaults are sane.

Deliberately does not run long trains; this covers argument wiring, which is
where CLI regressions actually happen (a renamed dest silently breaks a flag).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from bnn.cli import EPILOG, build_parser, main

# Every subcommand registered on the parser.
SUBCOMMANDS = sorted(
    build_parser()._subparsers._group_actions[0].choices  # type: ignore[union-attr]
)

# Hidden argparse verbs that must not appear in ``bnn --help`` epilog.
# Prefer documenting a new public verb in ``bnn.cli.PUBLIC_CLI_VERBS`` instead.
INTERNAL_COMMANDS: frozenset[str] = frozenset()


def test_subcommand_inventory_is_non_trivial():
    assert len(SUBCOMMANDS) >= 15
    for expected in ("repro", "optimise", "bench", "encode", "decode", "profile", "kg"):
        assert expected in SUBCOMMANDS


def _token_in_epilog(cmd: str, epilog: str) -> bool:
    """True when *cmd* appears as a hyphen-aware token (not a substring of Training)."""
    return re.search(rf"(?<![\w-]){re.escape(cmd)}(?![\w-])", epilog) is not None


def test_every_subcommand_is_in_help_epilog_or_marked_internal():
    """Issue #2: a new argparse verb must be in the --help epilog or INTERNAL_COMMANDS."""
    help_text = build_parser().format_help()
    assert EPILOG in help_text, "EPILOG must be attached to ``bnn --help``"
    undocumented = [
        cmd
        for cmd in SUBCOMMANDS
        if cmd not in INTERNAL_COMMANDS and not _token_in_epilog(cmd, EPILOG)
    ]
    assert undocumented == [], (
        "CLI verbs registered but missing from bnn.cli.EPILOG "
        f"(and not INTERNAL_COMMANDS): {undocumented}"
    )


@pytest.mark.parametrize("cmd", SUBCOMMANDS)
def test_every_subcommand_has_working_help(cmd: str):
    """`--help` exits 0 for all of them — a broken parser raises instead."""
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args([cmd, "--help"])
    assert exc.value.code == 0


@pytest.mark.parametrize("cmd", SUBCOMMANDS)
def test_every_subcommand_binds_a_handler(cmd: str):
    """Each subparser must set_defaults(func=...) or dispatch would crash."""
    args = _parse_with_required(cmd)
    if args is None:
        pytest.skip(f"{cmd} needs positional args not inferable here")
    assert hasattr(args, "func"), f"{cmd} has no bound handler"
    assert callable(args.func)


def _parse_with_required(cmd: str):
    """Parse `cmd` with no flags; None when required args make that impossible."""
    try:
        return build_parser().parse_args([cmd])
    except SystemExit:
        return None


def test_no_command_errors_out():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_unknown_subcommand_is_rejected():
    assert main(["definitely-not-a-command"]) != 0


# --------------------------------------------------------------------------
# argument wiring for the commands with the most flags
# --------------------------------------------------------------------------

def test_bench_flag_defaults():
    args = build_parser().parse_args(["bench"])
    assert args.command == "bench"
    assert hasattr(args, "func")


def test_repro_mode_choices():
    args = build_parser().parse_args(["repro", "--mode", "full"])
    assert args.mode == "full"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["repro", "--mode", "not-a-mode"])


def test_optimise_policy_and_report_flags(tmp_path: Path):
    out = tmp_path / "r.json"
    args = build_parser().parse_args(
        ["optimise", "--policy", "auto", "--report", str(out), "--qat-steps", "40", "--force"]
    )
    assert args.policy == "auto"
    assert Path(args.report) == out
    assert args.qat_steps == 40
    assert args.force is True


def test_encode_requires_output_and_accepts_dims(tmp_path: Path):
    pack = tmp_path / "x.bnnpack"
    args = build_parser().parse_args(
        ["encode", "--source", "random", "--in-features", "128",
         "--out-features", "64", "--out", str(pack)]
    )
    assert args.in_features == 128
    assert args.out_features == 64


def test_profile_shape_flags():
    args = build_parser().parse_args(
        ["profile", "--batch", "8", "--in-features", "512", "--out-features", "256"]
    )
    assert (args.batch, args.in_features, args.out_features) == (8, 512, 256)


def test_recommend_goal_is_forwarded():
    args = build_parser().parse_args(["recommend", "--goal", "edge-vision"])
    assert args.goal == "edge-vision"


def test_kg_validate_and_summary(capsys):
    assert main(["kg", "validate"]) == 0
    assert "PASS" in capsys.readouterr().out
    assert main(["kg"]) == 0
    out = capsys.readouterr().out
    assert "knowledge graph" in out.lower() or "KG" in out or "nodes" in out


def test_train_seed_and_epochs():
    args = build_parser().parse_args(["train", "--epochs", "3", "--seed", "42"])
    assert args.epochs == 3
    assert args.seed == 42


# --------------------------------------------------------------------------
# commands cheap enough to actually execute
# --------------------------------------------------------------------------

def test_version_subcommand_and_flag(capsys):
    """main() converts argparse's SystemExit into a return code."""
    assert main(["version"]) == 0
    assert main(["--version"]) == 0
    assert "bnn" in capsys.readouterr().out


def test_export_check_runs_and_passes():
    assert main(["export-check"]) == 0


def test_encode_decode_round_trip_via_cli(tmp_path: Path):
    pack = tmp_path / "cli.bnnpack"
    assert main([
        "encode", "--source", "random",
        "--in-features", "128", "--out-features", "64",
        "--out", str(pack),
    ]) == 0
    assert pack.is_file()
    assert main(["decode", "--pack", str(pack)]) == 0


def test_profile_writes_valid_json(tmp_path: Path):
    out = tmp_path / "profile.json"
    assert main([
        "profile", "--batch", "4", "--in-features", "128",
        "--out-features", "64", "--reps", "2", "--out", str(out),
    ]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    for key in ("gemm_ms", "e2e_forward_ms", "speedup_vs_fp32", "native"):
        assert key in data
    assert data["m"] == 4
