"""CLI smoke: help, version, parser wiring (no long trains)."""

from __future__ import annotations

import subprocess
import sys

import pytest

from bnn import __version__
from bnn.cli import build_parser, main


def test_version_constant():
    assert __version__
    assert __version__[0].isdigit()


def test_cli_version_flag():
    assert main(["--version"]) == 0
    assert main(["version"]) == 0


def test_cli_help_lists_repro():
    p = build_parser()
    help_txt = p.format_help()
    assert "repro" in help_txt
    assert "optimise" in help_txt
    assert "packed" in help_txt.lower() or "CPU" in help_txt or "edge" in help_txt


def test_cli_optimise_help():
    code = main(["optimise", "--help"])
    assert code == 0
    p = build_parser()
    help_txt = p.format_help()
    assert "optimise" in help_txt


def test_cli_unknown_command_exits_nonzero():
    code = main(["not-a-real-command"])
    assert code != 0


def test_subprocess_bnn_version():
    r = subprocess.run(
        [sys.executable, "-m", "bnn.cli", "version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0
    assert __version__ in r.stdout


@pytest.mark.parametrize("cmd", ["export-check"])
def test_cli_export_check_via_main(cmd):
    assert main([cmd]) == 0
