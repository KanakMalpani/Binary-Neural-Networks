"""Native build orchestration and the `python -m bnn` entry point.

Compiler invocation is monkeypatched throughout: these assert the *decision*
logic (which toolchain, which fallback, when to skip) without needing MSVC,
GCC or Homebrew present on the machine running the tests.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from bnn.kernels import compile_native as cn

# --------------------------------------------------------------------------
# python -m bnn
# --------------------------------------------------------------------------

def test_module_entrypoint_runs_cli():
    r = subprocess.run(
        [sys.executable, "-m", "bnn", "version"],
        capture_output=True, text=True, check=False,
    )
    assert r.returncode == 0
    assert "bnn" in r.stdout


def test_module_entrypoint_propagates_failure():
    r = subprocess.run(
        [sys.executable, "-m", "bnn", "not-a-command"],
        capture_output=True, text=True, check=False,
    )
    assert r.returncode != 0


# --------------------------------------------------------------------------
# toolchain discovery
# --------------------------------------------------------------------------

def test_find_vcvars_returns_none_when_nothing_installed(monkeypatch):
    monkeypatch.setattr(cn, "_find_vcvars_via_vswhere", lambda: None)
    monkeypatch.setattr(cn, "_VCVARS_CANDIDATES", [Path("X:/nope/vcvars64.bat")])
    assert cn._find_vcvars() is None


def test_find_vcvars_prefers_vswhere_result(monkeypatch, tmp_path: Path):
    found = tmp_path / "vcvars64.bat"
    found.write_text("", encoding="utf-8")
    monkeypatch.setattr(cn, "_find_vcvars_via_vswhere", lambda: found)
    assert cn._find_vcvars() == found


def test_find_vcvars_falls_back_to_candidate_list(monkeypatch, tmp_path: Path):
    cand = tmp_path / "vcvars64.bat"
    cand.write_text("", encoding="utf-8")
    monkeypatch.setattr(cn, "_find_vcvars_via_vswhere", lambda: None)
    monkeypatch.setattr(cn, "_VCVARS_CANDIDATES", [Path("X:/nope.bat"), cand])
    assert cn._find_vcvars() == cand


def test_vswhere_missing_returns_none(monkeypatch):
    monkeypatch.setattr(cn.Path, "exists", lambda self: False)
    assert cn._find_vcvars_via_vswhere() is None


def test_brew_libomp_none_without_brew(monkeypatch):
    monkeypatch.setattr(cn.shutil, "which", lambda name: None)
    assert cn._brew_libomp() is None


def test_brew_libomp_none_when_prefix_has_no_include(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(cn.shutil, "which", lambda name: "/usr/local/bin/brew")
    monkeypatch.setattr(
        cn.subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=str(tmp_path), stderr=""),
    )
    assert cn._brew_libomp() is None  # no include/ dir


def test_brew_libomp_found(monkeypatch, tmp_path: Path):
    (tmp_path / "include").mkdir()
    monkeypatch.setattr(cn.shutil, "which", lambda name: "/usr/local/bin/brew")
    monkeypatch.setattr(
        cn.subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=str(tmp_path), stderr=""),
    )
    assert cn._brew_libomp() == tmp_path


# --------------------------------------------------------------------------
# gcc/clang ladder
# --------------------------------------------------------------------------

def test_compile_gcc_without_any_compiler_explains_how_to_install(monkeypatch, capsys):
    monkeypatch.setattr(cn.shutil, "which", lambda name: None)
    rc = cn._compile_gcc(openmp=True)
    assert rc == 1
    err = capsys.readouterr().err
    # The message must be actionable on the common distros, and must say the
    # NumPy fallback still works — an install failure is not a dead end.
    assert "apt-get" in err and "brew" in err
    assert "NumPy fallback" in err


def test_compile_gcc_stops_at_first_success(monkeypatch, tmp_path: Path):
    calls: list[list[str]] = []
    out = tmp_path / "lib.so"

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        out.write_bytes(b"\x7fELF")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(cn.shutil, "which", lambda name: "gcc" if name == "gcc" else None)
    monkeypatch.setattr(cn, "DLL", out)
    monkeypatch.setattr(cn.subprocess, "run", fake_run)
    assert cn._compile_gcc(openmp=True) == 0
    assert len(calls) == 1, "should not keep trying after a successful build"


def test_compile_gcc_falls_back_when_openmp_build_fails(monkeypatch, tmp_path: Path):
    """A missing libgomp must degrade to single-threaded, not fail the install."""
    calls: list[list[str]] = []
    out = tmp_path / "lib.so"

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if any("openmp" in a for a in cmd):
            return subprocess.CompletedProcess(cmd, 1)  # no output produced
        out.write_bytes(b"\x7fELF")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(cn.shutil, "which", lambda name: "gcc" if name == "gcc" else None)
    monkeypatch.setattr(cn, "DLL", out)
    monkeypatch.setattr(cn.subprocess, "run", fake_run)
    assert cn._compile_gcc(openmp=True) == 0
    assert len(calls) >= 2
    assert not any("openmp" in a for a in calls[-1])


# --------------------------------------------------------------------------
# main()
# --------------------------------------------------------------------------

def test_main_errors_when_source_missing(monkeypatch, tmp_path: Path, capsys):
    monkeypatch.setattr(cn, "SRC", tmp_path / "absent.c")
    assert cn.main([]) == 1
    assert "missing source" in capsys.readouterr().err


def test_main_skips_rebuild_when_library_exists(monkeypatch, tmp_path: Path, capsys):
    lib = tmp_path / "lib.so"
    lib.write_bytes(b"x")
    monkeypatch.setattr(cn, "DLL", lib)
    assert cn.main([]) == 0
    assert "use --force" in capsys.readouterr().out


def test_main_force_triggers_a_rebuild(monkeypatch, tmp_path: Path):
    lib = tmp_path / "lib.so"
    lib.write_bytes(b"x")
    monkeypatch.setattr(cn, "DLL", lib)
    built: list[bool] = []

    def fake_compile(openmp: bool) -> int:
        built.append(openmp)
        lib.write_bytes(b"\x7fELF")
        return 0

    monkeypatch.setattr(cn, "_compile_msvc", fake_compile)
    monkeypatch.setattr(cn, "_compile_gcc", fake_compile)
    assert cn.main(["--force"]) == 0
    assert built == [cn.default_openmp()]


def test_main_no_openmp_flag_is_honoured(monkeypatch, tmp_path: Path):
    lib = tmp_path / "lib.so"
    monkeypatch.setattr(cn, "DLL", lib)
    seen: list[bool] = []

    def fake_compile(openmp: bool) -> int:
        seen.append(openmp)
        lib.write_bytes(b"\x7fELF")
        return 0

    monkeypatch.setattr(cn, "_compile_msvc", fake_compile)
    monkeypatch.setattr(cn, "_compile_gcc", fake_compile)
    assert cn.main(["--no-openmp"]) == 0
    assert seen == [False]


def test_main_openmp_flag_forces_on(monkeypatch, tmp_path: Path):
    lib = tmp_path / "lib.so"
    monkeypatch.setattr(cn, "DLL", lib)
    seen: list[bool] = []

    def fake_compile(openmp: bool) -> int:
        seen.append(openmp)
        lib.write_bytes(b"\x7fELF")
        return 0

    monkeypatch.setattr(cn, "_compile_msvc", fake_compile)
    monkeypatch.setattr(cn, "_compile_gcc", fake_compile)
    assert cn.main(["--openmp"]) == 0
    assert seen == [True]


def test_main_resets_cached_loader_state(monkeypatch, tmp_path: Path):
    """After a rebuild the loader must retry, including after an earlier failure."""
    import bnn.kernels.packed as packed

    lib = tmp_path / "lib.so"
    monkeypatch.setattr(cn, "DLL", lib)
    monkeypatch.setattr(cn, "_compile_msvc", lambda openmp: (lib.write_bytes(b"\x7fELF"), 0)[1])
    monkeypatch.setattr(cn, "_compile_gcc", lambda openmp: (lib.write_bytes(b"\x7fELF"), 0)[1])

    saved = (packed._NATIVE, packed._NATIVE_TRIED, packed._THREADS_APPLIED)
    try:
        packed._NATIVE = None
        packed._NATIVE_TRIED = True   # simulate a previous failed load
        assert cn.main(["--force"]) == 0
        assert packed._NATIVE_TRIED is False
        assert packed._THREADS_APPLIED is None
    finally:
        packed._NATIVE, packed._NATIVE_TRIED, packed._THREADS_APPLIED = saved


def test_main_unknown_args_are_tolerated(monkeypatch, tmp_path: Path):
    lib = tmp_path / "lib.so"
    lib.write_bytes(b"x")
    monkeypatch.setattr(cn, "DLL", lib)
    assert cn.main(["--totally-unknown"]) == 0


@pytest.mark.parametrize("openmp", [True, False])
def test_openmp_flag_choice_never_includes_native_arch(openmp: bool):
    cmds = cn.unix_compile_commands("gcc", Path("o.so"), Path("i.c"), openmp=openmp)
    assert cmds
    for cmd in cmds:
        assert "-march=native" not in " ".join(cmd)
