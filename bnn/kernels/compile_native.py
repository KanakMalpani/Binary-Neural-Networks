"""Compile the native binary/ternary GEMM shared library (Windows MSVC or Linux GCC)."""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "binary_gemm.c"
DEF = HERE / "binary_gemm.def"
DLL = HERE / ("_binary_gemm_native.dll" if os.name == "nt" else "_binary_gemm_native.so")

_VCVARS_CANDIDATES = [
    Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"),
    Path(r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"),
    Path(r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"),
    Path(r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"),
    Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"),
    Path(r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"),
]


def _find_vcvars_via_vswhere() -> Path | None:
    vswhere = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe")
    if not vswhere.exists():
        return None
    try:
        r = subprocess.run(
            [
                str(vswhere),
                "-latest",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property",
                "installationPath",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    root = (r.stdout or "").strip()
    if not root:
        return None
    cand = Path(root) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    return cand if cand.exists() else None


def _find_vcvars() -> Path | None:
    found = _find_vcvars_via_vswhere()
    if found is not None:
        return found
    for p in _VCVARS_CANDIDATES:
        if p.exists():
            return p
    return None


def _find_cl_on_path() -> str | None:
    return shutil.which("cl")


def _compile_msvc(openmp: bool) -> int:
    vcvars = _find_vcvars()
    cl = _find_cl_on_path()
    openmp_flag = "/openmp" if openmp else ""

    def_arg = f" /DEF:{DEF.name}" if DEF.exists() else ""
    if vcvars is not None:
        cmd = (
            f'"{vcvars}" && cl /nologo /O2 {openmp_flag} /LD '
            f'/Fe:{DLL.name} {SRC.name}{def_arg}'
        )
        while "  " in cmd:
            cmd = cmd.replace("  ", " ")
        print("Running:", cmd)
        r = subprocess.run(cmd, shell=True, cwd=str(HERE))
        return int(r.returncode)

    if cl is not None:
        # Already in a developer prompt. Distinct name from the shell string
        # built above -- this branch passes an argv list, not a command line.
        argv = ["cl", "/nologo", "/O2"]
        if openmp:
            argv.append("/openmp")
        argv += ["/LD", f"/Fe:{DLL.name}", SRC.name]
        if DEF.exists():
            argv.append(f"/DEF:{DEF.name}")
        print("Running:", " ".join(argv))
        r = subprocess.run(argv, cwd=str(HERE))
        return int(r.returncode)

    print(
        "ERROR: could not find MSVC x64 toolchain.\n"
        "  Install Visual Studio 2022 Build Tools with 'Desktop development with C++',\n"
        "  or open an 'x64 Native Tools' prompt and re-run.\n"
        "  Looked for vswhere +:",
        file=sys.stderr,
    )
    for c in _VCVARS_CANDIDATES:
        print("   ", c, "EXISTS" if c.exists() else "missing", file=sys.stderr)
    return 1


def _brew_libomp() -> Path | None:
    """Homebrew libomp prefix, if present (Apple clang ships no OpenMP)."""
    brew = shutil.which("brew")
    if brew is None:
        return None
    try:
        r = subprocess.run(
            [brew, "--prefix", "libomp"], capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    prefix = (r.stdout or "").strip()
    if not prefix:
        return None
    p = Path(prefix)
    return p if (p / "include").is_dir() else None


def unix_compile_commands(cc: str, out: Path, src: Path, openmp: bool) -> list[list[str]]:
    """Candidate compiler invocations, most-preferred first.

    Deliberately no ``-march=native``: the library selects AVX2 / AVX-512 /
    NEON at *run* time, so the object must stay portable to any CPU of the
    same architecture. Baking in build-host ISA would produce binaries that
    SIGILL on older machines — the opposite of what runtime dispatch is for.
    """
    base = ["-O3", "-shared", "-fPIC"]
    cmds: list[list[str]] = []
    if openmp:
        cmds.append([cc, *base, "-fopenmp", "-o", str(out), str(src)])
        libomp = _brew_libomp() if sys.platform == "darwin" else None
        if libomp is not None:
            # Apple clang needs libomp routed through the preprocessor.
            cmds.append([
                cc, *base,
                "-Xpreprocessor", "-fopenmp",
                f"-I{libomp / 'include'}",
                f"-L{libomp / 'lib'}",
                "-lomp",
                "-o", str(out), str(src),
            ])
    # Single-threaded fallback always builds; correctness never depends on OpenMP.
    cmds.append([cc, *base, "-o", str(out), str(src)])
    return cmds


def _compile_gcc(openmp: bool) -> int:
    cc = shutil.which("gcc") or shutil.which("clang") or shutil.which("cc")
    if cc is None:
        print(
            "ERROR: no C compiler found on PATH (looked for gcc, clang, cc).\n"
            "  Debian/Ubuntu: sudo apt-get install gcc libomp-dev\n"
            "  Fedora/RHEL:   sudo dnf install gcc libomp-devel\n"
            "  macOS:         xcode-select --install && brew install libomp\n"
            "  Alpine:        apk add build-base\n"
            "  Without a compiler the NumPy fallback still gives correct results.",
            file=sys.stderr,
        )
        return 1

    rc = 1
    for cmd in unix_compile_commands(cc, DLL, SRC, openmp):
        print("Running:", " ".join(cmd))
        r = subprocess.run(cmd, cwd=str(HERE))
        rc = int(r.returncode)
        if rc == 0 and DLL.exists():
            return 0
        if DLL.exists():
            with contextlib.suppress(OSError):
                DLL.unlink()
        print("  -> failed, trying next configuration", flush=True)
    return rc


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Build native packed GEMM library")
    p.add_argument("--force", action="store_true", help="Rebuild even if library exists")
    p.add_argument(
        "--no-openmp",
        action="store_true",
        help="Disable OpenMP (fallback single-thread)",
    )
    args, _unknown = p.parse_known_args(argv)

    if not SRC.exists():
        print(f"ERROR: missing source {SRC}", file=sys.stderr)
        return 1

    if DLL.exists() and not args.force:
        print(f"DLL exists: {DLL} (use --force to rebuild)")
        return 0
    if DLL.exists():
        try:
            DLL.unlink()
        except OSError as e:
            print(f"ERROR: cannot remove {DLL}: {e}", file=sys.stderr)
            return 1

    openmp = not args.no_openmp
    if os.name == "nt":
        rc = _compile_msvc(openmp)
        if rc != 0 and openmp:
            print("OpenMP build failed — retrying without /openmp", flush=True)
            if DLL.exists():
                with contextlib.suppress(OSError):
                    DLL.unlink()
            rc = _compile_msvc(openmp=False)
    else:
        rc = _compile_gcc(openmp)

    exists = DLL.exists()
    size = DLL.stat().st_size if exists else 0
    print("exists", exists, "size", size)
    if exists and rc == 0:
        # Reset cached loader in this process if re-imported later
        try:
            import bnn.kernels.packed as packed

            packed._NATIVE = None
            # Must clear the sticky "already tried and failed" flag too, or a
            # rebuild after a failed load would never be picked up.
            packed._NATIVE_TRIED = False
            packed._THREADS_APPLIED = None
        except Exception:
            pass
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
