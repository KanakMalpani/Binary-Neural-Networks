"""Compile the native binary/ternary GEMM shared library (Windows MSVC or Linux GCC)."""

from __future__ import annotations

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
        # Already in a developer prompt
        cmd = ["cl", "/nologo", "/O2"]
        if openmp:
            cmd.append("/openmp")
        cmd += ["/LD", f"/Fe:{DLL.name}", SRC.name]
        if DEF.exists():
            cmd.append(f"/DEF:{DEF.name}")
        print("Running:", " ".join(cmd))
        r = subprocess.run(cmd, cwd=str(HERE))
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


def _compile_gcc(openmp: bool) -> int:
    cc = shutil.which("gcc") or shutil.which("clang")
    if cc is None:
        print("ERROR: gcc/clang not found on PATH", file=sys.stderr)
        return 1
    cmd = [cc, "-O3", "-shared", "-fPIC", "-o", str(DLL), str(SRC)]
    if openmp:
        # Insert -fopenmp before -o
        cmd = [cc, "-O3", "-fopenmp", "-shared", "-fPIC", "-o", str(DLL), str(SRC)]
    print("Running:", " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(HERE))
    if r.returncode != 0 and openmp:
        print("OpenMP build failed — retrying without -fopenmp", flush=True)
        if DLL.exists():
            DLL.unlink()
        cmd2 = [cc, "-O3", "-shared", "-fPIC", "-o", str(DLL), str(SRC)]
        print("Running:", " ".join(cmd2))
        r = subprocess.run(cmd2, cwd=str(HERE))
    return int(r.returncode)


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
                try:
                    DLL.unlink()
                except OSError:
                    pass
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
            packed._THREADS_APPLIED = None
        except Exception:
            pass
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
