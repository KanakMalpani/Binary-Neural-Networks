"""Build the native packed-GEMM library into the wheel.

Everything user-facing lives in pyproject.toml; this file exists only because
the kernel is C. The goal is that `pip install bnn-lab` gives a working fast path on
an ordinary machine with no compiler and no MSVC/Xcode setup.

Two rules shape this file:

1. **Never fail the install.** If no compiler, no OpenMP, or an unsupported
   platform, the extension is dropped and the package still imports — the NumPy
   fallback is correct, just slower. A quantisation library that refuses to
   install is worse than a slow one.
2. **Never bake in the build host's ISA.** No `-march=native`. The kernel picks
   AVX-512 / AVX2 / NEON at *run* time, so one wheel must stay valid on every
   CPU of that architecture.
"""

from __future__ import annotations

import contextlib
import os
import sys

from setuptools import setup
from setuptools.command.build_ext import build_ext
from setuptools.extension import Extension

SOURCE = "bnn/kernels/binary_gemm.c"

# Loaded via ctypes, not imported as a Python module — the name only has to
# match what bnn/kernels/packed.py globs for.
EXTENSION = Extension(
    "bnn.kernels._binary_gemm_native",
    sources=[SOURCE],
    optional=True,  # a build failure must not abort the install
)


def _openmp_flags(compiler_type: str) -> tuple[list[str], list[str]]:
    """(compile_args, link_args) enabling OpenMP, or ([], []) if unavailable.

    Only flags setuptools does *not* already supply. MSVC gets
    ``/O2 /W3 /GL /DNDEBUG /MD`` by default, so repeating ``/O2`` here just
    passed it twice. On Unix, Python's own ``CFLAGS`` usually carry ``-O2``;
    ``-O3`` is appended deliberately because the last optimisation flag wins and
    the popcount loops benefit from it.
    """
    if compiler_type == "msvc":
        return ["/openmp"], []
    if sys.platform == "darwin":
        # Apple clang ships no OpenMP runtime; libomp comes from brew/conda and
        # is frequently absent. Try it, fall back to single-threaded.
        prefix = os.environ.get("LIBOMP_PREFIX")
        if prefix:
            return (
                ["-O3", "-Xpreprocessor", "-fopenmp", f"-I{prefix}/include"],
                [f"-L{prefix}/lib", "-lomp"],
            )
        return ["-O3", "-Xpreprocessor", "-fopenmp"], ["-lomp"]
    return ["-O3", "-fopenmp"], ["-fopenmp"]


def _plain_flags(compiler_type: str) -> list[str]:
    """Single-threaded fallback: MSVC's default /O2 is already correct."""
    return [] if compiler_type == "msvc" else ["-O3"]


class BuildExtWithOpenMPFallback(build_ext):
    """Try OpenMP; silently retry single-threaded; never abort the install."""

    def get_export_symbols(self, ext: Extension) -> list[str]:  # type: ignore[override]
        """No ``PyInit_*`` — this is a plain shared library loaded via ctypes.

        setuptools otherwise passes ``/EXPORT:PyInit__binary_gemm_native`` to
        the MSVC linker, which fails with an unresolved external and leaves a
        zero-byte artifact behind.
        """
        return []

    def _discard_stale_output(self, ext: Extension) -> None:
        """Remove a partial artifact so the retry actually relinks.

        Without this, ``build_extension`` sees an existing output newer than the
        sources, reports "up-to-date", and silently keeps the broken file.
        """
        path = self.get_ext_fullpath(ext.name)
        if os.path.exists(path):
            with contextlib.suppress(OSError):
                os.remove(path)

    def build_extension(self, ext: Extension) -> None:  # type: ignore[override]
        compiler_type = self.compiler.compiler_type
        saved_compile = list(ext.extra_compile_args or [])
        saved_link = list(ext.extra_link_args or [])

        compile_args, link_args = _openmp_flags(compiler_type)
        ext.extra_compile_args = saved_compile + compile_args
        ext.extra_link_args = saved_link + link_args
        try:
            super().build_extension(ext)
            if self._output_is_usable(ext):
                return
            print("NOTE: OpenMP build produced no usable library; retrying without it.")
        except Exception as exc:  # noqa: BLE001 - any toolchain error is non-fatal
            print(f"NOTE: OpenMP build failed ({exc}); retrying without it.")

        self._discard_stale_output(ext)
        self.force = True
        ext.extra_compile_args = saved_compile + _plain_flags(compiler_type)
        ext.extra_link_args = saved_link
        try:
            super().build_extension(ext)
        except Exception as exc:  # noqa: BLE001
            print(
                f"NOTE: native kernel unavailable ({exc}). "
                "Installing without it — the NumPy fallback is correct, just slower. "
                "Build later with: python -m bnn.kernels.compile_native",
            )
        if not self._output_is_usable(ext):
            # Never ship an empty stub: the loader would try it and warn.
            self._discard_stale_output(ext)

    def _output_is_usable(self, ext: Extension) -> bool:
        path = self.get_ext_fullpath(ext.name)
        return os.path.exists(path) and os.path.getsize(path) > 0


setup(ext_modules=[EXTENSION], cmdclass={"build_ext": BuildExtWithOpenMPFallback})
