"""Compile the native binary GEMM DLL (Windows x64)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "binary_gemm.c"
DLL = HERE / "_binary_gemm_native.dll"
VCVARS = Path(
    r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
)


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true", help="Rebuild even if DLL exists")
    args, _unknown = p.parse_known_args(argv)

    if DLL.exists() and not args.force:
        print(f"DLL exists: {DLL} (use --force to rebuild)")
        return 0
    if DLL.exists():
        DLL.unlink()
    if not VCVARS.exists():
        print("vcvars64.bat not found — install VS Build Tools x64", file=sys.stderr)
        print("Tried:", VCVARS, file=sys.stderr)
        return 1
    # MSVC needs __popcnt64 from intrin.h — patch source for MSVC if needed
    src_text = SRC.read_text(encoding="utf-8")
    if "__builtin_popcountll" in src_text and "_MSC_VER" not in src_text:
        SRC.write_text(
            """#include <stdint.h>
#include <stddef.h>
#ifdef _MSC_VER
#include <intrin.h>
static inline int pop64(uint64_t x) { return (int)__popcnt64(x); }
#else
static inline int pop64(uint64_t x) { return __builtin_popcountll(x); }
#endif

#ifdef _WIN32
__declspec(dllexport)
#endif
void binary_gemm_u64(
    const uint64_t* X,
    const uint64_t* W,
    float* Y,
    int B, int M, int words, int n
) {
    for (int b = 0; b < B; ++b) {
        const uint64_t* xb = X + (size_t)b * words;
        float* yb = Y + (size_t)b * M;
        for (int m = 0; m < M; ++m) {
            const uint64_t* wm = W + (size_t)m * words;
            int dist = 0;
            int w = 0;
            for (; w + 3 < words; w += 4) {
                dist += pop64(xb[w] ^ wm[w]);
                dist += pop64(xb[w + 1] ^ wm[w + 1]);
                dist += pop64(xb[w + 2] ^ wm[w + 2]);
                dist += pop64(xb[w + 3] ^ wm[w + 3]);
            }
            for (; w < words; ++w) {
                dist += pop64(xb[w] ^ wm[w]);
            }
            yb[m] = (float)(n - 2 * dist);
        }
    }
}
""",
            encoding="utf-8",
        )

    cmd = f'"{VCVARS}" && cl /nologo /O2 /LD /Fe:{DLL.name} {SRC.name}'
    print("Running:", cmd)
    r = subprocess.run(cmd, shell=True, cwd=str(HERE))
    print("exists", DLL.exists(), "size", DLL.stat().st_size if DLL.exists() else 0)
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
