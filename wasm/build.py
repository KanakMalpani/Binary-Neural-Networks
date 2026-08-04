#!/usr/bin/env python3
"""Optional builder for pedagogy WASM GEMM (W2.T06).

Tries, in order unless forced:
  1. emcc (Emscripten)
  2. clang --target=wasm32
  3. cargo + wasm32-unknown-unknown (rust/ crate)

Never fails the repo gate - exit 0 with a skip message when no toolchain is
present. Successful builds write wasm/dist/binary_gemm_wasm.wasm.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist"
SRC_C = ROOT / "binary_gemm_wasm.c"
RUST_DIR = ROOT / "rust"
OUT_WASM = DIST / "binary_gemm_wasm.wasm"


def _which(name: str) -> str | None:
    return shutil.which(name)


def build_emcc() -> bool:
    emcc = _which("emcc")
    if not emcc:
        return False
    DIST.mkdir(parents=True, exist_ok=True)
    cmd = [
        emcc,
        str(SRC_C),
        "-O2",
        "-msimd128",
        "-s",
        "STANDALONE_WASM=1",
        "-s",
        "EXPORTED_FUNCTIONS=_binary_gemm_wasm_u64,_binary_gemm_wasm_kernel_id,_binary_gemm_wasm_set_kernel",
        "-s",
        "EXPORTED_RUNTIME_METHODS=",
        "--no-entry",
        "-o",
        str(OUT_WASM),
    ]
    print("+", " ".join(cmd))
    subprocess.check_call(cmd)
    return True


def build_clang() -> bool:
    clang = _which("clang")
    if not clang:
        return False
    DIST.mkdir(parents=True, exist_ok=True)
    obj = DIST / "binary_gemm_wasm.o"
    cmd_cc = [
        clang,
        "--target=wasm32",
        "-O2",
        "-msimd128",
        "-c",
        str(SRC_C),
        "-o",
        str(obj),
    ]
    print("+", " ".join(cmd_cc))
    subprocess.check_call(cmd_cc)
    wasm_ld = _which("wasm-ld") or _which("wasm-ld.exe")
    if not wasm_ld:
        cmd_link = [
            clang,
            "--target=wasm32",
            "-nostdlib",
            "-Wl,--no-entry",
            "-Wl,--export=binary_gemm_wasm_u64",
            "-Wl,--export=binary_gemm_wasm_kernel_id",
            "-Wl,--export=binary_gemm_wasm_set_kernel",
            "-Wl,--allow-undefined",
            str(obj),
            "-o",
            str(OUT_WASM),
        ]
        print("+", " ".join(cmd_link))
        subprocess.check_call(cmd_link)
    else:
        cmd_link = [
            wasm_ld,
            str(obj),
            "--no-entry",
            "--export=binary_gemm_wasm_u64",
            "--export=binary_gemm_wasm_kernel_id",
            "--export=binary_gemm_wasm_set_kernel",
            "-o",
            str(OUT_WASM),
        ]
        print("+", " ".join(cmd_link))
        subprocess.check_call(cmd_link)
    return True


def _prefer_bnn_wasm(candidates: list[Path]) -> Path | None:
    """Prefer newest bnn_wasm_gemm.wasm-like name, else newest *.wasm."""
    if not candidates:
        return None
    named = [p for p in candidates if "bnn_wasm_gemm" in p.name or p.name.startswith("bnn") and "wasm" in p.name]
    pool = named if named else candidates
    return max(pool, key=lambda p: p.stat().st_mtime)


def _find_rust_wasm() -> Path:
    """Locate cargo cdylib .wasm under default or redirected CARGO_TARGET_DIR."""
    release_dirs: list[Path] = [
        RUST_DIR / "target" / "wasm32-unknown-unknown" / "release",
    ]
    cargo_target = os.environ.get("CARGO_TARGET_DIR")
    if cargo_target:
        release_dirs.append(
            Path(cargo_target) / "wasm32-unknown-unknown" / "release"
        )

    candidates: list[Path] = []
    for d in release_dirs:
        preferred = d / "bnn_wasm_gemm.wasm"
        if preferred.is_file():
            candidates.append(preferred)
        if d.is_dir():
            candidates.extend(p for p in d.glob("*.wasm") if p.is_file())

    # Recursive fallback under RUST_DIR/target (and CARGO_TARGET_DIR if set)
    search_roots = [RUST_DIR / "target"]
    if cargo_target:
        search_roots.append(Path(cargo_target))
    for root in search_roots:
        if not root.is_dir():
            continue
        for pat in ("bnn*wasm*.wasm", "*.wasm"):
            candidates.extend(p for p in root.rglob(pat) if p.is_file())

    # Deduplicate while preserving paths
    uniq: dict[Path, Path] = {}
    for p in candidates:
        try:
            uniq[p.resolve()] = p
        except OSError:
            uniq[p] = p
    found = _prefer_bnn_wasm(list(uniq.values()))
    if found is None:
        raise FileNotFoundError("cargo built no .wasm")
    return found


def build_rust() -> bool:
    cargo = _which("cargo")
    if not cargo or not (RUST_DIR / "Cargo.toml").is_file():
        return False
    rustup = _which("rustup")
    if rustup:
        subprocess.call(
            [rustup, "target", "add", "wasm32-unknown-unknown"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    DIST.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    extra = "-C target-feature=+simd128"
    prev = env.get("RUSTFLAGS", "").strip()
    env["RUSTFLAGS"] = f"{prev} {extra}".strip()
    cmd = [
        cargo,
        "build",
        "--release",
        "--target",
        "wasm32-unknown-unknown",
        "--manifest-path",
        str(RUST_DIR / "Cargo.toml"),
    ]
    print("+", " ".join(cmd), f"(RUSTFLAGS={env['RUSTFLAGS']!r})")
    subprocess.check_call(cmd, env=env)
    built = _find_rust_wasm()
    shutil.copy2(built, OUT_WASM)
    print(f"copied {built} -> {OUT_WASM}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emcc", action="store_true", help="force Emscripten")
    ap.add_argument("--clang", action="store_true", help="force clang wasm32")
    ap.add_argument("--rust", action="store_true", help="force Rust cdylib")
    args = ap.parse_args()

    forced = args.emcc or args.clang or args.rust
    try:
        if args.emcc:
            ok = build_emcc()
        elif args.clang:
            ok = build_clang()
        elif args.rust:
            ok = build_rust()
        else:
            ok = build_emcc() or build_clang() or build_rust()
    except subprocess.CalledProcessError as e:
        print(f"build failed: {e}", file=sys.stderr)
        return 1

    if ok and OUT_WASM.is_file():
        print(f"OK: {OUT_WASM} ({OUT_WASM.stat().st_size} bytes)")
        return 0
    if forced:
        print("requested toolchain unavailable or failed", file=sys.stderr)
        return 1
    print(
        "SKIP: no emcc/clang/cargo wasm toolchain - JS/Python scalar paths still work"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
