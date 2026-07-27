#!/usr/bin/env python3
"""Ternary weight pack path (2-bit) — size/kernel pedagogy for BitNet-style weights.

Not a full bitnet.cpp replacement. Packs {-1,0,+1} into 2 bits/weight and verifies
unpack round-trip + float GEMM reference. Documents why speed needs a ternary kernel
(not FP dequant).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bnn.kernels.ternary_pack import (  # noqa: E402
    pack_ternary_2bit,
    ternary_bytes,
    unpack_ternary_2bit,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=1024)
    p.add_argument("--cols", type=int, default=4096)
    p.add_argument("--out", type=Path, default=ROOT / "results" / "ternary_pack.json")
    args = p.parse_args()

    rng = np.random.default_rng(0)
    # Simulate absmean ternary
    w = rng.normal(size=(args.rows, args.cols)).astype(np.float32)
    scale = float(np.mean(np.abs(w)))
    q = np.clip(np.round(w / max(scale, 1e-8)), -1, 1).astype(np.int8)

    packed = pack_ternary_2bit(q)
    q2 = unpack_ternary_2bit(packed, args.rows, args.cols)
    err = int(np.sum(q != q2))
    fp_bytes = args.rows * args.cols * 4
    t_bytes = ternary_bytes(args.rows, args.cols)
    # Naive dequant GEMM vs FP reference on small slice
    x = rng.normal(size=(8, args.cols)).astype(np.float32)
    y_ref = x @ (q.astype(np.float32).T * scale)
    y_deq = x @ (q2.astype(np.float32).T * scale)
    max_err = float(np.max(np.abs(y_ref - y_deq)))

    payload = {
        "shape": [args.rows, args.cols],
        "pack_roundtrip_errors": err,
        "fp32_weight_bytes": fp_bytes,
        "ternary_2bit_bytes": t_bytes,
        "compression_vs_fp32": fp_bytes / t_bytes,
        "dequant_gemm_max_err": max_err,
        "note": (
            "CLOSED ternary pack path: 2-bit store + unpack verified. "
            "Inference speed still requires a dedicated ternary/bitnet kernel "
            "(this path alone = size win; dequant GEMM is not faster)."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
