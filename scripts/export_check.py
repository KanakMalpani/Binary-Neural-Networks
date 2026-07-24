#!/usr/bin/env python3
"""Verify packed weight compression and GEMM numerical agreement."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bnn.determinism import set_repro_seed  # noqa: E402
from bnn.kernels.packed import binary_gemm_packed, pack_binary_pm1  # noqa: E402
from bnn.layers import BinaryLinear  # noqa: E402
from bnn.ste import binary_sign  # noqa: E402


def main() -> None:
    set_repro_seed(0, deterministic=True, force_cpu=True)
    layer = BinaryLinear(1024, 512)
    with torch.no_grad():
        w = binary_sign(layer.weight).cpu().numpy()
        x = binary_sign(torch.randn(32, 1024)).numpy()
        y_sim = (torch.from_numpy(x) @ torch.from_numpy(w).T).numpy()
        y_pack = binary_gemm_packed(x, w)
        err = np.max(np.abs(y_sim - y_pack))
        assert err < 1e-6, err

        packed, n = pack_binary_pm1(w, axis=1)
        packed_bytes = packed.nbytes
        fp_bytes = w.size * 4
        ratio = fp_bytes / packed_bytes
        print(f"GEMM max err: {err}")
        print(f"Weight FP32 bytes: {fp_bytes}")
        print(f"Packed uint64 bytes: {packed_bytes}")
        print(f"Compression: {ratio:.2f}x (ideal ~32x for bit packing into bytes)")
        # uint64 packing stores 64 bits/word => 8 bytes / 64 weights = 1 bit/weight
        ideal = 32.0
        assert ratio > 30, f"Expected ~{ideal}x, got {ratio}"
        print("export_check: PASS")


if __name__ == "__main__":
    main()
