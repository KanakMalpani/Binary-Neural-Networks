"""export_check compression gate + packed export."""

from __future__ import annotations

import numpy as np
import torch

from bnn.export import pack_linear_weight, save_checkpoint, load_checkpoint
from bnn.kernels.packed import binary_gemm_packed, pack_binary_pm1
from bnn.layers import BinaryLinear
from bnn.models import build_model
from bnn.ste import binary_sign


def test_export_check_compression_and_err():
    torch.manual_seed(0)
    layer = BinaryLinear(1024, 512)
    with torch.no_grad():
        w = binary_sign(layer.weight).cpu().numpy()
        x = binary_sign(torch.randn(32, 1024)).numpy()
        y_sim = (torch.from_numpy(x) @ torch.from_numpy(w).T).numpy()
        y_pack = binary_gemm_packed(x, w)
        err = float(np.max(np.abs(y_sim - y_pack)))
        assert err < 1e-6
        packed, _ = pack_binary_pm1(w, axis=1)
        ratio = (w.size * 4) / packed.nbytes
        assert abs(ratio - 32.0) < 0.01 or ratio > 30.0


def test_checkpoint_roundtrip(tmp_path):
    m = build_model("binary_mlp", hidden=64)
    path = tmp_path / "m.pt"
    save_checkpoint(m, path, meta={"name": "binary_mlp"})
    m2 = build_model("binary_mlp", hidden=64)
    meta = load_checkpoint(m2, path)
    assert meta["name"] == "binary_mlp"
    x = torch.randn(2, 1, 28, 28)
    assert torch.allclose(m(x), m2(x), atol=1e-5)


def test_pack_linear_weight_bytes():
    w = torch.randn(128, 256)
    blob = pack_linear_weight(w)
    assert blob["compression"] >= 30.0
