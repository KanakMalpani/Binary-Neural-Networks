"""Profile CLI smoke."""

from __future__ import annotations

import json
from pathlib import Path

from bnn.cli import main as cli_main
from bnn.profile import profile_packed_linear


def test_profile_breakdown_keys():
    br = profile_packed_linear(m=8, n=256, k=256, reps=3, warmup=1)
    d = br.to_dict()
    for key in (
        "pack_weight_ms",
        "pack_act_ms",
        "gemm_ms",
        "e2e_forward_ms",
        "torch_fp32_ms",
        "speedup_vs_fp32",
    ):
        assert key in d
        assert d[key] >= 0
    # overhead can be slightly negative under timer noise when e2e < isolated gemm
    assert "overhead_vs_gemm" in d
    assert isinstance(d["overhead_vs_gemm"], float)

def test_cli_profile(tmp_path: Path):
    out = tmp_path / "prof.json"
    assert (
        cli_main(
            [
                "profile",
                "--batch",
                "8",
                "--in-features",
                "256",
                "--out-features",
                "256",
                "--reps",
                "3",
                "--warmup",
                "1",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["n"] == 256
