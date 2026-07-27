"""Public API surface locks (W1.T03 / W1.T04)."""

from __future__ import annotations

import torch
import torch.nn as nn

import bnn
import bnn.optimise as optimise
import bnn.wrap as wrap

REQUIRED_BNN = {
    "optimise_model",
    "OptimiseConfig",
    "OptimiseResult",
    "wrap_model",
    "__version__",
}

REQUIRED_WRAP = {
    "wrap_model",
    "WrapReport",
    "CalibConfig",
    "SCHEMA_ID",
    "validate_optimise_report",
    "attach_effectiveness",
    "measure_agreement",
}

REQUIRED_OPTIMISE = {
    "optimise_model",
    "OptimiseConfig",
    "OptimiseResult",
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "validate_optimise_report",
}


def test_bnn_public_exports():
    for name in REQUIRED_BNN:
        assert name in bnn.__all__
        assert hasattr(bnn, name)


def test_wrap_public_exports():
    for name in REQUIRED_WRAP:
        assert name in wrap.__all__
        assert hasattr(wrap, name)


def test_optimise_public_exports():
    for name in REQUIRED_OPTIMISE:
        assert name in optimise.__all__
        assert hasattr(optimise, name)


def test_optimise_model_smoke(tmp_path):
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(64, 64)
            self.ffn_fc1 = nn.Linear(64, 256)
            self.ffn_fc2 = nn.Linear(256, 64)
            self.lm_head = nn.Linear(64, 10)

        def forward(self, x):
            h = self.embed(x)
            return self.lm_head(self.ffn_fc2(torch.relu(self.ffn_fc1(h))))

    m = Tiny()
    x = torch.randn(4, 64)
    pack = tmp_path / "opt.bnnpack"
    result = optimise.optimise_model(
        m,
        x,
        policy="hybrid_ffn",
        mode="binary_xnor",
        min_in_features=32,
        qat_steps=0,
        encode_path=pack,
        encode_min_width=32,
        force=True,
    )
    assert result.report.replaced
    assert result.payload["schema"] == optimise.SCHEMA_ID
    assert result.payload["schema_version"] == optimise.SCHEMA_VERSION
    errs = optimise.validate_optimise_report(result.payload)
    assert errs == [], errs
    assert pack.is_file()
    y = result.model(x)
    assert y.shape == (4, 10)


def test_schema_envelope_valid():
    from bnn.wrap.schema import envelope, is_valid_optimise_report

    payload = envelope(
        policy="hybrid_ffn",
        mode="binary_xnor",
        replaced=["ffn_fc1"],
        skipped=[],
        compression_replaced_weights=32.0,
        fp32_weight_bytes_replaced=1024,
        packed_weight_bytes=32,
        native_kernel=False,
        drop_in_ok=True,
        forced=False,
        status="OK",
    )
    assert is_valid_optimise_report(payload)
