"""Light STE/QAT recovery: guardrails, targeting, and Linear round-trip."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from bnn.layers import BinaryLinear
from bnn.wrap.qat import _swap_linear_to_binary, light_qat_recover


class Tiny(nn.Module):
    """Names chosen so the default ffn/mlp/fc heuristic matches ffn_* only."""

    def __init__(self, d: int = 16):
        super().__init__()
        self.embed = nn.Linear(d, d)
        self.attn_qkv = nn.Linear(d, d)
        self.ffn_fc1 = nn.Linear(d, d * 2)
        self.ffn_fc2 = nn.Linear(d * 2, d)
        self.head = nn.Linear(d, d)

    def forward(self, x):
        h = torch.relu(self.embed(x))
        h = self.attn_qkv(h)
        h = torch.relu(self.ffn_fc1(h))
        h = self.ffn_fc2(h)
        return self.head(h)


def _mse_to_self(out: torch.Tensor, inp: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(out, torch.zeros_like(out))


def test_zero_steps_is_a_no_op():
    model = Tiny()
    report = light_qat_recover(model, torch.randn(4, 16), steps=0, loss_fn=_mse_to_self)
    assert report == {"steps": 0, "skipped": True}
    # Nothing swapped.
    assert not any(isinstance(m, BinaryLinear) for m in model.modules())


def test_requires_teacher_or_loss_fn():
    """The self-argmax CE fallback was removed as harmful; refuse silently doing nothing."""
    with pytest.raises(ValueError, match="teacher"):
        light_qat_recover(Tiny(), torch.randn(4, 16), steps=5)


def test_no_matching_layers_reports_reason():
    class NoFFN(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(8, 8)

        def forward(self, x):
            return self.embed(x)

    report = light_qat_recover(
        NoFFN(), torch.randn(2, 8), steps=3, loss_fn=_mse_to_self
    )
    assert report["skipped"] is True
    assert "no target" in report["reason"].lower()


def test_default_heuristic_targets_only_ffn_layers():
    model = Tiny()
    report = light_qat_recover(
        model, torch.randn(4, 16), steps=2, loss_fn=_mse_to_self
    )
    assert set(report["restored_linears"]) == {"ffn_fc1", "ffn_fc2"}
    # attn / embed / head must be left alone — wrapping attention is the
    # documented accuracy trap.
    assert "attn_qkv" not in report["restored_linears"]
    assert "embed" not in report["restored_linears"]


def test_explicit_layer_names_override_heuristic():
    model = Tiny()
    report = light_qat_recover(
        model,
        torch.randn(4, 16),
        steps=2,
        loss_fn=_mse_to_self,
        layer_names=["embed"],
    )
    assert report["restored_linears"] == ["embed"]


def test_layers_are_restored_to_plain_linear_and_model_still_runs():
    """After recovery the model must be packable again, i.e. plain nn.Linear."""
    model = Tiny()
    x = torch.randn(4, 16)
    report = light_qat_recover(model, x, steps=3, loss_fn=_mse_to_self)
    assert report["skipped"] is False

    mods = dict(model.named_modules())
    for name in report["restored_linears"]:
        assert isinstance(mods[name], nn.Linear)
        assert not isinstance(mods[name], BinaryLinear)
    assert mods["ffn_fc1"].in_features == 16
    assert mods["ffn_fc2"].out_features == 16

    out = model(x)
    assert out.shape == (4, 16)
    assert torch.isfinite(out).all()


def test_training_actually_updates_latent_weights():
    torch.manual_seed(0)
    model = Tiny()
    before = model.ffn_fc1.weight.detach().clone()
    light_qat_recover(model, torch.randn(8, 16), steps=5, lr=1e-2, loss_fn=_mse_to_self)
    after = dict(model.named_modules())["ffn_fc1"].weight.detach()
    assert not torch.allclose(before, after), "QAT ran but weights never moved"
    assert torch.isfinite(after).all()


def test_teacher_distillation_path_runs():
    torch.manual_seed(0)
    student, teacher = Tiny(), Tiny()
    report = light_qat_recover(
        student, torch.randn(4, 16), steps=3, teacher=teacher
    )
    assert report["skipped"] is False
    assert isinstance(report["last_loss"], float)
    assert report["last_loss"] == report["last_loss"]  # not NaN


def test_model_left_in_eval_mode():
    model = Tiny()
    model.train()
    light_qat_recover(model, torch.randn(4, 16), steps=2, loss_fn=_mse_to_self)
    assert not model.training


def test_swap_preserves_weights_and_bias_presence():
    lin = nn.Linear(6, 3, bias=True)
    swapped = _swap_linear_to_binary(lin)
    assert isinstance(swapped, BinaryLinear)
    assert torch.allclose(swapped.weight, lin.weight)
    assert swapped.bias is not None
    assert torch.allclose(swapped.bias, lin.bias)

    nobias = nn.Linear(6, 3, bias=False)
    assert _swap_linear_to_binary(nobias).bias is None


def test_report_carries_production_caveat():
    """The note keeps callers from reading a toy recovery as production QAT."""
    report = light_qat_recover(Tiny(), torch.randn(4, 16), steps=1, loss_fn=_mse_to_self)
    assert "BitDistill" in report["note"]
