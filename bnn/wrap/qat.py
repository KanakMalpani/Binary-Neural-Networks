"""Light STE/QAT recovery after wrap (few steps — not full BitDistill).

Limits: recovers cosine/accuracy only modestly on toy stacks; production HF
models need real data + longer distillation (see docs/12, docs/33).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import BinaryLinear
from ..ste import clip_weights_


def _swap_linear_to_binary(lin: nn.Linear) -> BinaryLinear:
    bl = BinaryLinear(lin.in_features, lin.out_features, bias=lin.bias is not None)
    with torch.no_grad():
        bl.weight.copy_(lin.weight)
        if lin.bias is not None and bl.bias is not None:
            bl.bias.copy_(lin.bias)
    return bl


def light_qat_recover(
    model: nn.Module,
    calib_x: torch.Tensor,
    *,
    teacher: nn.Module | None = None,
    steps: int = 50,
    lr: float = 1e-3,
    layer_names: list[str] | None = None,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> dict:
    """Short STE fine-tune on named Linears (default: modules named *ffn* / *mlp*).

    If ``teacher`` is given, distill via KL on logits; else CE on random labels
    is **not** used — requires ``calib_y`` via loss_fn or teacher.
    """
    if steps <= 0:
        return {"steps": 0, "skipped": True}

    if teacher is None and loss_fn is None:
        raise ValueError(
            "light_qat_recover requires teacher=... or loss_fn=... "
            "(self-argmax CE fallback removed — it was a no-op / harmful)"
        )

    # Collect target Linears
    targets: list[tuple[str, nn.Linear]] = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        lname = name.lower()
        if layer_names is not None:
            if name not in layer_names:
                continue
        elif not any(k in lname for k in ("ffn", "mlp", "fc1", "fc2", "intermediate")):
            continue
        targets.append((name, mod))

    if not targets:
        return {"steps": 0, "skipped": True, "reason": "no target Linears"}

    def _set(root: nn.Module, path: str, new: nn.Module) -> None:
        parts = path.split(".")
        parent = root
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], new)

    # Swap to BinaryLinear for STE
    for name, lin in targets:
        _set(model, name, _swap_linear_to_binary(lin))

    model.train()
    if teacher is not None:
        teacher.eval()

    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
    )
    last_loss = 0.0
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        student_out = model(calib_x)
        if loss_fn is not None:
            loss = loss_fn(student_out, calib_x)
        elif teacher is not None:
            with torch.no_grad():
                t_out = teacher(calib_x)
            loss = F.kl_div(
                F.log_softmax(student_out, dim=-1),
                F.softmax(t_out, dim=-1),
                reduction="batchmean",
            )
        else:
            # Self-consistency fallback removed — require teacher or loss_fn
            raise ValueError("unreachable: teacher/loss_fn required")

        loss.backward()
        opt.step()
        clip_weights_(model)
        last_loss = float(loss.detach().item())

    model.eval()
    # Restore nn.Linear with learned latent weights for subsequent packed wrap
    restored: list[str] = []
    for name, _ in targets:
        bl = dict(model.named_modules())[name]
        if not isinstance(bl, BinaryLinear):
            continue
        lin = nn.Linear(bl.in_features, bl.out_features, bias=bl.bias is not None)
        with torch.no_grad():
            lin.weight.copy_(bl.weight)
            if bl.bias is not None and lin.bias is not None:
                lin.bias.copy_(bl.bias)
        _set(model, name, lin)
        restored.append(name)

    return {
        "steps": steps,
        "skipped": False,
        "last_loss": last_loss,
        "restored_linears": restored,
        "note": "Light STE only; production needs BitDistill-scale QAT on real data",
    }
