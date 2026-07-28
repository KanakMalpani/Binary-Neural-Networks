"""Memory footprint accounting for wrapped models (W13.T05).

Reports where the bytes actually go: resident weights, per-layer packing gain,
and the transient buffers a packed forward allocates.

Dual-metric rule applies here too. Weight bytes are **measured** from the real
buffers on the module, not inferred from a pack ratio — a `TernaryWeightOnlyLinear`
stores int8 on disk while advertising a theoretical 2-bit size, and conflating
the two is exactly the kind of overclaim this repo exists to avoid. Both numbers
are reported side by side and labelled.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch.nn as nn

from .wrap.packed_linear import (
    BinaryWeightOnlyDequantLinear,
    PackedBinaryConv2d,
    PackedBinaryXNORLinear,
    TernaryWeightOnlyLinear,
)

_PACKED_TYPES = (
    PackedBinaryXNORLinear,
    TernaryWeightOnlyLinear,
    BinaryWeightOnlyDequantLinear,
    PackedBinaryConv2d,
)


@dataclass
class LayerFootprint:
    """Bytes attributable to one module."""

    name: str
    kind: str
    packed: bool
    resident_bytes: int
    theoretical_bytes: int
    fp32_equivalent_bytes: int

    @property
    def resident_compression(self) -> float:
        """What you actually save in RAM today."""
        return self.fp32_equivalent_bytes / max(self.resident_bytes, 1)

    @property
    def theoretical_compression(self) -> float:
        """What the encoding allows — not necessarily what is stored."""
        return self.fp32_equivalent_bytes / max(self.theoretical_bytes, 1)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["resident_compression"] = self.resident_compression
        d["theoretical_compression"] = self.theoretical_compression
        return d


@dataclass
class MemoryReport:
    """Whole-model footprint, split into packed and unpacked contributions."""

    layers: list[LayerFootprint] = field(default_factory=list)
    other_param_bytes: int = 0
    other_buffer_bytes: int = 0

    @property
    def packed_layers(self) -> list[LayerFootprint]:
        return [x for x in self.layers if x.packed]

    def totals(self) -> dict[str, int]:
        resident = sum(x.resident_bytes for x in self.layers)
        theoretical = sum(x.theoretical_bytes for x in self.layers)
        fp32 = sum(x.fp32_equivalent_bytes for x in self.layers)
        return {
            "tracked_resident_bytes": resident,
            "tracked_theoretical_bytes": theoretical,
            "tracked_fp32_equivalent_bytes": fp32,
            "other_param_bytes": self.other_param_bytes,
            "other_buffer_bytes": self.other_buffer_bytes,
            "model_resident_bytes": resident + self.other_param_bytes + self.other_buffer_bytes,
        }

    def to_dict(self) -> dict[str, Any]:
        t = self.totals()
        fp32 = t["tracked_fp32_equivalent_bytes"]
        # Whole-model ratio including the FP parts that were deliberately not
        # wrapped (attention, norms, embeddings) — the honest end-to-end number.
        whole_fp32 = fp32 + t["other_param_bytes"] + t["other_buffer_bytes"]
        return {
            "schema": "bnn_memory_report_v1",
            **t,
            "tracked_resident_compression": fp32 / max(t["tracked_resident_bytes"], 1),
            "tracked_theoretical_compression": fp32 / max(t["tracked_theoretical_bytes"], 1),
            "whole_model_resident_compression": (
                whole_fp32 / max(t["model_resident_bytes"], 1)
            ),
            "packed_layer_count": len(self.packed_layers),
            "layer_count": len(self.layers),
            "layers": [x.to_dict() for x in self.layers],
            "thesis_note": (
                "resident_* is measured from real buffers; theoretical_* is the "
                "encoding's pack ratio. Neither is a latency claim — use "
                "`bnn profile` / `bnn bench` for wall-clock."
            ),
        }


def _module_bytes(mod: nn.Module) -> int:
    """Resident bytes owned directly by this module (no children)."""
    own = 0
    for p in mod.parameters(recurse=False):
        own += p.numel() * p.element_size()
    for b in mod.buffers(recurse=False):
        own += b.numel() * b.element_size()
    return int(own)


def _fp32_equivalent(mod: nn.Module) -> int:
    """Bytes the same weights would need as dense FP32."""
    if isinstance(mod, PackedBinaryXNORLinear):
        return int(mod.in_features * mod.out_features * 4)
    if isinstance(mod, TernaryWeightOnlyLinear):
        return int(mod.in_features * mod.out_features * 4)
    if isinstance(mod, BinaryWeightOnlyDequantLinear):
        return int(mod.in_features * mod.out_features * 4)
    if isinstance(mod, PackedBinaryConv2d):
        kh, kw = mod.kernel_size
        return int(mod.in_channels * mod.out_channels * kh * kw * 4)
    if isinstance(mod, nn.Linear):
        return int(mod.in_features * mod.out_features * 4)
    if isinstance(mod, nn.Conv2d):
        return int(mod.weight.numel() * 4)
    return _module_bytes(mod)


def memory_report(model: nn.Module) -> MemoryReport:
    """Per-layer resident vs theoretical footprint for ``model``.

    Only Linear/Conv-shaped modules (packed or not) are tracked as layers;
    everything else — embeddings, norms, biases on other modules — is summed
    into ``other_*`` so the totals still reconcile with the real model size.
    """
    report = MemoryReport()
    tracked: set[int] = set()

    for name, mod in model.named_modules():
        if not isinstance(mod, (*_PACKED_TYPES, nn.Linear, nn.Conv2d)):
            continue
        packed = isinstance(mod, _PACKED_TYPES)
        resident = _module_bytes(mod)
        fp32 = _fp32_equivalent(mod)
        # isinstance against the literal tuple, so the packed types narrow:
        # nn.Module.__getattr__ is typed Tensor | Module, and neither a
        # hasattr guard nor a bool flag lets a type checker resolve the call.
        if isinstance(
            mod,
            (
                PackedBinaryXNORLinear,
                TernaryWeightOnlyLinear,
                BinaryWeightOnlyDequantLinear,
                PackedBinaryConv2d,
            ),
        ):
            theoretical = int(mod.packed_weight_bytes())
        else:
            theoretical = resident
        report.layers.append(
            LayerFootprint(
                name=name or "<root>",
                kind=type(mod).__name__,
                packed=packed,
                resident_bytes=resident,
                theoretical_bytes=theoretical,
                fp32_equivalent_bytes=fp32,
            )
        )
        tracked.update(id(p) for p in mod.parameters(recurse=False))
        tracked.update(id(b) for b in mod.buffers(recurse=False))

    report.other_param_bytes = int(
        sum(p.numel() * p.element_size() for p in model.parameters() if id(p) not in tracked)
    )
    report.other_buffer_bytes = int(
        sum(b.numel() * b.element_size() for b in model.buffers() if id(b) not in tracked)
    )
    return report


def forward_transient_bytes(
    batch: int, in_features: int, out_features: int
) -> dict[str, float]:
    """Transient bytes a packed Linear forward allocates, by stage.

    Useful for sizing edge deployments: the weight saving is permanent, but a
    forward still needs packed activations and an FP32 output.
    """
    words = (in_features + 63) // 64
    packed_act = batch * words * 8
    output = batch * out_features * 4
    fp32_act = batch * in_features * 4
    return {
        "packed_activation_bytes": int(packed_act),
        "output_bytes": int(output),
        "fp32_activation_bytes": int(fp32_act),
        "total_transient_bytes": int(packed_act + output),
        "activation_pack_compression": float(fp32_act) / max(packed_act, 1),
    }
