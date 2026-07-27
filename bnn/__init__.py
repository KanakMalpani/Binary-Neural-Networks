"""bnn — Extreme low-bit inference lab (packed binary/ternary CPU kernels + STE).

Thesis
------
Cut **inference** latency/memory on **CPU/edge** with real XNOR/popcount (or
ternary) kernels. Training with STE is simulation — not a GPU 32× claim.
For commodity NVIDIA GPUs prefer INT4/FP8 stacks (torchao / vLLM / AWQ).
"""

from __future__ import annotations

from ._version import __version__
from .determinism import set_repro_seed
from .export import (
    load_checkpoint,
    load_packed_linears,
    pack_linear_weight,
    save_checkpoint,
    save_packed_linears,
)
from .layers import (
    BinaryConv2d,
    BinaryLinear,
    BiRealBlock,
    TernaryLinear,
    fuse_binary_conv_bn_,
    fuse_bireal_bn_,
)
from .models import build_model, count_parameters
from .optimise import OptimiseConfig, OptimiseResult, optimise_model
from .ste import binary_sign, clip_weights_, ternary_weight
from .wrap.policy import recommend_wrap_policy
from .wrapper import model_param_bytes, wrap_linear_modules, wrap_model

__all__ = [
    "__version__",
    # STE / layers
    "BinaryLinear",
    "BinaryConv2d",
    "BiRealBlock",
    "TernaryLinear",
    "fuse_binary_conv_bn_",
    "fuse_bireal_bn_",
    "binary_sign",
    "ternary_weight",
    "clip_weights_",
    # Models
    "build_model",
    "count_parameters",
    # Optimiser (preferred product API)
    "optimise_model",
    "OptimiseConfig",
    "OptimiseResult",
    # Wrap / export
    "wrap_linear_modules",
    "wrap_model",
    "model_param_bytes",
    "recommend_wrap_policy",
    "save_checkpoint",
    "load_checkpoint",
    "save_packed_linears",
    "load_packed_linears",
    "pack_linear_weight",
    # Repro
    "set_repro_seed",
]
