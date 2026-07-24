"""Reproducibility helpers for golden / gated runs.

Training and demos should call ``set_repro_seed`` once at process start.
Golden verification forces CPU; bit-identical floats across OS/BLAS are not
guaranteed — gates compare conclusions (accuracy floors, exact compression,
native err=0 when the MSVC DLL is present).
"""

from __future__ import annotations

import os
import random
from typing import Any


def set_repro_seed(
    seed: int = 0,
    *,
    deterministic: bool = True,
    force_cpu: bool = True,
) -> dict[str, Any]:
    """Seed Python / NumPy / Torch; optionally enable deterministic algorithms.

    Returns a small status dict (useful for logging in result JSON).
    """
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    status: dict[str, Any] = {
        "seed": seed,
        "force_cpu": force_cpu,
        "deterministic_requested": deterministic,
        "cuda_available": torch.cuda.is_available(),
        "notes": [],
    }

    if force_cpu:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        status["device_policy"] = "cpu"
    else:
        status["device_policy"] = "auto"

    if deterministic:
        # Prefer reproducibility over throughput for golden / smoke trains.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            status["torch_deterministic"] = True
        except (TypeError, RuntimeError) as exc:
            # Older torch: warn_only may be unavailable.
            try:
                torch.use_deterministic_algorithms(True)
                status["torch_deterministic"] = True
            except RuntimeError:
                status["torch_deterministic"] = False
                status["notes"].append(f"deterministic_algorithms unavailable: {exc}")
        # Avoid nondeterministic CPU reduction paths when possible.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    else:
        status["torch_deterministic"] = False

    status["notes"].append(
        "Some ops (certain CUDA kernels, rare Conv paths) remain nondeterministic; "
        "golden repro uses CPU + tolerance gates, not bit-identical floats."
    )
    return status
