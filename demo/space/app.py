"""Gradio CPU Space: wrap paradox on committed ultra_wrap / wrap_demo shapes.

Thesis lock: 32× is uint64 pack *size*, not GPU latency from sign(). Dual
metrics. Never imply drop-in when cosine is junk.

Live path calls ``bnn.optimise.optimise_model`` (and a short FP distill for
the ternary column). Cosine / latency / REFUSE are measured here — not
hardcoded 0.70 — so a later wrap win can show up after a ``bnn-lab`` bump.
"""

from __future__ import annotations

import copy
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from bnn import __version__ as BNN_VERSION
from bnn.determinism import set_repro_seed
from bnn.kernels.packed import native_kernel_available
from bnn.optimise import OptimiseConfig, optimise_model
from bnn.wrap.policy import detect_hardware, recommend_wrap_policy

# cpu-basic = 2 vCPU. Keep BLAS from oversubscribing.
torch.set_num_threads(2)

DROP_IN = 0.85
QAT_STEPS = 20  # same recipe as scripts/ultra_wrap_demo.py ternary path
WARMUP, REPS = 2, 8

# Published goldens — labels only. Do not use these as live outputs.
GOLDEN_AUTO = (
    "Default **auto** (committed `ultra_wrap.json` primary, d=512 ff=2048 "
    "batch=64): cosine **~0.70**, e2e **~1.61×**, status **REFUSE** "
    "(`drop_in_ok: false`). That is hybrid + native XNOR, not the 0.31 wrap."
)
GOLDEN_LEGACY = (
    "Legacy **`wrap_demo.json`** (binary_xnor, no QAT, hidden=4096 batch=64): "
    "cosine **0.31**, e2e **~4.8×**. Fast-ish kernel, cosine junk — not drop-in."
)
GOLDEN_TERNARY = (
    "Committed **ternary + QAT** (`ultra_wrap.json` ternary_accurate_path): "
    "cosine **~0.991**, e2e **~0.73×**. Accuracy-first; **loses** wall-clock "
    "(no ternary GEMM). Does not satisfy the hybrid AND-gate (0.85 **and** 1.5×)."
)

THESIS = (
    "**32× is uint64 pack compression, not GPU speed from `sign()`.** "
    "Pack ratio, cosine, and e2e latency are different physics. "
    "This Space is CPU-only (`cpu-basic`)."
)


class TinyBlock(nn.Module):
    """Same stack as ``scripts/ultra_wrap_demo.py`` (committed ultra_wrap class)."""

    def __init__(self, d: int = 512, ff: int = 2048, n_classes: int = 10):
        super().__init__()
        self.embed = nn.Linear(28 * 28, d)
        self.attn_qkv = nn.Linear(d, d)
        self.ffn_fc1 = nn.Linear(d, ff)
        self.ffn_fc2 = nn.Linear(ff, d)
        self.lm_head = nn.Linear(d, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.embed(x))
        h = h + F.relu(self.attn_qkv(h))
        h = h + self.ffn_fc2(F.relu(self.ffn_fc1(h)))
        return self.lm_head(h)


class TinyCNN(nn.Module):
    """Stem stays FP (hybrid skip); FFN names match the wrap allowlist."""

    def __init__(self, d: int = 128, ff: int = 512, n_classes: int = 10):
        super().__init__()
        self.stem = nn.Conv2d(1, 8, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.embed = nn.Linear(8 * 4 * 4, d)
        self.ffn_fc1 = nn.Linear(d, ff)
        self.ffn_fc2 = nn.Linear(ff, d)
        self.head = nn.Linear(d, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.stem(x))
        h = self.pool(h).flatten(1)
        h = F.relu(self.embed(h))
        h = self.ffn_fc2(F.relu(self.ffn_fc1(h)))
        return self.head(h)


def _fmt_bytes(n: int) -> str:
    if n >= 1_048_576:
        return f"{n / 1_048_576:.2f} MiB"
    if n >= 1024:
        return f"{n / 1024:.1f} KiB"
    return f"{n} B"


def _time_ms(fn, *, warmup: int = WARMUP, reps: int = REPS) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        return (time.perf_counter() - t0) / max(reps, 1) * 1e3


def _total_bytes(payload: dict[str, Any] | None, fallback_model: nn.Module) -> int:
    if payload:
        after = payload.get("param_bytes_after") or {}
        if "total_bytes" in after:
            return int(after["total_bytes"])
        before = payload.get("param_bytes_before") or {}
        if "total_bytes" in before:
            return int(before["total_bytes"])
    p = sum(t.numel() * t.element_size() for t in fallback_model.parameters())
    b = sum(t.numel() * t.element_size() for t in fallback_model.buffers())
    return int(p + b)


def _fp_mse_distill(
    student: nn.Module,
    teacher: nn.Module,
    x: torch.Tensor,
    *,
    steps: int = QAT_STEPS,
) -> dict[str, Any]:
    """Pre-ternary FP distill — same kind as ultra_wrap ternary_accurate_path.

    ``optimise_model`` skips STE QAT when ``policy=ternary_wo``; this is the
    public-equivalent of that demo recipe, not a new golden.
    """
    student.train()
    teacher.eval()
    opt = torch.optim.Adam(student.parameters(), lr=1e-3)
    last = 0.0
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            t_out = teacher(x)
        loss = F.mse_loss(student(x), t_out)
        loss.backward()
        opt.step()
        last = float(loss.detach().item())
    student.eval()
    return {
        "steps": steps,
        "skipped": False,
        "last_loss": last,
        "kind": "fp_mse_distill_pre_ternary",
        "note": "FP distill then ternary snap — not BinaryLinear STE",
    }


def _column(
    *,
    title: str,
    size_bytes: int,
    cosine: float | None,
    latency_ms: float,
    speedup: float | None,
    status: str,
    compression: float | None = None,
    replaced: list[str] | None = None,
    extra: str = "",
) -> str:
    st = status.upper()
    if "REFUSE" in st:
        badge = f'<span style="background:#b91c1c;color:#fff;padding:2px 8px;border-radius:6px;font-weight:700">{st}</span>'
    elif st in {"OK", "BASELINE"}:
        badge = f'<span style="background:#15803d;color:#fff;padding:2px 8px;border-radius:6px;font-weight:700">{st}</span>'
    else:
        badge = f'<span style="background:#a16207;color:#fff;padding:2px 8px;border-radius:6px;font-weight:700">{st}</span>'

    cos_s = "—" if cosine is None else f"{cosine:.3f}"
    drop_note = ""
    if cosine is not None and cosine < DROP_IN and "BASELINE" not in st:
        drop_note = (
            f"<br/>Cosine {cos_s} &lt; {DROP_IN:.2f} drop-in gate — "
            "<strong>not a drop-in replacement</strong>."
        )
    pack = (
        f"{compression:.2f}× on replaced weights (uint64 **size**, not latency)"
        if compression
        else "n/a (FP32 teacher)"
    )
    spd = "1.00× (baseline)" if speedup is None else f"{speedup:.2f}× vs this host's FP32"
    layers = ", ".join(replaced) if replaced else "—"
    extra_html = f"<p>{extra}</p>" if extra else ""
    return f"""
### {title}
{badge}

| | |
|---|---|
| **Size** | `{_fmt_bytes(size_bytes)}` |
| **Pack compression** | {pack} |
| **Cosine vs FP** | **{cos_s}** |
| **E2E latency** | {latency_ms:.2f} ms ({spd}) |
| **Replaced** | `{layers}` |

{drop_note}
{extra_html}
"""


def _make_model(arch: str) -> tuple[nn.Module, torch.Tensor, str]:
    set_repro_seed(0, deterministic=True, force_cpu=True)
    if arch == "cnn":
        model = TinyCNN()
        x = torch.randn(8, 1, 16, 16)
        label = (
            "Tiny CNN: FP stem 16×16, FFN 128→512→128 (hybrid allowlist). "
            "Pedagogy shape; published CNN goldens are not invented here."
        )
        return model, x, label
    model = TinyBlock(d=512, ff=2048)
    x = torch.randn(16, 28 * 28)
    label = (
        "Tiny MLP = committed **ultra_wrap** TinyBlock (d=512, ff=2048). "
        "Live batch=16 (golden used 64) so cpu-basic stays interactive. "
        "Does not re-run wrap_demo hidden=4096."
    )
    return model, x, label


def measure_paradox(arch: str) -> dict[str, Any]:
    """Run FP32 / binary packed / ternary+QAT on one tiny model. CPU only."""
    if arch not in {"mlp", "cnn"}:
        arch = "mlp"
    fp, x, shape_note = _make_model(arch)
    fp.eval()

    hw = detect_hardware()
    auto = recommend_wrap_policy(None, hw)
    native = bool(native_kernel_available())

    t_fp = _time_ms(lambda: fp(x))
    fp_bytes = _total_bytes(None, fp)

    bin_cfg = OptimiseConfig(
        policy="hybrid_ffn",
        mode="binary_xnor",
        min_in_features=64,
        qat_steps=0,
        force=False,
        drop_in_threshold=DROP_IN,
    )
    bin_res = optimise_model(fp, x, bin_cfg, teacher=fp)
    t_bin = _time_ms(lambda: bin_res.model(x))
    bin_eff = (bin_res.payload.get("effectiveness") or {}) if bin_res.payload else {}

    student = copy.deepcopy(fp)
    qat_info = _fp_mse_distill(student, fp, x, steps=QAT_STEPS)
    ter_cfg = OptimiseConfig(
        policy="ternary_wo",
        mode="ternary_weight_only",
        min_in_features=64,
        qat_steps=0,
        force=False,
        drop_in_threshold=DROP_IN,
    )
    ter_res = optimise_model(student, x, ter_cfg, teacher=fp)
    t_ter = _time_ms(lambda: ter_res.model(x))
    ter_eff = (ter_res.payload.get("effectiveness") or {}) if ter_res.payload else {}

    def pack_col(res, latency: float, qat: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = res.payload or {}
        eff = payload.get("effectiveness") or {}
        cos = eff.get("cosine")
        return {
            "size_bytes": _total_bytes(payload, res.model),
            "compression": payload.get("compression_replaced_weights"),
            "cosine": None if cos is None else float(cos),
            "latency_ms": float(latency),
            "speedup": (t_fp / latency) if latency else None,
            "status": str(payload.get("status") or "UNKNOWN"),
            "drop_in_ok": payload.get("drop_in_ok"),
            "replaced": list(payload.get("replaced") or []),
            "policy": payload.get("policy"),
            "mode": payload.get("mode"),
            "native_kernel": payload.get("native_kernel"),
            "qat": qat,
        }

    return {
        "bnn_lab": BNN_VERSION,
        "arch": arch,
        "shape_note": shape_note,
        "native_kernel": native,
        "cuda": bool(hw.has_cuda),
        "auto_recommendation": {
            "policy": auto.policy,
            "mode": auto.mode,
            "reason": auto.reason,
        },
        "fp32": {
            "size_bytes": fp_bytes,
            "compression": None,
            "cosine": 1.0,
            "latency_ms": float(t_fp),
            "speedup": 1.0,
            "status": "BASELINE",
            "drop_in_ok": True,
            "replaced": [],
        },
        "binary": pack_col(bin_res, t_bin),
        "ternary": pack_col(ter_res, t_ter, qat_info),
        "binary_effectiveness": dict(bin_eff),
        "ternary_effectiveness": dict(ter_eff),
    }


def format_run(data: dict[str, Any]) -> tuple[str, str, str, str]:
    fp = data["fp32"]
    bn = data["binary"]
    tr = data["ternary"]
    col_fp = _column(
        title="FP32",
        size_bytes=int(fp["size_bytes"]),
        cosine=fp.get("cosine"),
        latency_ms=float(fp["latency_ms"]),
        speedup=1.0,
        status=str(fp["status"]),
        extra="Unwrapped teacher. Cosine is 1 by definition.",
    )
    col_bin = _column(
        title="Binary packed",
        size_bytes=int(bn["size_bytes"]),
        cosine=bn.get("cosine"),
        latency_ms=float(bn["latency_ms"]),
        speedup=bn.get("speedup"),
        status=str(bn["status"]),
        compression=bn.get("compression"),
        replaced=list(bn.get("replaced") or []),
        extra=(
            f"Live `optimise_model(policy=hybrid_ffn, mode=binary_xnor)`. "
            f"Native XNOR kernel: **{bn.get('native_kernel')}**."
        ),
    )
    col_ter = _column(
        title="Ternary + QAT",
        size_bytes=int(tr["size_bytes"]),
        cosine=tr.get("cosine"),
        latency_ms=float(tr["latency_ms"]),
        speedup=tr.get("speedup"),
        status=str(tr["status"]),
        compression=tr.get("compression"),
        replaced=list(tr.get("replaced") or []),
        extra=(
            f"Short FP distill ({QAT_STEPS} steps) then `policy=ternary_wo`. "
            "Ternary pack is ~16× size; e2e often &lt; 1× vs FP (no ternary GEMM)."
        ),
    )
    auto = data["auto_recommendation"]
    notes = f"""
{THESIS}

**This run** (`bnn-lab=={data["bnn_lab"]}`, arch=`{data["arch"]}`): {data["shape_note"]}
Native kernel loaded: **{data["native_kernel"]}**. CUDA: **{data["cuda"]}** (must stay false here).
On this host, `policy=auto` would pick **{auto["policy"]}** / **{auto["mode"]}** — {auto["reason"]}.
The binary *column* still forces `binary_xnor` so the paradox is visible even if auto would skip it.

### Published goldens (not this live run)
- {GOLDEN_AUTO}
- {GOLDEN_LEGACY}
- {GOLDEN_TERNARY}

Wall-clock is **this Space**, not the golden machine. Compression **32×** is pack math on replaced FFN weights.
"""
    return col_fp, col_bin, col_ter, notes


def run_paradox(arch_label: str) -> tuple[str, str, str, str]:
    arch = "cnn" if "cnn" in arch_label.lower() else "mlp"
    try:
        return format_run(measure_paradox(arch))
    except Exception as exc:  # noqa: BLE001 — surface any library failure in the UI
        err = (
            f"### Run failed\n\n```\n{type(exc).__name__}: {exc}\n```\n\n"
            f"{THESIS}"
        )
        return err, err, err, err


def _build_demo():
    import gradio as gr

    css = """
    #col-container { max-width: 1100px; margin: 0 auto; }
    .dark .gradio-container { color: var(--body-text-color); }
    """
    with (
        gr.Blocks(theme=gr.themes.Soft(), css=css, title="bnn-lab wrap paradox") as blocks,
        gr.Column(elem_id="col-container"),
    ):
        gr.Markdown(
            f"""
# bnn-lab — wrap paradox
{THESIS}

Three columns, one tiny model, **live** `optimise_model`. CPU only.
"""
        )
        with gr.Row():
            arch = gr.Radio(
                choices=[
                    "Tiny MLP (ultra_wrap d=512/ff=2048)",
                    "Tiny CNN (FP stem + FFN wrap)",
                ],
                value="Tiny MLP (ultra_wrap d=512/ff=2048)",
                label="Architecture (committed-class MLP, or a tiny CNN)",
            )
            run_btn = gr.Button("Run live wrap", variant="primary")
        with gr.Row():
            out_fp = gr.Markdown(label="FP32")
            out_bin = gr.Markdown(label="Binary packed")
            out_ter = gr.Markdown(label="Ternary + QAT")
        out_notes = gr.Markdown(
            f"""
Click **Run live wrap**. Expect ~10–30 s on cpu-basic (wrap + {QAT_STEPS} distill steps + timing).

### Published goldens (labels)
- {GOLDEN_AUTO}
- {GOLDEN_LEGACY}
- {GOLDEN_TERNARY}

Repo: [Binary-Neural-Networks](https://github.com/KanakMalpani/Binary-Neural-Networks) · pin `bnn-lab==1.0.0`
"""
        )
        run_btn.click(
            fn=run_paradox,
            inputs=[arch],
            outputs=[out_fp, out_bin, out_ter, out_notes],
        )
    return blocks


try:
    demo = _build_demo()
except ImportError:
    demo = None


if __name__ == "__main__":
    if demo is None:
        import json

        print(json.dumps(measure_paradox("mlp"), indent=2, default=str))
    else:
        demo.queue(max_size=4).launch()
