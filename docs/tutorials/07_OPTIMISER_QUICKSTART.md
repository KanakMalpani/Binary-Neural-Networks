# 07 — Optimiser quickstart

**Master guide:** [`../GUIDE_E2E.md`](../GUIDE_E2E.md) · **Prev:** [06](06_encoder_decoder.md) · **Next:** [08](08_HF_OPTIMISER.md)

**Goal:** take an FP toy model → auto/hybrid wrap → versioned JSON report → optional `.bnnpack`.  
**Time:** < 10 minutes. **Thesis:** compression ≠ wall-clock; no GPU 32× from `sign()`.

## Install

```bat
pip install -e ".[dev]" -c constraints.txt
python -m bnn.kernels.compile_native
```

## One verb (CLI)

```bat
bnn optimise --policy auto --report results/optimise_report.json
```

Expect a JSON suite (primary + ternary accurate path + optional wide efficiency probe).
Primary objects use schema `bnn_optimise_report_v1`.

Useful flags:

| Flag | Meaning |
|------|---------|
| `--policy auto` | Hardware-aware binary vs ternary vs skip |
| `--qat-steps 40` | Light STE recovery on FFN names (binary path) |
| `--force` | Allow drop-in claim below cosine threshold (honest `forced: true`) |
| `--pack results/demo.bnnpack` | Also smoke-encode a toy MLP packfile |

Legacy: `bnn wrap --ultra …` still works; prefer `bnn optimise`.

## Python API

```python
import torch
import torch.nn as nn
from bnn.optimise import OptimiseConfig, optimise_model

class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(64, 64)
        self.ffn_fc1 = nn.Linear(64, 256)
        self.ffn_fc2 = nn.Linear(256, 64)
        self.lm_head = nn.Linear(64, 10)

    def forward(self, x):
        h = torch.relu(self.embed(x))
        return self.lm_head(self.ffn_fc2(torch.relu(self.ffn_fc1(h))))

model = Tiny()
x = torch.randn(8, 64)
result = optimise_model(
    model,
    x,
    OptimiseConfig(policy="hybrid_ffn", mode="binary_xnor", min_in_features=32, force=True),
)
print(result.payload["compression_replaced_weights"], result.payload["status"])
```

## Read the report honestly

- `compression_replaced_weights` ≈ **32** for aligned binary pack — **theory**.
- `e2e_latency_ms_*` / `layer_microbench.speedup_gemm_only_vs_torch` — **wall-clock**.
- `drop_in_ok` / `status: REFUSE_DROP_IN_CLAIM` — do not market as drop-in without metrics.

## Next

- Full narrative: [`../GUIDE_E2E.md`](../GUIDE_E2E.md)
- HF models: [08_HF_OPTIMISER.md](08_HF_OPTIMISER.md)
- ADR: [`docs/adr/0001_public_optimiser_api.md`](../adr/0001_public_optimiser_api.md)
- Roadmap: [`ROADMAP.md`](../../ROADMAP.md)
