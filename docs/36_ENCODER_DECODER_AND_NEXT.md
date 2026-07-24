# Encoder / Decoder + weight codec + next-lane demos

What shipped in this pass, how to run it, and how it ties to the thesis
(packed CPU kernels; FP attn/softmax/norms; no fake GPU 32×).

## Why Encoder + Decoder

Two complementary “encode/decode” stories:

1. **Architecture** — `BinaryTransformerEncoder` + `BinaryTransformerDecoder`
   (seq2seq reverse task) and `BinaryAutoEncoder` show binary/ternary **FFN**
   with FP attention/LayerNorm, trainable via STE.
2. **Weight codec** — `bnn encode` / `bnn decode` turn Linear/BinaryLinear (or
   already-packed modules) into portable **`.bnnpack`** artifacts and back to
   `PackedBinaryXNORLinear` with **GEMM err = 0** and **32×** weight compression.

Together they make the wrap layer *shipable* (file format + runnable packed
modules), not just an in-process demo.

## Packages

| Path | Role |
|------|------|
| `bnn/seq/` | Encoder, Decoder, `BinarySeq2Seq`, `BinaryAutoEncoder` |
| `bnn/codec/` | `.bnnpack` encode/decode API |
| `bnn/profile.py` | Pack / GEMM / overhead breakdown |
| `scripts/train_seq2seq.py` | Reverse task + AE smoke |
| `scripts/tiny_transformer_wrap_demo.py` | hybrid_ffn + QAT + metrics |
| `scripts/bridges/` | torchao INT4 + llama.cpp/bitnet.cpp recipes |

## Commands

```bat
:: Weight codec round-trip
bnn encode --source random --in-features 512 --out-features 512 --out results\demo.bnnpack
bnn decode --pack results\demo.bnnpack

:: Encoder–Decoder reverse task (+ AE)
bnn train-seq2seq --task both --steps 80 --out results\seq2seq_encoder_decoder.json

:: Tiny Transformer wrap lane
bnn wrap-transformer --qat-steps 40 --out results\tiny_transformer_wrap.json

:: Profile pack vs GEMM vs FP32
bnn profile --batch 64 --in-features 4096 --out-features 4096 --out results\profile.json

:: Production bridges (no heavy deps pulled)
python scripts\bridges\torchao_int4_recipe.py --probe
python scripts\bridges\llamacpp_bitnet_recipe.py
```

Python API sketch:

```python
from bnn.seq import BinarySeq2Seq, make_reverse_batch
from bnn.codec import encode_linear_state, decode_to_packed_linear, roundtrip_gemm_err

model = BinarySeq2Seq(vocab=16, dim=64, depth=2)
src, tgt_in, tgt = make_reverse_batch(8, 8, 16, seed=0)
logits = model(src, tgt_in)

import torch
blob = encode_linear_state(torch.randn(256, 256))
mod = decode_to_packed_linear(blob)
assert roundtrip_gemm_err(torch.randn(256, 256))["max_abs_err"] == 0.0
```

## Expected results (smoke, this machine)

| Check | Result |
|-------|--------|
| Codec compression (aligned) | **32.00×** |
| Codec / decode GEMM vs ±1 FP | **err = 0** |
| Seq2seq reverse (80 STE steps) | **eval token acc = 1.0** |
| Tiny Transformer wrap | FFN×4 replaced; cosine **~0.91**; top1 **~0.97**; 32× on replaced |
| `bnn repro` | `REPRO: PASS` |

Wall-clock profile numbers are machine-dependent; correctness gates are exact.

## Thesis lock

- Encoder/Decoder **do not** claim GPU XNOR speedups.
- Softmax / LayerNorm / attention projections stay FP in these demos.
- GPU servers → INT4/FP8 (`scripts/bridges/torchao_int4_recipe.py`).
- CPU LLMs → GGUF / bitnet.cpp (`scripts/bridges/llamacpp_bitnet_recipe.py`).

## Tests

```bat
pytest tests/test_codec.py tests/test_seq_encoder_decoder.py tests/test_profile.py -q
bnn repro
```
