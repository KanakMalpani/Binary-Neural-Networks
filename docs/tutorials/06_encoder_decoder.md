# Tutorial 06 — Encoder / Decoder + `.bnnpack`

**Master guide:** [`../GUIDE_E2E.md`](../GUIDE_E2E.md) · **Prev:** [05](05_audio.md) · **Next:** [07](07_OPTIMISER_QUICKSTART.md)

Short path from STE Encoder–Decoder to a portable packed weight file.

## 1. Train reverse seq2seq

```bat
bnn train-seq2seq --task seq2seq --steps 80 --out results\seq2seq_encoder_decoder.json
```

Architecture: FP self/cross-attn + LayerNorm; binary (or ternary) FFN via STE.

## 2. Encode weights

```bat
bnn encode --source mlp --hidden 256 --out results\demo.bnnpack
bnn decode --pack results\demo.bnnpack
```

Expect `DECODE: PASS` and compression **32×**.

## 3. Wrap a tiny Transformer

```bat
bnn wrap-transformer --qat-steps 40
```

Writes `results/tiny_transformer_wrap.json` (agreement, compression, native flag).

## Next

- Optimiser product path: [07_OPTIMISER_QUICKSTART.md](07_OPTIMISER_QUICKSTART.md)
- [`../36_ENCODER_DECODER_AND_NEXT.md`](../36_ENCODER_DECODER_AND_NEXT.md)
- [`02_wrap_linear.md`](02_wrap_linear.md)
