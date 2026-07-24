# BitNet → bitnet.cpp bridge

For **ternary / 1.58-bit LLMs**, production CPU inference is **bitnet.cpp** (Microsoft), not classic BNN XNOR.

## Steps

1. Obtain a native BitNet checkpoint (HF BitNet cards) **or** plan BitDistill / gradual-λ QAT — do not absmean-PTQ a chat Llama and ship.
2. Follow upstream bitnet.cpp build instructions for your OS.
3. Convert / load the checkpoint per bitnet.cpp docs.
4. Measure tokens/s and energy vs FP/GGUF baselines on the same machine.

## This repo’s role

- `TernaryLinear` + `ternary_pack` = **pedagogy / size**
- Speed for LLMs = **bitnet.cpp** (or custom Hexagon kernels on Snapdragon)

## Related

- `docs/12_WRAPPER_AND_EXISTING_MODELS.md`
- `docs/20_NPU_VENDOR_CLOSURE.md` (no stock HTP ternary)
- `bnn recommend --goal cpu-llm`
