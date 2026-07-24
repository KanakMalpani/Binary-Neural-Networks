# HF → GGUF / llama.cpp checklist

For **normal** (non-BitNet) HF LLMs on CPU, prefer **GGUF + llama.cpp** — not this repo’s XNOR wrap.

## Steps

1. Pick a base model with a compatible license.
2. Convert with the current llama.cpp / Hugging Face GGUF conversion tooling for that model family.
3. Quantize to `Q4_K_M` (or `Q5_K_M` if quality-sensitive).
4. Serve with `llama-cli` / `llama-server`; set threads to physical cores.
5. Benchmark tokens/s at batch-1 (chat) separately from batch throughput.

## Do not

- Expect this repo’s `binary_xnor` wrap to make Llama chat-quality without QAT/distill.
- Confuse GGUF INT4 with BitNet 1.58-bit (different format + runtime).

## Related

- BitNet path: `docs/23_BITNET_CPP_BRIDGE.md`
- Decision tree: `docs/18_DECISION_TREE_AND_COMPLETE_ROADMAP.md`
- Recommend: `bnn recommend --goal cpu-llm`
