# third_party — BitNet pin policy

**No microsoft/BitNet submodule lives here.**

BitNet recursively vendors a llama.cpp fork (`isHuangXin/llama.cpp`). Checking
that tree into this lab would dominate clone size and fight the thesis that
**LLM serve is delegated**, not reimplemented.

Pinned SHAs/tags: [`scripts/bridges/llamacpp_bitnet_pins.json`](../scripts/bridges/llamacpp_bitnet_pins.json).  
Human recipe: [`docs/23_BITNET_CPP_BRIDGE.md`](../docs/23_BITNET_CPP_BRIDGE.md).

If a future acceptance bar ever requires an in-tree checkout, prefer a
**sparse/shallow** pin + `workflow_dispatch`-only CI — never a default CI gate.
