#!/usr/bin/env python3
"""Validate knowledge_graph/bnn_kg.json — no dangling edges, no orphans, thesis nodes present."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bnn.kg import clear_kg_cache, load_kg, validate_graph  # noqa: E402


def main() -> int:
    clear_kg_cache()
    g = load_kg()
    errs = validate_graph(g)
    meta = g.get("meta", {})
    print(
        f"KG: {meta.get('node_count', len(g['nodes']))} nodes, "
        f"{meta.get('edge_count', len(g['edges']))} edges"
    )
    if errs:
        print("FAIL:")
        for e in errs:
            print(" -", e)
        return 1
    print("KG: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
