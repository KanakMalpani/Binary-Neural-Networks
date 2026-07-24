"""Quick identity check script (no pytest required)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bnn.math import (  # noqa: E402
    effectiveness_report,
    pack_unpack_roundtrip,
    prove_identity_sample,
)


def main() -> int:
    shapes = [1, 63, 64, 65, 128, 257, 1024]
    results = []
    for n in shapes:
        r = prove_identity_sample(n, seed=n)
        rt = pack_unpack_roundtrip(
            __import__("numpy").where(
                __import__("numpy").random.default_rng(n).integers(0, 2, n) == 0,
                1.0,
                -1.0,
            )
        )
        results.append(
            {
                "n": n,
                "identity_ok": r["ok"],
                "dot": r["dot_pm1"],
                "roundtrip_ok": rt["ok"],
                "pad_ok": rt["pad_bits_zero"],
            }
        )
    report = effectiveness_report(k=4096, m=4096)
    out = {"identities": results, "effectiveness": report, "pass": all(x["identity_ok"] and x["roundtrip_ok"] for x in results)}
    print(json.dumps(out, indent=2))
    print("MATH_IDENTITY: PASS" if out["pass"] else "MATH_IDENTITY: FAIL")
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
