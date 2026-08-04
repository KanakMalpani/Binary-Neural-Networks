# Torch upper-bound policy (W14.T03)

**Pin:** `torch>=2.1,<2.13` in [`pyproject.toml`](../pyproject.toml) and
[`constraints.txt`](../constraints.txt).

## Why an upper bound

1. **Portability CI** (linux-arm64, macos-arm64, macos-x86_64) must resolve the
   *same* constraints band. macOS x86_64 historically lags newer torch wheels;
   an unbounded `torch>=2.1` silently picks a version the Intel runner cannot
   install and turns Scorecard/CI red for unrelated PRs.
2. **NumPy ABI:** older macOS torch builds require NumPy 1.x
   (`numpy>=1.24,<2` in constraints). Jumping torch majors without re-checking
   that pairing breaks `torch.from_numpy` with “Numpy is not available”.
3. **Goldens are conclusion-stable, not float-identical.** A silent torch major
   bump can change SDPA / matmul kernels enough to flake soft latency budgets
   even when thesis gates (32× pack, err=0) still pass.

## How to raise the ceiling

1. Bump `<2.N` in **both** `pyproject.toml` and `constraints.txt`.
2. Run the portability matrix (or wait for CI `portability` + `linux-py-matrix`).
3. Spot-check `bnn profile --batch 8 --in-features 256 --out-features 256` soft
   budgets and `bnn repro`.
4. Note the bump in `CHANGELOG.md` (integrator applies ROADMAP checkbox).

Dependabot must **not** auto-open unbounded torch major bumps — see
[`.github/dependabot.yml`](../.github/dependabot.yml) (torch/numpy ignored).

## Related

- OS × arch matrix: [`COMPATIBILITY_MATRIX.md`](COMPATIBILITY_MATRIX.md)
- Optional HF / torchao probes: [`OPTIONAL_EXTRAS_MATRIX.md`](OPTIONAL_EXTRAS_MATRIX.md)
