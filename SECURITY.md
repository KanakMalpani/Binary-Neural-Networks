# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| `0.2.x` (main) | Yes — security fixes preferred on `main` |
| `< 0.2` | Best-effort only |

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security-sensitive reports.

1. Email or privately message the maintainer listed in [`CODEOWNERS`](.github/CODEOWNERS), **or**
2. Use GitHub **Security Advisories** → *Report a vulnerability* on
   [KanakMalpani/Binary-Neural-Networks](https://github.com/KanakMalpani/Binary-Neural-Networks)
   if enabled.

Include: affected version / commit, repro steps, impact, and (if available) a minimal PoC.

We aim to acknowledge within **7 days** and ship a fix or mitigation advisory when confirmed.

## Safe loading policy (this repo)

- Prefer `weights_only=True` (or equivalent) for `torch.load` of untrusted checkpoints.
- `.bnnpack` loads go through `bnn.codec` with path guards (`bnn.paths`).
- Do **not** commit secrets, API keys, or datasets under `data/`.
- Native kernels are **local DLLs/SOs** built from this tree — treat third-party binaries as untrusted.

## Thesis-related “attacks”

Reports that only claim “GPU 32× from `sign()`” are **thesis violations**, not security bugs —
see [`ROADMAP.md`](ROADMAP.md) §0.2. Open a `thesis-violation` issue template instead.
