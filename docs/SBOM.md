# SBOM & supply chain (W8.T06 / W10.T05)

## Lightweight (default)

```bat
python scripts/generate_sbom.py --out sbom.json
```

Emits a CycloneDX-shaped JSON from `pip freeze` + `bnn` version. Attached to
GitHub Releases for `v0.3.0+`.

## Full CycloneDX (optional)

```bat
pip install cyclonedx-bom
cyclonedx-py environment -o sbom-cyclonedx.json --of JSON
```

## Dependency audit

```bat
pip install pip-audit
pip-audit
```

CI job `supply-chain` runs soft `pip-audit` + SBOM smoke (does not fail the
matrix until findings are triaged).

## Related

- `SECURITY.md` — vulnerability reporting
- `LICENSE` — MIT
- Codec loads use `weights_only` / path guards (`bnn.paths`)
