# SBOM & supply chain (W8.T06 / W10.T05)

## Lightweight (default)

```bat
python scripts/generate_sbom.py --out sbom.json
```

Emits a CycloneDX-shaped JSON from `pip freeze` + `bnn` version. Attached to
GitHub Releases for `v0.3.0+` (current tag `v1.0.0`).

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

## Artifact attestations (W8.T07)

`wheels.yml` attaches GitHub [artifact attestations](https://docs.github.com/actions/security-guides/using-artifact-attestations)
(SLSA provenance) to every wheel and the sdist on
`KanakMalpani/Binary-Neural-Networks` (skipped on forks — no OIDC).

```bat
gh attestation verify bnn_lab-1.0.0-*.whl --repo KanakMalpani/Binary-Neural-Networks
gh attestation verify bnn_lab-1.0.0.tar.gz --repo KanakMalpani/Binary-Neural-Networks
```

Filenames use the normalised dist name `bnn_lab` (PyPI project `bnn-lab`).
Publish to PyPI is OIDC Trusted Publishing only — see
[`PYPI_PUBLISH.md`](PYPI_PUBLISH.md). First upload: `bnn-lab` 1.0.0 live.

## Related

- `SECURITY.md` — vulnerability reporting
- `LICENSE` — MIT
- Codec loads use `weights_only` / path guards (`bnn.paths`)
