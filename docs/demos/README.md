# Demo assets (W9.T10)

Short recorded demos for strangers who prefer watching over reading
[`GUIDE_E2E.md`](../GUIDE_E2E.md).

## Asciinema — optimiser quick path

File: [`optimise_quickstart.cast`](optimise_quickstart.cast) (asciinema v2).

Play locally (optional `pip install asciinema`):

```bash
asciinema play docs/demos/optimise_quickstart.cast
```

Or paste into [asciinema.org](https://asciinema.org) → Upload.

Commands shown (honest, no GPU 32× claims):

1. `bnn --version`
2. `bnn profile --batch 8 --in-features 256 --out-features 256 --reps 3 --warmup 1`
3. Pointer to `bnn optimise` / `bnn repro`

## Regenerating

Prefer a real capture on a clean shell:

```bash
asciinema rec docs/demos/optimise_quickstart.cast
# … run the three commands …
# Ctrl-D
```

The committed cast is a **pedagogy** recording (fixed timings) so CI does not
depend on asciinema being installed. Re-record before marketing screenshots.
