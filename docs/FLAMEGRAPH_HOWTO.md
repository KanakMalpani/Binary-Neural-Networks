# Flamegraph / profiling howto (W13.T02)

Use this when diagnosing **why** packed GEMM is fast or slow. Dual-metric:
`bnn profile` already splits pack / GEMM / overhead vs FP32.

## Built-in

```bat
bnn profile
```

Emits pack_weight / pack_act / gemm / e2e / torch_fp32 timings. Prefer this
before external tools.

## py-spy (recommended, optional)

```bat
pip install py-spy
py-spy record -o profile.svg -- python -m bnn.cli profile
```

Open `profile.svg` in a browser. Look for time outside XNOR-popcount (Python
pack loops, torch autograd leftovers, OpenMP oversubscription).

## Windows ETW / Linux perf (advanced)

- Linux: `perf record -g -- python …` then `perf script | stackcollapse | flamegraph.pl`
- Windows: Visual Studio Diagnostic Hub or WPA — only if py-spy insufficient

## Rules

- Do not invent new bench shapes for flamegraphs sold as goldens
- Always pair SVG with wall-clock numbers from `bnn profile` / Pareto JSON
- Thread count matters — document `BNN_NUM_THREADS` / OMP
