# Energy bound to measured latency

- FP latency: 34.14 ms → E≈1194.7 mJ @ 35.0 W
- Binary wrap: 7.09 ms → E≈177.2 mJ @ 25.0 W
- Reduction (assumed P): **6.74×**
- Reduction (latency-only, same P): **4.82×**
- Pareto energy_proxy (FP=1): binary=0.14833426047196713
- CLOSED-BY-PROXY: Windows has no portable RAPL in stdlib; E=P*t with measured t + assumed P brackets + literature anchors. Sufficient for decision thesis.
