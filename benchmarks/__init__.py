"""Measurable benchmark writers for the adaptive 2.5D LiDAR mapping core.

Run from the repository root::

    python -m benchmarks.bench_frame     # per-frame latency table
    python -m benchmarks.bench_modes     # uniform vs distance vs adaptive
    python -m benchmarks.bench_budget    # compute-budget sweep

Artifacts are written to ``results/`` as JSON + CSV, with ``label: Measured``.
All timings are wall-clock measurements from the machine that ran them.
"""