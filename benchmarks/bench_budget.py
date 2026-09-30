"""Compute-budget sweep benchmark (coarsening only, never deletion).

Usage::

    python -m benchmarks.bench_budget --frames 10

Writes ``results/bench_budget.json`` + ``results/bench_budget.csv``.
Every row reports the measured post-budget cost against the allowed budget,
the number of merge operations, and whether the budget could be met.
"""

from __future__ import annotations

import argparse
import csv

from adaptive_lidar import AdaptiveLidarMapper, load_config
from adaptive_lidar.data import SyntheticPointCloudSource
from examples._util import write_results

PRESETS = (100.0, 75.0, 50.0, 25.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="compute budget sweep benchmark")
    parser.add_argument("--config", default=None)
    parser.add_argument("--scenario", default="urban_crossing")
    parser.add_argument("--frames", type=int, default=10)
    args = parser.parse_args()

    source = SyntheticPointCloudSource(scenario=args.scenario, num_frames=args.frames)
    frames = [source.load_frame(i) for i in range(len(source))]
    presets = {}
    for pct in PRESETS:
        cfg = load_config(args.config, {"mode": "adaptive", "budget": {"pct": pct}})
        mapper = AdaptiveLidarMapper(config=cfg)
        results = [mapper.process_frame(f) for f in frames]
        agg = {
            "cells_mean": round(__import__("statistics").fmean(
                [r.metrics.total_cells for r in results]), 1),
            "latency_ms_mean": round(__import__("statistics").fmean(
                [r.metrics.frame_latency for r in results]) * 1e3, 2),
            "budget_feasible": all(r.extras.get("budget", {}).get("feasible", True)
                                   for r in results),
        }
        agg.update(results[-1].extras["budget"])
        presets[f"{pct:g}%"] = agg

    payload = {"label": "Measured", "scenario": args.scenario,
               "frames": args.frames, "presets": presets}

    def writer(fh, payload):
        w = csv.writer(fh)
        w.writerow(["pct", "pre_cost", "post_cost", "allowed", "feasible",
                    "pre_leaves", "post_leaves", "post_internal",
                    "post_memory_bytes", "merged_groups", "floor_reached",
                    "cells_mean", "latency_ms_mean"])
        for name, d in payload["presets"].items():
            w.writerow([name[:name.index("%")] if "%" in name else name,
                        d["pre_cost"], d["post_cost"], d["allowed"], d["feasible"],
                        d["pre_leaves"], d["post_leaves"], d["post_internal"],
                        d["post_memory_bytes"], d["merged_groups"],
                        d["floor_reached"], d["cells_mean"], d["latency_ms_mean"]])

    write_results(payload, "bench_budget", writer, args.scenario)


if __name__ == "__main__":  # pragma: no cover
    main()