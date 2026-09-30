"""Measured per-frame benchmark on a deterministic synthetic scenario.

Usage::

    python -m benchmarks.bench_frame --frames 30 --trials 1

Writes ``results/bench_frame.json`` + ``results/bench_frame.csv``.
All timings are wall-clock measurements from this machine (label: Measured).
"""

from __future__ import annotations

import argparse
import csv
import statistics

from adaptive_lidar import AdaptiveLidarMapper, load_config
from adaptive_lidar.data import SyntheticPointCloudSource
from examples._util import write_results


def main() -> None:
    parser = argparse.ArgumentParser(description="per-frame latency benchmark")
    parser.add_argument("--config", default=None)
    parser.add_argument("--scenario", default="urban_crossing")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--trials", type=int, default=1)
    args = parser.parse_args()

    cfg = load_config(args.config, {"mode": "adaptive"})
    out = {}
    for trial in range(1, args.trials + 1):
        source = SyntheticPointCloudSource(scenario=args.scenario, num_frames=args.frames)
        mapper = AdaptiveLidarMapper(config=cfg)
        rows = []
        for i in range(len(source)):
            res = mapper.process_frame(source.load_frame(i))
            rows.append({
                "frame_id": i,
                "latency_ms": round(res.metrics.frame_latency * 1e3, 3),
                "controller_ms": round(res.metrics.controller_latency * 1e3, 3),
                "cells": res.metrics.total_cells,
                "memory_bytes": res.metrics.memory_estimate,
                "compute_units": round(res.metrics.compute_usage, 1),
                "tracks": len(res.tracks),
            })
        out[f"trial_{trial}"] = rows

    lat = [r["latency_ms"] for rows in out.values() for r in rows]
    payload = {
        "label": "Measured", "scenario": args.scenario,
        "frames": args.frames, "trials": args.trials,
        "per_frame_latency_ms": {"mean": round(statistics.fmean(lat), 3),
                                 "min": round(min(lat), 3),
                                 "max": round(max(lat), 3),
                                 "p50": round(float(__import__("numpy").median(lat)), 3),
                                 "p95": round(float(__import__("numpy").percentile(lat, 95)), 3)},
        "fps_mean": round(statistics.fmean([1e3 / l for l in lat]), 2),
        "trials": out,
        "provenance": {
            "cpu": __import__("platform").processor(),
        },
    }

    def writer(fh, payload):
        w = csv.writer(fh)
        w.writerow(["trial", "frame_id", "latency_ms", "controller_ms", "cells",
                    "memory_bytes", "compute_units", "tracks"])
        for trial, rows in payload["trials"].items():
            for r in rows:
                w.writerow([trial] + [r["frame_id"], r["latency_ms"],
                                      r["controller_ms"], r["cells"],
                                      r["memory_bytes"], r["compute_units"],
                                      r["tracks"]])

    write_results(payload, "bench_frame", writer, args.scenario)


if __name__ == "__main__":  # pragma: no cover
    main()