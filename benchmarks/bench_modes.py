"""Three-policy comparison benchmark on an identical frame with shared state.

Usage::

    python -m benchmarks.bench_modes --frames 20 --trials 1

Writes ``results/bench_modes.json`` + ``results/bench_modes.csv``.
The direct evidence pinned to the thesis line: resolution is allocated by risk,
not distance. Cells / memory / compute usage are measured, so "distance policy"
being cheaper per frame is a *measured* reduction, not a claim.
"""

from __future__ import annotations

import argparse
import csv

import numpy as np

from adaptive_lidar import AdaptiveLidarMapper, load_config
from adaptive_lidar.data import SyntheticPointCloudSource
from examples._util import write_results


def main() -> None:
    parser = argparse.ArgumentParser(description="three-policy comparison benchmark")
    parser.add_argument("--config", default=None)
    parser.add_argument("--scenario", default="urban_crossing")
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--trials", type=int, default=1)
    args = parser.parse_args()

    cfg = load_config(args.config, {"mode": "adaptive"})
    rows = []
    for trial in range(1, args.trials + 1):
        source = SyntheticPointCloudSource(scenario=args.scenario, num_frames=args.frames)
        frames = [source.load_frame(i) for i in range(len(source))]
        mapper = AdaptiveLidarMapper(config=cfg)
        for f in frames[:-1]:
            mapper.process_frame(f)
        modes = mapper.compare_modes(frames[-1], replay=frames[:-1])
        rows.append({"trial": trial, "frame_id": frames[-1].frame_id,
                     "modes": modes})

    ref = rows[0]["modes"]["adaptive"]["cells"]
    summary = {}
    for mode, data in rows[0]["modes"].items():
        summary[mode] = {
            **data,
            "cells_vs_adaptive_pct": round(100.0 * data["cells"] / ref, 1)
            if ref else 0.0,
        }

    payload = {
        "label": "Measured", "scenario": args.scenario,
        "frames_processed": args.frames, "trials": args.trials,
        "modes": summary,
        "trials": rows,
    }

    def writer(fh, payload):
        w = csv.writer(fh)
        w.writerow(["mode", "cells", "high", "medium", "coarse", "memory_bytes",
                    "latency_ms", "compute_usage", "budget_feasible", "tracks"])
        for mode, d in payload["modes"].items():
            w.writerow([mode, d["cells"], d["tiers"]["high"], d["tiers"]["medium"],
                        d["tiers"]["coarse"], d["memory_bytes"], d["latency_s"] * 1e3,
                        d["compute_usage"], d["budget_feasible"], d["tracks"]])

    write_results(payload, "bench_modes", writer, args.scenario)


if __name__ == "__main__":  # pragma: no cover
    main()