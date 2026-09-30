"""Shared helpers for the runnable examples and benchmark writers."""

from __future__ import annotations

import json
import statistics

import numpy as np

from adaptive_lidar.pipeline import summarize


def frame_rows(results):
    """Print one compact measured row per frame + an aggregate."""
    if not results:
        print("no frames processed")
        return
    header = ("frame mode cells high med coarse split merge usage ms mem_kb "
              "semantic tracks")
    print(header)
    for r in results:
        m = r.metrics
        t = r.extras.get("tier_histogram", {})
        print(
            f"{r.frame_id:>5} {r.mode:>8} {m.total_cells:>5} "
            f"{t.get('high', 0):>4} {t.get('medium', 0):>3} {t.get('coarse', 0):>6} "
            f"{m.split_operations:>5} {m.merge_operations:>5} "
            f"{m.compute_usage:>5.0f} {m.frame_latency * 1e3:>5.1f} "
            f"{m.memory_estimate / 1024.0:>6.1f} "
            f"{r.extras.get('semantic_source', ''):>7} {len(r.tracks):>6}"
        )


def aggregate(results) -> dict:
    """Measured aggregate over the frames (JSON-ready, no invented numbers)."""
    if not results:
        return {}
    lat = [r.metrics.frame_latency for r in results]
    return {
        "frames": len(results),
        "mode": results[0].mode,
        "latency_ms_mean": round(statistics.fmean(lat) * 1e3, 2),
        "latency_ms_p95": round(float(np.percentile(lat, 95)) * 1e3, 2),
        "fps_mean": round(statistics.fmean([1.0 / l for l in lat]), 2),
        "cells_mean": round(statistics.fmean(
            [r.metrics.total_cells for r in results]), 1),
        "memory_bytes_mean": round(statistics.fmean(
            [r.metrics.memory_estimate for r in results]), 1),
        "compute_units_mean": round(statistics.fmean(
            [r.metrics.compute_usage for r in results]), 1),
        "tracks_mean": round(statistics.fmean([len(r.tracks) for r in results]), 2),
    }


def print_json(payload) -> None:
    print(json.dumps(payload, indent=2, default=str))


def json_file_path(name: str) -> str:
    from pathlib import Path
    return str(Path("results") / name)


def write_results(payload, name: str, writer, sample: str) -> None:
    """Write ``results/<name>.json`` and ``results/<name>.csv`` from a writer."""
    from pathlib import Path
    Path("results").mkdir(exist_ok=True)
    with open(json_file_path(name + ".json"), "w", encoding="utf-8") as fh:
        json.dump({"sample": sample, **payload}, fh, indent=2, default=str)
    with open(json_file_path(name + ".csv"), "w", encoding="utf-8",
              newline="") as fh:
        writer(fh, payload)
    print(f"wrote results/{name}.json and results/{name}.csv")