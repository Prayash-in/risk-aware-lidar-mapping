"""Command line interface: `python -m adaptive_lidar <command>`.

Commands
--------
run         process frames and print measured per-frame / aggregate metrics
compare     run one identical frame through uniform / distance / adaptive
ablate      run the four importance ablations (range, +semantic, +dynamic, +prediction)
budget      sweep the compute budget (100 / 75 / 50 / 25 %)
status      print the resolved configuration and perception capabilities
info        describe the data sources available in this environment

Every number printed is measured on this machine at run time. Nothing is
simulated, estimated or filled in.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys

import numpy as np

from adaptive_lidar.config import load_config
from adaptive_lidar.data import (
    SyntheticPointCloudSource,
    SemanticKITTIPointCloudSource,
    make_source,
)
from adaptive_lidar.pipeline import AdaptiveLidarMapper, summarize
from adaptive_lidar.perception.randlanet import is_torch_available


# --------------------------------------------------------------------- helpers
def _build_source(args):
    if args.source == "synthetic":
        return SyntheticPointCloudSource(
            scenario=args.scenario, num_frames=args.frames, stride=args.stride)
    return make_source("semantic_kitti", root=args.kitti_root,
                       sequence=args.sequence, start_frame=args.start_frame,
                       num_frames=args.frames, stride_points=args.stride_points)


def _mapper(args, overrides: dict | None = None) -> AdaptiveLidarMapper:
    return AdaptiveLidarMapper(
        config_path=args.config, overrides=overrides or {},
        use_model=args.use_model, use_gt_labels=args.gt_labels)


def _override(args) -> dict:
    out: dict = {"mode": args.mode}
    if args.budget is not None:
        out["budget"] = {"pct": args.budget}
    return out


def _aggregate(results) -> dict:
    """Aggregate measured per-frame results (no invented numbers)."""
    if not results:
        return {}
    lat = [r.metrics.frame_latency for r in results]
    ctrl = [r.metrics.controller_latency for r in results]
    cells = [r.metrics.total_cells for r in results]
    mem = [r.metrics.memory_estimate for r in results]
    usage = [r.metrics.compute_usage for r in results]
    return {
        "frames": len(results),
        "mode": results[0].mode,
        "latency_ms": {
            "mean": round(statistics.fmean(lat) * 1e3, 2),
            "min": round(min(lat) * 1e3, 2),
            "max": round(max(lat) * 1e3, 2),
            "p95": round(float(np.percentile(lat, 95)) * 1e3, 2),
        },
        "controller_latency_ms": {
            "mean": round(statistics.fmean(ctrl) * 1e3, 2),
            "p95": round(float(np.percentile(ctrl, 95)) * 1e3, 2),
        },
        "fps_mean": round(statistics.fmean([1.0 / l for l in lat]), 2),
        "cells": {"mean": round(statistics.fmean(cells), 1),
                  "min": min(cells), "max": max(cells)},
        "memory_bytes": {"mean": round(statistics.fmean(mem), 1), "max": max(mem)},
        "compute_units": {"mean": round(statistics.fmean(usage), 1),
                          "max": round(max(usage), 1)},
        "tiers_mean": _mean_tiers(results),
        "tracks_mean": round(statistics.fmean([len(r.tracks) for r in results]), 2),
        "split_ops": sum(r.metrics.split_operations for r in results),
        "merge_ops": sum(r.metrics.merge_operations for r in results),
        "budget_feasible": all(r.extras.get("budget", {}).get("feasible", True)
                               for r in results),
    }


def _mean_tiers(results) -> dict:
    if not results:
        return {}
    keys = ("high", "medium", "coarse")
    return {k: round(statistics.fmean(
        [r.extras.get("tier_histogram", {}).get(k, 0) for r in results]), 1) for k in keys}


# --------------------------------------------------------------------- commands
def cmd_run(args) -> int:
    source = _build_source(args)
    mapper = _mapper(args, _override(args))
    results = []
    for i in range(len(source)):
        res = mapper.process_frame(source.load_frame(i))
        results.append(res)
        if not args.quiet:
            print(json.dumps(summarize(res)))
    agg = _aggregate(results)
    agg["source"] = source.describe()
    print(json.dumps({"aggregate": agg}, indent=2))
    return 0


def cmd_compare(args) -> int:
    source = _build_source(args)
    mapper = _mapper(args, _override(args))
    frames = [source.load_frame(i) for i in range(min(len(source), args.frames))]
    for f in frames[:-1]:
        mapper.process_frame(f)
    out = mapper.compare_modes(frames[-1], replay=frames[:-1])
    print(json.dumps(out, indent=2))
    return 0


def cmd_ablate(args) -> int:
    source = _build_source(args)
    frames = [source.load_frame(i) for i in range(min(len(source), args.frames))]
    variants = [
        ("range_only", {"use_range": True, "use_semantic": False, "use_dynamic": False,
                        "use_bubble": False, "use_prediction": False}),
        ("range_semantic", {"use_range": True, "use_semantic": True, "use_dynamic": False,
                            "use_bubble": False, "use_prediction": False}),
        ("plus_dynamic", {"use_range": True, "use_semantic": True, "use_dynamic": True,
                          "use_bubble": False, "use_prediction": False}),
        ("plus_prediction", {"use_range": True, "use_semantic": True, "use_dynamic": True,
                             "use_bubble": True, "use_prediction": True}),
    ]
    out = {}
    for name, flags in variants:
        cfg = _override(args)
        cfg["controller"] = flags
        mapper = _mapper(args, cfg)
        results = [mapper.process_frame(f) for f in frames]
        row = _aggregate(results)
        row["enabled_factors"] = flags
        out[name] = row
    print(json.dumps({"ablation": out, "source": source.describe()}, indent=2))
    return 0


def cmd_budget(args) -> int:
    source = _build_source(args)
    out = {}
    for pct in (100.0, 75.0, 50.0, 25.0):
        cfg = _override(args)
        cfg["budget"] = {"pct": pct}
        mapper = _mapper(args, cfg)
        results = [mapper.process_frame(source.load_frame(i))
                   for i in range(min(len(source), args.frames))]
        row = _aggregate(results)
        row["budget"] = results[-1].extras.get("budget", {})
        out[f"{pct:g}%"] = row
    print(json.dumps({"budget_sweep": out}, indent=2))
    return 0


def cmd_status(args) -> int:
    mapper = _mapper(args, _override(args))
    print(json.dumps(mapper.status(), indent=2, default=str))
    return 0


def cmd_info(args) -> int:
    info = {
        "torch_available": is_torch_available(),
        "randlanet_note": (
            "no checkpoint is shipped; pass --use-model with a checkpoint at "
            "models/randlanet/checkpoint.pth to run real inference"
        ),
        "kitti_root_from_env": __import__("os").environ.get("KITTI_ROOT"),
        "sources": {},
    }
    try:
        src = SyntheticPointCloudSource("urban_crossing", num_frames=3)
        info["sources"]["synthetic"] = src.describe()
    except Exception as exc:  # pragma: no cover
        info["sources"]["synthetic"] = {"error": str(exc)}
    try:
        k = SemanticKITTIPointCloudSource(num_frames=2)
        info["sources"]["semantic_kitti"] = k.describe()
    except Exception as exc:
        info["sources"]["semantic_kitti"] = {
            "available": False,
            "error": str(exc).splitlines()[0],
            "hint": "set KITTI_ROOT to an existing SemanticKITTI directory",
        }
    print(json.dumps(info, indent=2))
    return 0


# ------------------------------------------------------------------- argparse
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="adaptive_lidar",
        description="Adaptive variable-resolution 2.5D LiDAR mapping (SIH26053)")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
        sp.add_argument("--source", choices=("synthetic", "kitti"), default="synthetic")
        sp.add_argument("--scenario", default="urban_crossing")
        sp.add_argument("--frames", type=int, default=20)
        sp.add_argument("--stride", type=int, default=1, help="sim ticks per frame")
        sp.add_argument("--kitti-root", default=None)
        sp.add_argument("--sequence", default="08")
        sp.add_argument("--start-frame", type=int, default=0)
        sp.add_argument("--stride-points", type=int, default=6000)
        sp.add_argument("--mode", choices=("uniform", "distance", "adaptive"),
                        default="adaptive")
        sp.add_argument("--budget", type=float, default=None,
                        help="compute budget %% (100, 75, 50, 25)")
        sp.add_argument("--use-model", action="store_true",
                        help="use RandLA-Net (requires torch + checkpoint)")
        sp.add_argument("--gt-labels", action="store_true",
                        help="use dataset ground-truth labels for track classes")
        sp.add_argument("--quiet", action="store_true")
        return sp

    common(sub.add_parser("run", help="process frames and report measured metrics")).set_defaults(
        func=cmd_run)
    common(sub.add_parser("compare", help="identical frame, three resolution policies")).set_defaults(
        func=cmd_compare)
    common(sub.add_parser("ablate", help="importance factor ablation")).set_defaults(
        func=cmd_ablate)
    common(sub.add_parser("budget", help="compute budget sweep")).set_defaults(
        func=cmd_budget)
    common(sub.add_parser("status", help="resolved configuration + capabilities")).set_defaults(
        func=cmd_status)
    sub.add_parser("info", help="data sources available in this environment").set_defaults(
        func=cmd_info)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
