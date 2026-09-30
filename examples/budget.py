"""Example 4: compute-budget sweep (deterministic coarsening, no deletion).

Usage::

    python -m examples.budget --frames 8

At 100% the budget lets the adaptive projection stand; at 75/50/25% the
budget manager only *coarsens* — it never deletes a cell's information, and it
always leaves a protected footprint around tracked objects. Feasibility and
actual usage are reported for every preset.
"""

from __future__ import annotations

import argparse

from adaptive_lidar import AdaptiveLidarMapper, load_config
from adaptive_lidar.data import SyntheticPointCloudSource
from examples._util import aggregate, print_json

PRESETS = (100.0, 75.0, 50.0, 25.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="compute budget sweep")
    parser.add_argument("--config", default=None)
    parser.add_argument("--scenario", default="urban_crossing")
    parser.add_argument("--frames", type=int, default=8)
    args = parser.parse_args()

    source = SyntheticPointCloudSource(scenario=args.scenario, num_frames=args.frames)
    frames = [source.load_frame(i) for i in range(len(source))]
    out = {}
    for pct in PRESETS:
        cfg = load_config(args.config, {"mode": "adaptive", "budget": {"pct": pct}})
        mapper = AdaptiveLidarMapper(config=cfg)
        results = [mapper.process_frame(f) for f in frames]
        row = aggregate(results)
        row["budget"] = results[-1].extras.get("budget", {})
        out[f"{pct:g}%"] = row
    print_json({"budget_sweep": out, "source": source.describe()})


if __name__ == "__main__":  # pragma: no cover
    main()