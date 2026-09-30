"""Example 3: importance-factor ablation on one synthetic scenario.

Usage::

    python -m examples.ablation --frames 10

Four factor sets are run under adaptive mode on the same frames:

* range_only        each factor off except range
* range_semantic    + semantic importance
* plus_dynamic      + temporal change detection
* plus_prediction   + bubbles and prediction corridors (what the thesis adds)

This is the direct evidence that dynamic/bubble/prediction factors move
resolution onto moving and risk-relevant cells instead of just remote ones.
"""

from __future__ import annotations

import argparse

from adaptive_lidar import AdaptiveLidarMapper, load_config
from adaptive_lidar.data import SyntheticPointCloudSource
from examples._util import aggregate, print_json

VARIANTS = [
    ("range_only", {"use_range": True, "use_semantic": False, "use_dynamic": False,
                    "use_bubble": False, "use_prediction": False}),
    ("range_semantic", {"use_range": True, "use_semantic": True, "use_dynamic": False,
                        "use_bubble": False, "use_prediction": False}),
    ("plus_dynamic", {"use_range": True, "use_semantic": True, "use_dynamic": True,
                      "use_bubble": False, "use_prediction": False}),
    ("plus_prediction", {"use_range": True, "use_semantic": True, "use_dynamic": True,
                         "use_bubble": True, "use_prediction": True}),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="importance factor ablation")
    parser.add_argument("--config", default=None)
    parser.add_argument("--scenario", default="urban_crossing")
    parser.add_argument("--frames", type=int, default=10)
    args = parser.parse_args()

    source = SyntheticPointCloudSource(scenario=args.scenario, num_frames=args.frames)
    frames = [source.load_frame(i) for i in range(len(source))]
    out = {}
    for name, flags in VARIANTS:
        cfg = load_config(args.config, {"mode": "adaptive", "controller": flags})
        mapper = AdaptiveLidarMapper(config=cfg)
        results = [mapper.process_frame(f) for f in frames]
        row = aggregate(results)
        row["enabled_factors"] = flags
        out[name] = row
    print_json({"ablation": out, "source": source.describe()})


if __name__ == "__main__":  # pragma: no cover
    main()