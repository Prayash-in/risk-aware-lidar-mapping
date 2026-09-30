"""Example 1: drive a synthetic urban scene through the adaptive pipeline.

Usage (from the repository root)::

    python -m examples.basic_mapping --frames 15

Every number in the output is measured on this machine at run time.
"""

from __future__ import annotations

import argparse

from adaptive_lidar import AdaptiveLidarMapper, load_config
from adaptive_lidar.data import SyntheticPointCloudSource
from examples._util import aggregate, frame_rows, print_json


def main() -> None:
    parser = argparse.ArgumentParser(description="basic adaptive mapping example")
    parser.add_argument("--config", default=None)
    parser.add_argument("--scenario", default="urban_crossing")
    parser.add_argument("--frames", type=int, default=15)
    args = parser.parse_args()

    cfg = load_config(args.config, {"mode": "adaptive"})
    source = SyntheticPointCloudSource(scenario=args.scenario, num_frames=args.frames)
    mapper = AdaptiveLidarMapper(config=cfg)

    print(f"source: {source.describe()}")
    print(f"mode:   adaptive  extent_m: {mapper.extent_m}  max_depth: {mapper.max_depth}")
    results = []
    for i in range(len(source)):
        results.append(mapper.process_frame(source.load_frame(i)))

    frame_rows(results)
    summary = aggregate(results)
    summary["source"] = source.describe()
    print_json({"aggregate": summary, "semantics": mapper.status()["semantics"]})


if __name__ == "__main__":  # pragma: no cover
    main()