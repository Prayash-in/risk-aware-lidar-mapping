"""Example 2: one identical frame under uniform / distance / adaptive policy.

Usage::

    python -m examples.compare_modes --frames 12

The three modes process the *same* input frame with the same temporal state
(tracking, EMA, change-detection memory), so the cell counts directly compare
how each policy spends resolution on the same scene.
"""

from __future__ import annotations

import argparse

from adaptive_lidar import AdaptiveLidarMapper, load_config
from adaptive_lidar.data import SyntheticPointCloudSource
from examples._util import print_json


def main() -> None:
    parser = argparse.ArgumentParser(description="compare the three resolution policies")
    parser.add_argument("--config", default=None)
    parser.add_argument("--scenario", default="urban_crossing")
    parser.add_argument("--frames", type=int, default=12)
    args = parser.parse_args()

    cfg = load_config(args.config, {"mode": "adaptive"})
    source = SyntheticPointCloudSource(scenario=args.scenario, num_frames=args.frames)
    frames = [source.load_frame(i) for i in range(len(source))]

    mapper = AdaptiveLidarMapper(config=cfg)
    for f in frames[:-1]:
        mapper.process_frame(f)
    print(f"identical frame {frames[-1].frame_id}, replayed {len(frames) - 1} prior frames")
    print_json(mapper.compare_modes(frames[-1], replay=frames[:-1]))


if __name__ == "__main__":  # pragma: no cover
    main()