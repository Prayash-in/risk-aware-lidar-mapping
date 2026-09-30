"""Adaptive variable-resolution 2.5D LiDAR mapping (SIH26053).

Research/engineering core: perception pipeline, quadtree mapping,
risk-aware resolution control, tracking, prediction and compute budget.
No frontend required — see `AdaptiveLidarMapper` in `pipeline.py`.

Typical use::

    from adaptive_lidar import AdaptiveLidarMapper
    from adaptive_lidar.data import SyntheticPointCloudSource

    mapper = AdaptiveLidarMapper("configs/default.yaml")
    for i in range(10):
        result = mapper.process_frame(source.load_frame(i))
        print(result.metrics.total_cells, result.metrics.frame_latency)
"""

from adaptive_lidar.types import (
    CLASS_NAMES,
    DYNAMIC_CLASSES,
    FrameResult,
    ImportanceScore,
    LiDARPoint,
    Metrics,
    TrackedObject,
    resolution_tier,
)
from adaptive_lidar.config import PipelineConfig, load_config
from adaptive_lidar.pipeline import AdaptiveLidarMapper, metrics_tiers, summarize

__version__ = "0.1.0"

__all__ = [
    "CLASS_NAMES",
    "DYNAMIC_CLASSES",
    "AdaptiveLidarMapper",
    "FrameResult",
    "ImportanceScore",
    "LiDARPoint",
    "Metrics",
    "PipelineConfig",
    "TrackedObject",
    "load_config",
    "metrics_tiers",
    "resolution_tier",
    "summarize",
    "__version__",
]
