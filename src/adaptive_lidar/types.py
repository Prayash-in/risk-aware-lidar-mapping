"""Typed data structures for adaptive-2-5d-lidar-mapping.

All pipeline stages exchange these types; the visualizer (out of scope for
this repository) consumes them through the Python API or CLI — never the
other way around. Units are SI (meters, seconds) unless noted.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Semantic classes (shared across synthetic labels, dataset GT and model output)
# ---------------------------------------------------------------------------
UNKNOWN = 0
ROAD = 1
CURB = 2
VEGETATION = 3
BUILDING = 4
STATIC_OBSTACLE = 5
VEHICLE = 6
PEDESTRIAN = 7
CYCLIST = 8

CLASS_NAMES: dict[int, str] = {
    UNKNOWN: "unknown",
    ROAD: "road",
    CURB: "curb",
    VEGETATION: "vegetation",
    BUILDING: "building",
    STATIC_OBSTACLE: "static_obstacle",
    VEHICLE: "vehicle",
    PEDESTRIAN: "pedestrian",
    CYCLIST: "cyclist",
}

DYNAMIC_CLASSES = frozenset({VEHICLE, PEDESTRIAN, CYCLIST})


# ---------------------------------------------------------------------------
# Resolution tiers (shared definition: size -> tier name)
# ---------------------------------------------------------------------------
HI_MAX_SIZE = 0.10
MED_MAX_SIZE = 0.25


def resolution_tier(size_m: float) -> str:
    """Map a physical cell size to its resolution tier (high/medium/coarse)."""
    if size_m <= HI_MAX_SIZE:
        return "high"
    if size_m <= MED_MAX_SIZE:
        return "medium"
    return "coarse"


@dataclass
class LiDARPoint:
    """One LiDAR return in the ego frame (+x forward, z up)."""

    x: float
    y: float
    z: float
    intensity: float = 1.0
    semantic_class: int = UNKNOWN
    semantic_confidence: float = 1.0
    timestamp: float = 0.0


@dataclass
class TrackedObject:
    """Lightweight tracked dynamic object.

    `label` holds the object class ('vehicle', 'pedestrian', 'cyclist',
    'unknown') — `class` is a Python keyword, hence the name.
    """

    object_id: int
    label: str
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    confidence: float = 0.5
    timestamp: float = 0.0
    bubble_radius: float = 0.0
    predicted_path: list[list[float]] = field(default_factory=list)
    #: Where `label` came from: 'ground_truth' (dataset labels) or 'semantics'
    #: (model output). Never conflate the two when reporting results.
    label_source: str = "semantics"


@dataclass
class ImportanceScore:
    """Per-cell (or per-point) multi-factor importance, all factors in [0, 1]."""

    range: float = 0.0
    semantic: float = 0.0
    dynamic: float = 0.0
    bubble: float = 0.0
    prediction: float = 0.0
    total: float = 0.0


@dataclass
class Metrics:
    """Measured (never fabricated) per-frame pipeline metrics."""

    fps: float = 0.0
    frame_latency: float = 0.0       # wall seconds, end to end
    controller_latency: float = 0.0  # wall seconds, importance + hysteresis
    memory_estimate: int = 0         # bytes, measured array storage + node struct
    total_cells: int = 0
    high_resolution_cells: int = 0
    medium_resolution_cells: int = 0
    coarse_resolution_cells: int = 0
    split_operations: int = 0
    merge_operations: int = 0
    compute_budget: float = 0.0      # allowed cost units (0 = uncapped)
    compute_usage: float = 0.0       # actual post-enforcement cost units


@dataclass
class FrameResult:
    """Everything `AdaptiveLidarMapper.process_frame` produces for one scan."""

    frame_id: int
    timestamp: float
    mode: str
    leaves: list = field(default_factory=list)  # list[MapCell], see mapping.map_cell
    tracks: list[TrackedObject] = field(default_factory=list)
    metrics: Metrics = field(default_factory=Metrics)
    extras: dict = field(default_factory=dict)
