"""Configuration: YAML files -> typed dataclasses with documented defaults.

All algorithm weights, thresholds and horizons live here (or in YAML that
maps here) — never hardcoded inside the algorithm modules.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore


def _repo_path(*parts: str) -> str:
    """Resolve a shipped file relative to the repository root, not the CWD.

    `src/adaptive_lidar/config.py` -> repository root is three levels up. This
    keeps `configs/` and `models/` reachable when the package is used from an
    arbitrary working directory (installed CLI, notebooks, tests).
    """
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))     # .../<repo>/src/adaptive_lidar
    return os.path.join(root, *parts)


@dataclass
class LidarConfig:
    height_m: float = 1.8
    max_range_m: float = 60.0
    fps_target: float = 10.0


@dataclass
class QuadtreeConfig:
    extent_m: float = 100.0
    min_cell_size: float = 0.05
    max_cell_size: float = 0.50
    max_depth: int = 11  # 100 / 2^11 = 0.049 m
    uniform_cell_size: float = 0.10  # baseline (§8)


@dataclass
class LodConfig:
    s_min: float = 0.05
    s_max: float = 0.50
    lam: float = 0.12  # s(r) = s_min + (s_max - s_min) * (1 - exp(-lam*r))


@dataclass
class ControllerConfig:
    w_r: float = 0.2
    w_d: float = 0.15
    w_s: float = 0.25
    w_b: float = 0.25
    w_p: float = 0.15
    range_scale: float = 30.0  # f_r = exp(-d / range_scale)
    split_threshold: float = 0.60
    merge_threshold: float = 0.30
    split_frames: int = 2      # persistence required to SPLIT (§16)
    merge_frames: int = 3      # persistence required to MERGE (§16)
    ema_alpha: float = 0.5
    ref_depth: int = 9         # hysteresis control-grid level
    halo_radius_m: float = 3.0
    halo_value: float = 0.7
    # Semantic importance table (heuristic prototype values — NOT optimal).
    semantic_importance: dict = field(default_factory=lambda: {
        0: 0.6, 1: 1.0, 2: 0.8, 3: 0.3, 4: 0.5, 5: 0.7, 6: 0.9, 7: 0.9, 8: 0.9,
    })
    # Ablation switches (§24): disable factors independently.
    use_range: bool = True
    use_semantic: bool = True
    use_dynamic: bool = True
    use_bubble: bool = True
    use_prediction: bool = True


@dataclass
class TrackerConfig:
    r_min: dict = field(default_factory=lambda: {
        "pedestrian": 2.0, "cyclist": 3.0, "vehicle": 5.0, "unknown": 2.0,
    })
    kappa: float = 1.5
    horizon: float = 2.0
    pred_horizon: float = 2.0
    pred_samples: int = 8
    gate_m: float = 8.0
    coast_frames: int = 5


@dataclass
class BudgetConfig:
    pct: float = 100.0  # 100/75/50/25 presets; enforced every frame when < 100
    hot_rate: float = 1.0
    warm_rate: float = 0.5
    cold_rate: float = 0.15


@dataclass
class RandLAConfig:
    checkpoint: str = "models/randlanet/checkpoint.pth"
    num_classes: int = 9
    voxel_size: float = 0.06
    k_neighbors: int = 16
    device: str = "cpu"


@dataclass
class DataConfig:
    kind: str = "synthetic"  # or "semantic_kitti"
    scenario: str = "urban_crossing"
    kitti_root: str = "data/semantic_kitti"
    kitti_sequence: str = "08"
    kitti_frame: int = 0


@dataclass
class PipelineConfig:
    mode: str = "adaptive"  # uniform | distance | adaptive
    lidar: LidarConfig = field(default_factory=LidarConfig)
    quadtree: QuadtreeConfig = field(default_factory=QuadtreeConfig)
    lod: LodConfig = field(default_factory=LodConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    randlanet: RandLAConfig = field(default_factory=RandLAConfig)
    data: DataConfig = field(default_factory=DataConfig)


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _build(cls, data: dict):
    """Instantiate nested dataclass `cls` from a (possibly partial) dict."""
    import dataclasses
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name in data:
            v = data[f.name]
            ftype = f.type
            if isinstance(v, dict) and isinstance(ftype, type) and dataclasses.is_dataclass(ftype):
                kwargs[f.name] = _build(ftype, v)
            else:
                kwargs[f.name] = v
    return cls(**kwargs)


def load_config(path: str | None = None, overrides: dict | None = None) -> PipelineConfig:
    """Load configs/default.yaml (+ optional file), apply overrides dict.

    `path` may be absolute or relative to the current directory. With
    `path=None` the repository's `configs/default.yaml` is resolved relative to
    this file, so an installed package (or a CLI run from any directory) still
    finds the shipped defaults; a missing default is not an error.
    """
    data: dict = {}
    files = [_repo_path("configs", "default.yaml")] if path is None else [path]
    for fp in files:
        try:
            if yaml is None:
                raise RuntimeError("pyyaml is required to load configs")
            with open(fp, "r", encoding="utf-8") as f:
                data = _merge(data, yaml.safe_load(f) or {})
        except FileNotFoundError:
            if path is not None:
                raise
    # map flat yaml sections onto nested dataclass structure
    nested: dict = {}
    section_map = {
        "lidar": LidarConfig, "quadtree": QuadtreeConfig, "lod": LodConfig,
        "controller": ControllerConfig, "tracker": TrackerConfig,
        "budget": BudgetConfig, "randlanet": RandLAConfig, "data": DataConfig,
    }
    for key, cls in section_map.items():
        if key in data:
            nested[key] = _build(cls, data[key])
    if "mode" in data:
        nested["mode"] = data["mode"]
    if overrides:
        import dataclasses
        for key, val in overrides.items():
            if key in section_map and isinstance(val, dict):
                base = nested.get(key)
                current = {f.name: getattr(base, f.name) for f in dataclasses.fields(base)} \
                    if base is not None else {}
                nested[key] = _build(section_map[key], _merge(current, val))
            else:
                nested[key] = val
    return PipelineConfig(**nested)
