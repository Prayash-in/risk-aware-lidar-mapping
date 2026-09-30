"""End-to-end pipeline: the public API of this repository.

    AdaptiveLidarMapper(config).process_frame(points) -> FrameResult

Stages, in order:

    1. preprocessing      range filter, ego-frame convention (+x forward)
    2. semantics          RandLA-Net (if a checkpoint exists) or dataset GT
    3. dynamics           temporal change detection -> per-point dynamic score
    4. tracking           detection, association, Kalman filters
    5. prediction         resolution bubbles + constant-velocity corridors
    6. risk controller    importance factors, hysteresis, target cell sizes
    7. 2.5D projection    adaptive quadtree leaves + per-cell statistics
    8. budget             deterministic merging when over the compute budget

No frontend, no HTTP, no visualization: `process_frame` accepts a `PointCloud`,
an (N, 3)/(N, 4) array or a list of `LiDARPoint`, and returns a `FrameResult`
with cells, tracks, measured metrics and diagnostics.

The same instance can process the *identical* frame under three policies
(`mode="uniform" | "distance" | "adaptive"`), which is what the comparisons in
`examples/` and `benchmarks/` rely on.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from adaptive_lidar.config import PipelineConfig, load_config
from adaptive_lidar.controller import BudgetManager, ResolutionController
from adaptive_lidar.data.base import PointCloud, PointCloudSource
from adaptive_lidar.mapping.projection_2_5d import (
    distance_size_field,
    project_2_5d,
)
from adaptive_lidar.mapping.map_cell import MapCell
from adaptive_lidar.perception import DynamicScorer, SemanticLabeler
from adaptive_lidar.prediction.trajectory import PredictionConfig
from adaptive_lidar.tracking import ObjectTracker
from adaptive_lidar.types import (
    DYNAMIC_CLASSES,
    FrameResult,
    LiDARPoint,
    Metrics,
    TrackedObject,
    resolution_tier,
)

MODES = ("uniform", "distance", "adaptive")


def coerce_points(points) -> PointCloud:
    """Accept the shapes a caller is likely to have and return a `PointCloud`.

    Supported: `PointCloud`, (N, 3) or (N, 4) array-like, list of `LiDARPoint`.
    """
    if isinstance(points, PointCloud):
        return points
    if isinstance(points, list) and points and isinstance(points[0], LiDARPoint):
        n = len(points)
        xyz = np.empty((n, 3), dtype=np.float32)
        inten = np.ones(n, dtype=np.float32)
        sem = np.zeros(n, dtype=np.int32)
        conf = np.ones(n, dtype=np.float32)
        ts = 0.0
        for i, p in enumerate(points):
            xyz[i] = (p.x, p.y, p.z)
            inten[i] = p.intensity
            sem[i] = p.semantic_class
            conf[i] = p.semantic_confidence
            ts = p.timestamp
        return PointCloud(xyz=xyz, intensity=inten, semantic=sem, confidence=conf,
                          timestamp=ts, semantic_source="unknown")
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] not in (3, 4):
        raise ValueError("points must be (N, 3), (N, 4) or a list of LiDARPoint")
    xyz = arr[:, :3]
    inten = (arr[:, 3] if arr.shape[1] == 4
             else np.ones(arr.shape[0], dtype=np.float32))
    return PointCloud(xyz=xyz, intensity=inten, semantic=np.zeros(arr.shape[0], np.int32),
                      confidence=np.ones(arr.shape[0], dtype=np.float32),
                      semantic_source="unknown")


@dataclass
class _FrameState:
    """Pipeline state that must persist between frames."""

    last_timestamp: float | None = None
    frame_index: int = 0


class AdaptiveLidarMapper:
    """Risk-adaptive 2.5D mapping. One instance per map."""

    def __init__(self, config: PipelineConfig | None = None,
                 config_path: str | None = None, overrides: dict | None = None,
                 use_model: bool = False, use_gt_labels: bool = False,
                 collect_cells: bool = True):
        self.config = config or load_config(config_path, overrides)
        cfg = self.config
        if cfg.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {cfg.mode!r}")

        q = cfg.quadtree
        self.extent_m = float(q.extent_m)
        self.max_depth = int(q.max_depth)
        self.min_depth = max(0, int(np.log2(self.extent_m / max(q.max_cell_size, 1e-6))))
        self.collect_cells = bool(collect_cells)
        self.use_model = bool(use_model)
        self.use_gt_labels = bool(use_gt_labels)

        # --- perception
        self.semantic_labeler = SemanticLabeler(
            use_model=use_model, checkpoint=cfg.randlanet.checkpoint,
            device=cfg.randlanet.device, voxel_size=cfg.randlanet.voxel_size,
            num_classes=cfg.randlanet.num_classes,
            prefer_ground_truth=use_gt_labels or not use_model,
            semantic_importance=cfg.controller.semantic_importance)
        self.dynamics = DynamicScorer(
            cell_size=0.5, height_tol=0.15, z_tol=0.30, decay_frames=4)

        # --- tracking / prediction
        t = cfg.tracker
        self.prediction_config = PredictionConfig(
            r_min=dict(t.r_min), kappa=float(t.kappa), horizon=float(t.horizon),
            pred_horizon=float(t.pred_horizon), pred_samples=int(t.pred_samples))
        self.tracker = ObjectTracker(
            extent_m=self.extent_m, config=self.prediction_config,
            gate_m=float(t.gate_m), coast_frames=int(t.coast_frames),
            use_gt_labels=use_gt_labels)

        # --- control
        c = cfg.controller
        self.controller = ResolutionController(
            extent_m=self.extent_m, min_cell_size=q.min_cell_size,
            max_cell_size=q.max_cell_size, max_depth=self.max_depth,
            min_depth=self.min_depth, split_threshold=c.split_threshold,
            merge_threshold=c.merge_threshold, split_frames=c.split_frames,
            merge_frames=c.merge_frames, ema_alpha=c.ema_alpha,
            ref_depth=c.ref_depth,
            weights={"w_r": c.w_r, "w_d": c.w_d, "w_s": c.w_s,
                     "w_b": c.w_b, "w_p": c.w_p},
            range_scale=c.range_scale, halo_radius_m=c.halo_radius_m,
            halo_value=c.halo_value, semantic_importance=c.semantic_importance,
            use_range=c.use_range, use_semantic=c.use_semantic,
            use_dynamic=c.use_dynamic, use_bubble=c.use_bubble,
            use_prediction=c.use_prediction)
        self.budget = BudgetManager(
            budget_pct=cfg.budget.pct, min_depth=self.min_depth,
            footprint_min_depth=max(self.min_depth, self.min_depth + 1),
            hot_rate=cfg.budget.hot_rate, warm_rate=cfg.budget.warm_rate,
            cold_rate=cfg.budget.cold_rate,
            split_threshold=c.split_threshold, merge_threshold=c.merge_threshold)
        self.distance_field = distance_size_field(
            cfg.lod.s_min, cfg.lod.s_max, cfg.lod.lam)
        self._state = _FrameState()

    # ------------------------------------------------------------------ reset
    def reset(self) -> None:
        self.controller.reset()
        self.tracker.reset()
        self.dynamics.reset()
        self._state = _FrameState()

    # ---------------------------------------------------------------- process
    def process_frame(self, points, timestamp: float | None = None) -> FrameResult:
        """Process one scan and return the adaptive 2.5D map for that frame."""
        t_start = time.perf_counter()
        cloud = coerce_points(points)
        if timestamp is not None:
            cloud.timestamp = float(timestamp)

        # 1. preprocessing
        if self.config.lidar.max_range_m > 0:
            cloud = cloud.filter_range(float(self.config.lidar.max_range_m))

        frame_id = self._state.frame_index
        dt = self._dt(cloud.timestamp)

        # 2. semantics (provenance preserved)
        sem = self.semantic_labeler.label(cloud)
        labels = sem.labels

        # 3. dynamics
        dyn_score = self.dynamics.update(
            cloud.xyz, frame_id, cloud.timestamp,
            self.semantic_labeler.dynamic_labels(labels))

        # 4./5. tracking + prediction (bubbles, corridors)
        tracks: list[TrackedObject] = self.tracker.update(
            cloud.xyz, labels, dt, cloud.timestamp,
            gt_actors=cloud.meta.get("actors") if self.tracker.use_gt_labels else None)

        # 6. risk controller
        ctrl = self.controller.update(cloud.xyz, labels, dyn_score, tracks)

        # 7. 2.5D projection (policy depends on the mode)
        mode = self.config.mode
        target_size = None
        target_field = None
        depth_shift = None
        if mode == "uniform":
            target_size = np.full(cloud.num_points,
                                  float(self.config.quadtree.uniform_cell_size))
        elif mode == "distance":
            target_field = self.distance_field
        else:  # adaptive
            target_size = ctrl.target_size
            depth_shift = ctrl.decision.shift

        factors = ctrl.factors.as_dict() if mode == "adaptive" else None
        projection = project_2_5d(
            cloud.xyz, labels, extent_m=self.extent_m,
            target_size=target_size, target_field=target_field,
            depth_shift=depth_shift, max_depth=self.max_depth,
            min_depth=self.min_depth, min_cell_size=self.config.quadtree.min_cell_size,
            max_cell_size=self.config.quadtree.max_cell_size,
            num_classes=self.config.randlanet.num_classes,
            importance=ctrl.factors.total if mode == "adaptive" else None,
            factors=factors,
        )
        cells = projection.cells

        # 8. compute budget (structure preserved, only coarsening)
        protected = self._protected_keys(tracks, projection)
        cells, report = self.budget.enforce(cells, self.extent_m, protected)

        latency = time.perf_counter() - t_start
        metrics = self._metrics(latency, ctrl, projection, cells, report, sem, tracks)
        result = FrameResult(
            frame_id=frame_id, timestamp=cloud.timestamp, mode=mode,
            leaves=cells if self.collect_cells else [],
            tracks=tracks, metrics=metrics,
            extras={
                "semantic_source": sem.source,
                "semantic_checkpoint": sem.checkpoint,
                "semantic_note": sem.note,
                "importance_factors": ctrl.factors.summary(),
                "budget": report.to_dict(),
                "projection": projection.stats(),
                "controller": ctrl.decision.total_ops,
                "tier_histogram": metrics_tiers(cells),
                "semantic_histogram": self.semantic_labeler.segmenter.describe_labels(
                    labels),
            },
        )
        self._state.frame_index += 1
        self._state.last_timestamp = cloud.timestamp
        return result

    # ---------------------------------------------------------------- helpers
    def _dt(self, timestamp: float) -> float:
        last = self._state.last_timestamp
        if last is None:
            return 1.0 / max(1e-6, float(self.config.lidar.fps_target))
        dt = float(timestamp) - float(last)
        return dt if 1e-6 < dt < 10.0 else 1.0 / max(1e-6, float(self.config.lidar.fps_target))

    def _protected_keys(self, tracks: list[TrackedObject], projection) -> set:
        """Structural keys of every cell that overlaps a live track footprint.

        The adaptive map mixes levels (a depth-8..depth-11 leaf can hold the same
        physical point), so protecting only the reference level would never match
        a real leaf. Instead we return the whole ancestor chain of the reference
        cell, down to the coarsest level the budget may reach, and the budget
        manager holds any group that contains one of them.
        """
        if not tracks:
            return set()
        ref_depth = self.controller.hysteresis.ref_depth
        cell = self.extent_m / float(1 << ref_depth)
        half = self.extent_m / 2.0
        last = (1 << ref_depth) - 1
        floor = max(self.min_depth, self.budget.footprint_min_depth)
        out: set[tuple[int, int, int]] = set()
        for t in tracks:
            ix = int(np.clip((t.x + half) // cell, 0, last))
            iy = int(np.clip((t.y + half) // cell, 0, last))
            for depth in range(ref_depth, floor - 1, -1):
                shift = ref_depth - depth
                out.add((depth, ix >> shift, iy >> shift))
        return out

    def _metrics(self, latency: float, ctrl, projection, cells: list[MapCell],
                 report, sem, tracks) -> Metrics:
        tiers = {"high": 0, "medium": 0, "coarse": 0}
        for c in cells:
            tiers[resolution_tier(c.size)] += 1
        m = Metrics(
            fps=(1.0 / latency) if latency > 0 else 0.0,
            frame_latency=latency,
            controller_latency=ctrl.latency_s,
            # Post-budget footprint of the map that is actually returned. The
            # pre-budget projection is larger, so reporting it here would
            # misstate memory whenever the budget coarsened anything.
            memory_estimate=int(report.post_memory_bytes or
                                projection.memory_bytes()),
            total_cells=len(cells),
            high_resolution_cells=tiers["high"],
            medium_resolution_cells=tiers["medium"],
            coarse_resolution_cells=tiers["coarse"],
            split_operations=ctrl.decision.split_ops,
            merge_operations=ctrl.decision.merge_ops,
            compute_budget=report.allowed,
            compute_usage=report.post_cost,
        )
        return m

    # ------------------------------------------------------------------ batch
    def process_source(self, source: PointCloudSource, limit: int | None = None
                       ) -> list[FrameResult]:
        """Process consecutive frames of a `PointCloudSource`."""
        n = len(source) if limit is None else min(limit, len(source))
        return [self.process_frame(source.load_frame(i)) for i in range(n)]

    def compare_modes(self, points, replay: list | None = None) -> dict:
        """Run the identical frame through uniform / distance / adaptive.

        `replay` optionally re-processes earlier frames first so all three modes
        reach the frame with the same temporal state (tracking, EMA,
        change-detection memory); without it the comparison is single-frame.
        Returns one measured summary per mode — the direct evidence for
        "resolution is a resource allocated by risk, not distance".
        """
        out: dict = {}
        cloud = coerce_points(points)
        for mode in MODES:
            other = AdaptiveLidarMapper(config=_with_mode(self.config, mode),
                                        use_model=self.use_model,
                                        use_gt_labels=self.use_gt_labels)
            for prev in (replay or []):
                other.process_frame(prev)
            out[mode] = summarize(other.process_frame(cloud))
        return out

    # ----------------------------------------------------------------- status
    def status(self) -> dict:
        return {
            "mode": self.config.mode,
            "extent_m": self.extent_m,
            "max_depth": self.max_depth,
            "min_depth": self.min_depth,
            "frames_processed": self._state.frame_index,
            "semantics": self.semantic_labeler.status(),
            "controller": self.controller.status(),
            "tracker": self.tracker.status(),
            "budget": self.budget.last_report.to_dict() if self.budget.last_report else {},
            "thresholds": {
                "split": self.config.controller.split_threshold,
                "merge": self.config.controller.merge_threshold,
            },
        }


def _with_mode(config: PipelineConfig, mode: str) -> PipelineConfig:
    import copy
    cfg = copy.deepcopy(config)
    cfg.mode = mode
    return cfg


def metrics_tiers(cells: list[MapCell]) -> dict:
    hist = {"high": 0, "medium": 0, "coarse": 0}
    for c in cells:
        hist[resolution_tier(c.size)] += 1
    return hist


def summarize(result: FrameResult) -> dict:
    """Compact, JSON-ready summary of one frame (used by examples/benchmarks)."""
    m = result.metrics
    return {
        "frame_id": result.frame_id,
        "mode": result.mode,
        "cells": m.total_cells,
        "tiers": result.extras.get("tier_histogram", {}),
        "memory_bytes": m.memory_estimate,
        "latency_s": round(m.frame_latency, 6),
        "fps": round(m.fps, 2),
        "controller_latency_s": round(m.controller_latency, 6),
        "split_ops": m.split_operations,
        "merge_ops": m.merge_operations,
        "compute_usage": round(m.compute_usage, 1),
        "compute_budget": round(m.compute_budget, 1),
        "budget_feasible": result.extras.get("budget", {}).get("feasible"),
        "importance": result.extras.get("importance_factors", {}),
        "tracks": len(result.tracks),
        "semantic_source": result.extras.get("semantic_source"),
    }
