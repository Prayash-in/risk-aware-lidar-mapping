"""Resolution controller: importance factors + hysteresis -> per-point target size.

    "Resolution is a resource allocated by risk, not distance."

The controller is the only component that turns perception output into
*geometry* (which level a point is stored at). It is deliberately separated
into three testable pieces:

    importance.py     pure factor computation (no state)
    hysteresis.py     per-ref-cell EMA + persistence (state, no geometry)
    this module       orchestration, target-size mapping, measured latency

Split/merge operations are counted when a *hysteresis state transition* occurs,
which is the moment the map structure actually changes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from adaptive_lidar.controller.hysteresis import HysteresisController, HysteresisDecision
from adaptive_lidar.controller.importance import ImportanceFactors, ImportanceScorer
from adaptive_lidar.mapping.projection_2_5d import depth_to_size, size_to_depth
from adaptive_lidar.types import TrackedObject


@dataclass
class ControllerOutput:
    """Everything the pipeline needs from one controller tick."""

    factors: ImportanceFactors
    decision: HysteresisDecision
    target_size: np.ndarray
    base_depth: np.ndarray
    final_depth: np.ndarray
    latency_s: float = 0.0
    extras: dict = field(default_factory=dict)


class ResolutionController:
    """Risk-aware resolution controller (stateful, deterministic)."""

    def __init__(self, extent_m: float = 100.0,
                 min_cell_size: float = 0.05, max_cell_size: float = 0.50,
                 max_depth: int = 11, min_depth: int = 0,
                 split_threshold: float = 0.60, merge_threshold: float = 0.30,
                 split_frames: int = 2, merge_frames: int = 3,
                 ema_alpha: float = 0.5, ref_depth: int = 9,
                 weights: dict | None = None, range_scale: float = 30.0,
                 halo_radius_m: float = 3.0, halo_value: float = 0.7,
                 semantic_importance: dict | None = None,
                 use_range: bool = True, use_semantic: bool = True,
                 use_dynamic: bool = True, use_bubble: bool = True,
                 use_prediction: bool = True,
                 importance_to_size: bool = True,
                 split_threshold_importance: float = 0.60):
        self.extent_m = float(extent_m)
        self.min_cell_size = float(min_cell_size)
        self.max_cell_size = float(max_cell_size)
        self.max_depth = int(max_depth)
        self.min_depth = int(min_depth)
        self.importance_to_size = bool(importance_to_size)
        self.split_importance = float(split_threshold_importance)

        self.scorer = ImportanceScorer(
            weights=weights, range_scale=range_scale, halo_radius_m=halo_radius_m,
            halo_value=halo_value, semantic_importance=semantic_importance,
            use_range=use_range, use_semantic=use_semantic, use_dynamic=use_dynamic,
            use_bubble=use_bubble, use_prediction=use_prediction)
        self.hysteresis = HysteresisController(
            extent_m=self.extent_m, ref_depth=ref_depth,
            split_threshold=split_threshold, merge_threshold=merge_threshold,
            split_frames=split_frames, merge_frames=merge_frames,
            ema_alpha=ema_alpha, min_depth=self.min_depth, max_depth=self.max_depth)
        self.latency_s = 0.0
        self.last_factors: ImportanceFactors | None = None
        self.last_decision: HysteresisDecision | None = None

    # ------------------------------------------------------------------ reset
    def reset(self) -> None:
        self.hysteresis.reset()
        self.latency_s = 0.0
        self.last_factors = None
        self.last_decision = None

    # ----------------------------------------------------------------- update
    def update(self, xyz: np.ndarray, labels: np.ndarray,
               dynamic_score: np.ndarray | None = None,
               tracks: list[TrackedObject] | None = None) -> ControllerOutput:
        """One tick: score importance, run hysteresis, map to cell sizes."""
        t0 = time.perf_counter()
        factors = self.scorer.score(xyz, labels, dynamic_score, tracks)
        decision = self.hysteresis.update(xyz, factors.total)

        if self.importance_to_size:
            # Risk-driven base level: high importance -> deeper level.
            imp = np.clip(factors.total, 0.0, 1.0)
            target = self.max_cell_size * (1.0 - imp) + self.min_cell_size * imp
            base_depth = size_to_depth(self.extent_m, target)
        else:
            # Pure hysteresis: importance does not set a static size by itself.
            base_depth = np.full(xyz.shape[0], size_to_depth(
                self.extent_m, self.max_cell_size), dtype=np.int64)

        final_depth = np.clip(base_depth + decision.shift, self.min_depth, self.max_depth)
        target_size = depth_to_size(self.extent_m, final_depth)

        self.latency_s = time.perf_counter() - t0
        self.last_factors, self.last_decision = factors, decision
        return ControllerOutput(
            factors=factors, decision=decision, target_size=target_size,
            base_depth=base_depth, final_depth=final_depth, latency_s=self.latency_s,
        )

    # ----------------------------------------------------------------- status
    def status(self) -> dict:
        return {
            "latency_ms": round(self.latency_s * 1000.0, 3),
            "importance": self.scorer.describe(),
            "hysteresis": self.hysteresis.status(),
            "limits": {
                "min_cell_size_m": self.min_cell_size,
                "max_cell_size_m": self.max_cell_size,
                "min_depth": self.min_depth,
                "max_depth": self.max_depth,
                "importance_to_size": self.importance_to_size,
            },
            "factor_means": self.last_factors.summary() if self.last_factors else {},
        }

    def inspect(self, x: float, y: float) -> dict:
        """Explain one ref-cell (factors, EMA, candidate, thresholds)."""
        state = self.hysteresis.state_of_point(x, y)
        if self.last_factors is not None and self.last_factors.range.size:
            xy = np.array([[x, y, 0.0]], dtype=np.float32)
            f = self.last_factors
            state["factors_here"] = {
                "range": round(float(f.range_factor(xy)[0]), 3),
                "semantic": round(float(self.scorer.semantic_factor([0])[0]), 3),
            }
        return state
