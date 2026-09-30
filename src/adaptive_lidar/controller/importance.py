"""Importance factors (§7) and the weighted risk score.

    I(p) = w_r·f_r + w_d·f_d + w_s·f_s + w_b·f_b + w_p·f_p ,  each f ∈ [0,1]

* f_r  range        exp(-d / range_scale)                (§6 family)
* f_d  dynamics     temporal change detection + class dynamism + actor halo
* f_s  semantics    per-class importance table (prototype heuristic)
* f_b  bubbles      1 inside a tracked object's resolution bubble, linear
                    falloff to 0 at 1.5·r — bubbles change the map structure
* f_p  prediction   1 inside a predicted-path corridor, linear +2 m falloff

Ablation (§24): `use_*` flags switch individual factors off and the remaining
weights are renormalized, so an ablated run still scores in [0, 1] and the
comparison stays meaningful.

This module is pure: no state, no side effects, easy to unit test.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from adaptive_lidar.types import ImportanceScore, TrackedObject

#: Prototype heuristic semantic importance (not a trained/optimized table).
DEFAULT_SEMANTIC_IMPORTANCE: dict[int, float] = {
    0: 0.6,   # unknown
    1: 1.0,   # road (drivable)
    2: 0.8,   # curb
    3: 0.3,   # vegetation
    4: 0.5,   # building
    5: 0.7,   # static obstacle
    6: 0.9,   # vehicle
    7: 0.9,   # pedestrian
    8: 0.9,   # cyclist
}

FACTOR_KEYS = ("w_r", "w_d", "w_s", "w_b", "w_p")


@dataclass
class ImportanceFactors:
    """Per-point factor values, all in [0, 1]."""

    range: np.ndarray
    dynamic: np.ndarray
    semantic: np.ndarray
    bubble: np.ndarray
    prediction: np.ndarray
    total: np.ndarray
    weights: dict = field(default_factory=dict)

    def as_score(self, index: int) -> ImportanceScore:
        return ImportanceScore(
            range=float(self.range[index]), semantic=float(self.semantic[index]),
            dynamic=float(self.dynamic[index]), bubble=float(self.bubble[index]),
            prediction=float(self.prediction[index]), total=float(self.total[index]),
        )

    def as_dict(self) -> dict:
        """Per-point factor arrays keyed by MapCell score name.

        Used to hand the individual factors to the projection so every cell can
        report why it is fine (§12 introspection), not just the composite.
        """
        return {"range": self.range, "semantic": self.semantic,
                "dynamic": self.dynamic, "bubble": self.bubble,
                "prediction": self.prediction}

    def summary(self) -> dict:
        """Mean of every factor (used in results and ablations)."""
        return {
            "range": round(float(np.mean(self.range)), 4) if self.range.size else 0.0,
            "dynamic": round(float(np.mean(self.dynamic)), 4) if self.dynamic.size else 0.0,
            "semantic": round(float(np.mean(self.semantic)), 4) if self.semantic.size else 0.0,
            "bubble": round(float(np.mean(self.bubble)), 4) if self.bubble.size else 0.0,
            "prediction": round(float(np.mean(self.prediction)), 4) if self.prediction.size else 0.0,
            "total": round(float(np.mean(self.total)), 4) if self.total.size else 0.0,
        }


class ImportanceScorer:
    """Computes the five factors and their weighted combination."""

    def __init__(self, weights: dict | None = None, range_scale: float = 30.0,
                 halo_radius_m: float = 3.0, halo_value: float = 0.7,
                 semantic_importance: dict | None = None,
                 corridor_falloff_m: float = 2.0,
                 use_range: bool = True, use_semantic: bool = True,
                 use_dynamic: bool = True, use_bubble: bool = True,
                 use_prediction: bool = True):
        self.raw_weights = {k: float((weights or {}).get(k, v)) for k, v in
                            zip(FACTOR_KEYS, (0.20, 0.15, 0.25, 0.25, 0.15))}
        self.use = {
            "w_r": bool(use_range), "w_d": bool(use_dynamic), "w_s": bool(use_semantic),
            "w_b": bool(use_bubble), "w_p": bool(use_prediction),
        }
        self.range_scale = float(range_scale)
        self.halo_radius_m = float(halo_radius_m)
        self.halo_value = float(halo_value)
        self.corridor_falloff_m = float(corridor_falloff_m)
        self.semantic_importance = dict(semantic_importance or DEFAULT_SEMANTIC_IMPORTANCE)
        self.weights = self._normalize()

    def _normalize(self) -> dict:
        """Renormalize over the enabled factors so I stays in [0, 1]."""
        active = [k for k in FACTOR_KEYS if self.use[k]]
        total = sum(max(0.0, self.raw_weights[k]) for k in active)
        if total <= 0:
            raise ValueError("at least one enabled weight must be > 0")
        return {k: (max(0.0, self.raw_weights[k]) / total if k in active else 0.0)
                for k in FACTOR_KEYS}

    def set_weights(self, weights: dict) -> dict:
        for k, v in (weights or {}).items():
            if k in self.raw_weights:
                self.raw_weights[k] = min(1.0, max(0.0, float(v)))
        self.weights = self._normalize()
        return self.weights

    # ---------------------------------------------------------------- factors
    def range_factor(self, xyz: np.ndarray) -> np.ndarray:
        d = np.hypot(xyz[:, 0], xyz[:, 1])
        return np.exp(-d / max(self.range_scale, 1e-6)).astype(np.float32)

    def dynamic_factor(self, xyz: np.ndarray, dynamic_score: np.ndarray | None,
                       tracks: list[TrackedObject] | None = None) -> np.ndarray:
        """Change-detection score, raised to halo_value near live tracks."""
        if dynamic_score is None:
            f = np.zeros(xyz.shape[0], dtype=np.float32)
        else:
            f = np.clip(np.asarray(dynamic_score, dtype=np.float32).reshape(-1), 0, 1)
        if tracks:
            x, y = xyz[:, 0].astype(np.float64), xyz[:, 1].astype(np.float64)
            r2 = self.halo_radius_m ** 2
            for t in tracks:
                d2 = (x - t.x) ** 2 + (y - t.y) ** 2
                halo = d2 <= r2
                f = np.where(halo, np.maximum(f, self.halo_value), f)
        return f.astype(np.float32)

    def semantic_factor(self, labels: np.ndarray) -> np.ndarray:
        table = self.semantic_importance
        out = np.array([table.get(int(v), 0.5) for v in np.asarray(labels).reshape(-1)],
                       dtype=np.float32)
        return out

    def bubble_factor(self, xyz: np.ndarray, tracks: list[TrackedObject] | None) -> np.ndarray:
        """1 inside a resolution bubble, linear falloff to 0 at 1.5·radius."""
        f = np.zeros(xyz.shape[0], dtype=np.float32)
        if not tracks:
            return f
        x, y = xyz[:, 0].astype(np.float64), xyz[:, 1].astype(np.float64)
        for t in tracks:
            r = float(t.bubble_radius)
            if r <= 0:
                continue
            d = np.hypot(x - t.x, y - t.y)
            fall = np.clip((1.5 * r - d) / (0.5 * r), 0.0, 1.0)
            f = np.maximum(f, fall.astype(np.float32))
        return f

    def prediction_factor(self, xyz: np.ndarray,
                           tracks: list[TrackedObject] | None) -> np.ndarray:
        """1 on the predicted corridor, linear falloff of `corridor_falloff_m`."""
        f = np.zeros(xyz.shape[0], dtype=np.float32)
        if not tracks:
            return f
        x, y = xyz[:, 0].astype(np.float64), xyz[:, 1].astype(np.float64)
        fo = max(self.corridor_falloff_m, 1e-6)
        for t in tracks:
            path = np.asarray(t.predicted_path, dtype=np.float64).reshape(-1, 2)
            if path.shape[0] == 0:
                continue
            r = float(np.clip(0.4 * t.bubble_radius, 1.0, 4.0)) if t.bubble_radius > 0 else 1.0
            d2min = np.full(x.shape[0], np.inf)
            for p in path:
                np.minimum(d2min, (x - p[0]) ** 2 + (y - p[1]) ** 2, out=d2min)
            fall = np.clip((r + fo - np.sqrt(d2min)) / fo, 0.0, 1.0)
            f = np.maximum(f, fall.astype(np.float32))
        return f

    # ------------------------------------------------------------- combination
    def score(self, xyz: np.ndarray, labels: np.ndarray,
              dynamic_score: np.ndarray | None = None,
              tracks: list[TrackedObject] | None = None) -> ImportanceFactors:
        f_r = self.range_factor(xyz)
        f_d = self.dynamic_factor(xyz, dynamic_score, tracks)
        f_s = self.semantic_factor(labels)
        f_b = self.bubble_factor(xyz, tracks)
        f_p = self.prediction_factor(xyz, tracks)
        w = self.weights
        total = (w["w_r"] * f_r + w["w_d"] * f_d + w["w_s"] * f_s
                 + w["w_b"] * f_b + w["w_p"] * f_p).astype(np.float32)
        return ImportanceFactors(f_r, f_d, f_s, f_b, f_p, total, dict(w))

    def describe(self) -> dict:
        return {
            "raw_weights": dict(self.raw_weights),
            "weights": dict(self.weights),
            "enabled": {k: v for k, v in self.use.items()},
            "range_scale_m": self.range_scale,
            "halo_radius_m": self.halo_radius_m,
            "halo_value": self.halo_value,
            "corridor_falloff_m": self.corridor_falloff_m,
            "semantic_table": {str(k): v for k, v in self.semantic_importance.items()},
        }
