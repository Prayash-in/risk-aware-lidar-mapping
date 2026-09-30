"""Dynamic object detection + tracking + resolution bubbles.

No neural detector and no ground truth are required:

1. **Change detection** on consecutive scans, aggregated on the reference grid.
   Per cell: occupancy toggle (count >= `occ_min_count`), height change and
   dominant-class change combine into a score; a count gate keeps sparse
   background resampling from drowning the signal (movers are dense).
2. **Clustering**: 8-connected flagged cells become object candidates
   (centroid, mean height, point count, dominant dynamic class).
3. **Tracking**: gated association with constant-velocity Kalman filters.
   Tracks coast a few frames without detections, then die.
4. **Bubbles + rollout**: `prediction.trajectory` turns each track into a
   resolution bubble and a predicted corridor, which the risk controller
   consumes as the bubble and prediction factors.

Optional ground-truth assist
----------------------------
`update(..., gt_actors=...)` may label a candidate from a dataset actor
(`PointCloud.meta["actors"]`). This is **simulation ground truth** and is
disabled by default (`use_gt_labels=False`); when it is enabled the result is
tagged with `label_source="ground_truth"` so no downstream report can present
it as an independent estimate.
"""

from __future__ import annotations

import time
from bisect import bisect_left
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from adaptive_lidar.prediction.trajectory import PredictionConfig, constant_velocity_path
from adaptive_lidar.tracking.kalman import KalmanCV
from adaptive_lidar.types import DYNAMIC_CLASSES, TrackedObject

CLASS_NAME_BY_ID = {6: "vehicle", 7: "pedestrian", 8: "cyclist"}

#: detection / tracking constants (documented defaults, overridable in config)
CHANGE_DEPTH = 9
OCC_MIN_COUNT = 3
HEIGHT_NORM_M = 0.5
CHANGE_THRESHOLD = 0.35
LABEL_RADIUS_M = 4.0
MIN_CELLS_NEW = 3
CONFIRM_BASE_M = 1.5


@dataclass
class ObjectCandidate:
    """One clustered detection before association."""

    x: float
    y: float
    z: float = 0.0
    label: str = "unknown"
    points: int = 0
    cells: int = 0
    dyn_points: int = 0
    label_source: str = "semantics"


@dataclass
class Track:
    """A tracked object with its filter, bubble and rollout."""

    track_id: int
    label: str
    kf: KalmanCV = field(default_factory=KalmanCV)
    age: int = 0
    updates: int = 0
    coast: int = 0
    confidence: float = 0.5
    z: float = 0.8
    label_source: str = "semantics"

    def to_object(self, config: PredictionConfig, timestamp: float) -> TrackedObject:
        px, py = self.kf.pos
        vx, vy = self.kf.vel
        return TrackedObject(
            object_id=self.track_id, label=self.label, x=px, y=py, vx=vx, vy=vy,
            confidence=float(self.confidence), timestamp=float(timestamp),
            bubble_radius=config.bubble_radius(self.label, self.kf.speed),
            predicted_path=constant_velocity_path(
                px, py, vx, vy, config.pred_horizon, config.pred_samples),
            label_source=self.label_source,
        )


class ObjectTracker:
    """Stateful multi-object tracker: `update()` per frame, `reset()` on restart."""

    def __init__(self, extent_m: float = 100.0, config: PredictionConfig | None = None,
                 gate_m: float = 8.0, coast_frames: int = 5,
                 change_depth: int = CHANGE_DEPTH, occ_min_count: int = OCC_MIN_COUNT,
                 height_norm_m: float = HEIGHT_NORM_M,
                 change_threshold: float = CHANGE_THRESHOLD,
                 min_cells_new: int = MIN_CELLS_NEW, use_gt_labels: bool = False,
                 label_radius_m: float = LABEL_RADIUS_M):
        self.extent_m = float(extent_m)
        self.half = self.extent_m / 2.0
        self.config = config or PredictionConfig()
        self.gate_m = float(gate_m)
        self.coast_frames = int(coast_frames)
        self.change_depth = int(change_depth)
        self.occ_min_count = int(occ_min_count)
        self.height_norm_m = float(height_norm_m)
        self.change_threshold = float(change_threshold)
        self.min_cells_new = int(min_cells_new)
        self.use_gt_labels = bool(use_gt_labels)
        self.label_radius_m = float(label_radius_m)

        self.tracks: list[Track] = []
        self._next_id = 1
        self._prev: dict[int, tuple[int, float, int]] = {}
        self._prev_unconfirmed: list[tuple[float, float]] = []
        self.latency_s = 0.0
        self.change_cells = 0
        self.candidates = 0
        self._sorted_index: tuple = (None, [], [], [])
        self._index_len = -1

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        self.tracks.clear()
        self._next_id = 1
        self._prev.clear()
        self._prev_unconfirmed.clear()
        self.latency_s = 0.0
        self.change_cells = 0
        self.candidates = 0
        self._sorted_index = (None, [], [], [])
        self._index_len = -1

    # ------------------------------------------------------------- utilities
    def _cell_size(self) -> float:
        return self.extent_m / float(1 << self.change_depth)

    def _cell_keys(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        cell = self._cell_size()
        last = (1 << self.change_depth) - 1
        ix = np.clip(np.floor((x + self.half) / cell).astype(np.int64), 0, last)
        iy = np.clip(np.floor((y + self.half) / cell).astype(np.int64), 0, last)
        return (ix << 32) | iy

    def _aggregate(self, xyz: np.ndarray, semantic: np.ndarray):
        """Per-cell (count, sum_z, dominant class) plus the sort/group arrays."""
        x = xyz[:, 0].astype(np.float64)
        y = xyz[:, 1].astype(np.float64)
        keys = self._cell_keys(x, y)
        order = np.argsort(keys, kind="stable")
        skeys = keys[order]
        uniq, first, counts = np.unique(skeys, return_index=True, return_counts=True)
        n, L = skeys.shape[0], uniq.shape[0]
        g = np.searchsorted(first, np.arange(n, dtype=np.int64), side="right") - 1
        z = xyz[order][:, 2].astype(np.float64)
        sem = np.clip(semantic[order].astype(np.int64), 0, 8)
        sum_z = np.bincount(g, weights=z, minlength=L)
        # One bincount instead of np.add.at (an order of magnitude faster) for
        # the (group, class) vote table.
        flat = np.bincount(g * 9 + sem, minlength=L * 9).reshape(L, 9)
        votes = flat
        table = {int(k): (int(c), float(s), int(np.argmax(v)))
                 for k, s, c, v in zip(uniq.tolist(), sum_z.tolist(),
                                       counts.tolist(), votes)}
        self._sorted_index = (order, uniq.tolist(), first.tolist(), counts.tolist())
        self._index_len = int(xyz.shape[0])
        return table, order, g, uniq, first, counts

    def _build_index(self, x: np.ndarray, y: np.ndarray):
        keys = self._cell_keys(x, y)
        order = np.argsort(keys, kind="stable")
        uniq, first, counts = np.unique(keys[order], return_index=True, return_counts=True)
        return order, uniq.tolist(), first.tolist(), counts.tolist()

    # ------------------------------------------------------------- detection
    def detect(self, cur: dict[int, tuple[int, float, int]]) -> dict[int, float]:
        """Flagged ref-cells with their change score."""
        flagged: dict[int, float] = {}
        for key, (c, s, d) in cur.items():
            p = self._prev.get(key)
            pc, ps, pd = p if p is not None else (0, 0.0, -1)
            occ_now = c >= self.occ_min_count
            occ_prev = pc >= self.occ_min_count
            occ_change = 1.0 if occ_now != occ_prev else 0.0
            if occ_now and occ_prev:
                dh = abs(s / c - ps / pc)
                sem_change = 1.0 if d != pd else 0.0
            else:
                dh = sem_change = 0.0
            score = (0.4 * occ_change
                     + 0.4 * min(1.0, dh / self.height_norm_m)
                     + 0.2 * sem_change)
            if score >= self.change_threshold and max(c, pc) >= self.occ_min_count:
                flagged[key] = score
        return flagged

    def _cluster(self, flagged: dict[int, float]) -> list[list[int]]:
        """8-connected components over (ix, iy) of the flagged cells."""
        pts = {(int(k) >> 32, int(k) & 0xFFFFFFFF) for k in flagged}
        clusters: list[list[int]] = []
        seen: set[tuple[int, int]] = set()
        for seed in sorted(pts):
            if seed in seen:
                continue
            comp: list[int] = []
            dq = deque([seed])
            seen.add(seed)
            while dq:
                cx, cy = dq.popleft()
                comp.append((cx << 32) | cy)
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        nb = (cx + dx, cy + dy)
                        if nb in pts and nb not in seen:
                            seen.add(nb)
                            dq.append(nb)
            clusters.append(comp)
        return clusters

    def _candidates(self, xyz: np.ndarray, semantic: np.ndarray, clusters: list[list[int]],
                    gt_actors: list[dict] | None) -> list[ObjectCandidate]:
        x = xyz[:, 0].astype(np.float64)
        y = xyz[:, 1].astype(np.float64)
        z = xyz[:, 2].astype(np.float64)
        sem_all = np.asarray(semantic).astype(np.int64)
        order, ulist, ufirst, ucounts = self._sorted_index
        if order is None or self._index_len != xyz.shape[0]:  # pragma: no cover
            order, ulist, ufirst, ucounts = self._build_index(x, y)

        out: list[ObjectCandidate] = []
        for comp in clusters:
            parts = []
            for k in comp:
                j = bisect_left(ulist, k)
                if j < len(ulist) and ulist[j] == k:
                    parts.append(order[ufirst[j]:ufirst[j] + ucounts[j]])
            if not parts:
                continue
            ii = np.concatenate(parts)
            if ii.shape[0] == 0:
                continue
            dyn = np.isin(sem_all[ii], tuple(DYNAMIC_CLASSES))
            label, source = "unknown", "semantics"
            if dyn.any():
                votes = np.bincount(sem_all[ii][dyn], minlength=9)
                label = CLASS_NAME_BY_ID.get(int(np.argmax(votes)), "unknown")
            cx, cy = float(np.mean(x[ii])), float(np.mean(y[ii]))
            if self.use_gt_labels and gt_actors:
                best, best_d = None, self.label_radius_m
                for a in gt_actors:
                    d = abs(cx - float(a["x"])) + abs(cy - float(a["y"]))
                    if d < best_d:
                        best, best_d = a, d
                if best is not None:
                    label, source = str(best.get("label", "unknown")), "ground_truth"
            out.append(ObjectCandidate(
                x=cx, y=cy, z=float(np.mean(z[ii])), label=label,
                points=int(ii.shape[0]), cells=len(comp),
                dyn_points=int(np.count_nonzero(dyn)), label_source=source))
        return out

    # ----------------------------------------------------------------- update
    def update(self, xyz: np.ndarray, semantic: np.ndarray, dt: float,
               timestamp: float = 0.0, gt_actors: list[dict] | None = None
               ) -> list[TrackedObject]:
        """One tracker tick; returns the live tracks (with bubbles and rollouts)."""
        t0 = time.perf_counter()
        xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
        dt = float(dt)

        for tr in self.tracks:
            tr.kf.predict(dt)

        table, _, _, _, _, _ = self._aggregate(xyz, semantic)
        flagged = self.detect(table)
        self.change_cells = len(flagged)
        clusters = self._cluster(flagged)
        self.candidates = len(clusters)
        cands = self._candidates(xyz, semantic, clusters, gt_actors)

        # --- association: each track claims candidates inside the gate
        claimed = [False] * len(cands)
        for tr in self.tracks:
            px, py = tr.kf.pos
            near = [i for i, c in enumerate(cands)
                    if not claimed[i]
                    and abs(c.x - px) + abs(c.y - py) <= self.gate_m
                    and (c.label == tr.label or "unknown" in (c.label, tr.label))]
            if near:
                for i in near:
                    claimed[i] = True
                w = sum(cands[i].points for i in near)
                mx = sum(cands[i].x * cands[i].points for i in near) / w
                my = sum(cands[i].y * cands[i].points for i in near) / w
                mz = sum(cands[i].z * cands[i].points for i in near) / w
                tr.kf.update(mx, my)
                tr.z = 0.7 * tr.z + 0.3 * mz
                tr.updates += 1
                tr.coast = 0
                tr.confidence = min(1.0, 0.5 + 0.1 * tr.updates)
                if tr.label == "unknown":
                    labs = [cands[i].label for i in near if cands[i].label != "unknown"]
                    if labs:
                        tr.label, tr.label_source = labs[0], cands[near[0]].label_source
            else:
                tr.coast += 1
                tr.confidence = max(0.1, tr.confidence * 0.8)
            tr.age += 1

        # --- death, then birth of confirmed unclaimed candidates
        self.tracks = [t for t in self.tracks if t.coast <= self.coast_frames]
        positions = [tr.kf.pos for tr in self.tracks]
        confirm_r = max(CONFIRM_BASE_M, 12.0 * dt)
        for i, c in enumerate(cands):
            if claimed[i] or c.dyn_points == 0 or c.cells < self.min_cells_new:
                continue
            if any(abs(c.x - px) + abs(c.y - py) <= self.gate_m for px, py in positions):
                continue
            if not any(abs(c.x - px) + abs(c.y - py) <= confirm_r
                       for px, py in self._prev_unconfirmed):
                continue
            tr = Track(track_id=self._next_id, label=c.label, z=c.z,
                       label_source=c.label_source)
            self._next_id += 1
            tr.kf.init(c.x, c.y)
            tr.updates, tr.age = 1, 1
            self.tracks.append(tr)
            positions.append((c.x, c.y))
            claimed[i] = True
        self._prev_unconfirmed = [(c.x, c.y) for c in cands]

        self._prev = table
        self.latency_s = time.perf_counter() - t0
        return [tr.to_object(self.config, timestamp) for tr in self.tracks]

    # ----------------------------------------------------------------- status
    def status(self) -> dict:
        return {
            "tracks": len(self.tracks),
            "change_cells": self.change_cells,
            "candidates": self.candidates,
            "latency_ms": round(self.latency_s * 1000.0, 3),
            "gate_m": self.gate_m,
            "coast_frames": self.coast_frames,
            "change_depth": self.change_depth,
            "use_gt_labels": self.use_gt_labels,
            "prediction": self.config.to_dict(),
            "next_id": self._next_id,
        }
