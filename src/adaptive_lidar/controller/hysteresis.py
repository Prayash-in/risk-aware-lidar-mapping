"""Hysteresis resolution state machine (§16).

The controller decides *how much* importance exists (see `importance.py`); this
module decides *when* a cell's resolution may change, so the map neither
flickers nor thrashes.

Design (extracted from the validated prototype, generalized):

* Decisions are taken on a fixed **reference level** (`ref_depth`, 0.195 m for
  a 100 m extent) — finer than any LOD leaf and coarser than deep leaves, so
  one decision covers a small patch.
* Every occupied ref-cell keeps an EMA (α) of its frame-mean importance.
* Candidate: SPLIT if ema > split_threshold, MERGE if ema < merge_threshold,
  otherwise KEEP (the deadband between the thresholds).
* A candidate must persist `split_frames` (SPLIT) or `merge_frames` (MERGE)
  consecutive frames before it is applied.
* Ref-cells with no evidence this frame decay their pending progress instead of
  committing, which bounds state and keeps the place memory short.
* Cells unseen for `expiry_frames` frames are swept periodically (amortized,
  frame-count based, deterministic).
* SPLIT cells refine one extra level, MERGE cells coarsen one level; hard level
  floors/ceilings (`min_depth`, `max_depth`) always win.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from adaptive_lidar.mapping.map_cell import (
    STATE_COARSE,
    STATE_FINE,
    STATE_MERGE_PENDING,
    STATE_SPLIT_PENDING,
)

KEEP, SPLIT, MERGE = 0, 1, 2
STATE_NAMES = {KEEP: "KEEP", SPLIT: "SPLIT", MERGE: "MERGE"}
CELL_STATE_NAMES = {
    KEEP: STATE_FINE, SPLIT: STATE_FINE, MERGE: STATE_COARSE,
}
PENDING_STATE_NAMES = {
    KEEP: STATE_FINE, SPLIT: STATE_SPLIT_PENDING, MERGE: STATE_MERGE_PENDING,
}


@dataclass
class HysteresisDecision:
    """Result of one controller tick."""

    shift: np.ndarray                 # (N,) per-point depth adjustment
    split_ops: int = 0
    merge_ops: int = 0
    ref_cell_mean: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    ref_keys: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))

    @property
    def total_ops(self) -> int:
        return self.split_ops + self.merge_ops


class HysteresisController:
    """Per-ref-cell EMA + persistence state machine."""

    def __init__(self, extent_m: float = 100.0, ref_depth: int = 9,
                 split_threshold: float = 0.60, merge_threshold: float = 0.30,
                 split_frames: int = 2, merge_frames: int = 3,
                 ema_alpha: float = 0.5, expiry_frames: int = 30,
                 expiry_sweep_every: int = 10, min_depth: int = 0,
                 max_depth: int = 11):
        if not 0.0 <= merge_threshold < split_threshold <= 1.0:
            raise ValueError("thresholds must satisfy 0 <= merge < split <= 1")
        if split_frames < 1 or merge_frames < 1:
            raise ValueError("persistence must be >= 1 frame")
        self.extent_m = float(extent_m)
        self.half = self.extent_m / 2.0
        self.ref_depth = int(ref_depth)
        self.ref_cell = self.extent_m / float(1 << self.ref_depth)
        self.split_threshold = float(split_threshold)
        self.merge_threshold = float(merge_threshold)
        self.split_frames = int(split_frames)
        self.merge_frames = int(merge_frames)
        self.ema_alpha = float(ema_alpha)
        self.expiry_frames = int(expiry_frames)
        self.expiry_sweep_every = max(1, int(expiry_sweep_every))
        self.min_depth = int(min_depth)
        self.max_depth = int(max_depth)
        # key -> [ema, state, pending, last_seen_frame]
        self.cells: dict[int, list] = {}
        self.frames_seen = 0
        self.total_split_ops = 0
        self.total_merge_ops = 0

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        self.cells.clear()
        self.frames_seen = 0
        self.total_split_ops = 0
        self.total_merge_ops = 0

    def ref_keys(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        last = (1 << self.ref_depth) - 1
        ix = np.clip(np.floor((x + self.half) / self.ref_cell).astype(np.int64), 0, last)
        iy = np.clip(np.floor((y + self.half) / self.ref_cell).astype(np.int64), 0, last)
        return (ix << 32) | iy

    def ref_cell_center(self, key: int) -> tuple[float, float]:
        ix, iy = int(key) >> 32, int(key) & 0xFFFFFFFF
        return (-self.half + (ix + 0.5) * self.ref_cell,
                -self.half + (iy + 0.5) * self.ref_cell)

    # ----------------------------------------------------------------- update
    def update(self, xyz: np.ndarray, importance: np.ndarray) -> HysteresisDecision:
        n = int(xyz.shape[0])
        if n == 0:
            return HysteresisDecision(np.zeros(0, np.int8))
        keys = self.ref_keys(xyz[:, 0].astype(np.float64), xyz[:, 1].astype(np.float64))
        imp = np.asarray(importance, dtype=np.float64).reshape(-1)

        order = np.argsort(keys, kind="stable")
        skeys = keys[order]
        uniq, first, counts = np.unique(skeys, return_index=True, return_counts=True)
        g = np.searchsorted(first, np.arange(n, dtype=np.int64), side="right") - 1
        cell_mean = np.bincount(g, weights=imp[order], minlength=uniq.shape[0]) / counts

        occupied = set(uniq.tolist())
        split_ops = merge_ops = 0
        frame = self.frames_seen

        for key, mean in zip(uniq.tolist(), cell_mean.tolist()):
            st = self.cells.get(key)
            if st is None:
                # [ema, state, pending, last_seen, candidate the pending run is for]
                st = [mean, KEEP, 0, frame, -1]
                self.cells[key] = st
            else:
                st[0] = self.ema_alpha * mean + (1.0 - self.ema_alpha) * st[0]
                st[3] = frame
            cand = self._candidate(st[0])
            if cand == st[1] or cand == KEEP:
                # Already at this level, or inside the deadband: hold the current
                # level and drop any half-finished persistence run.
                st[2] = 0
                st[4] = -1
                continue
            if st[4] != cand:
                # Persistence is counted per candidate: a frame arguing for MERGE
                # must never count towards the frames needed to SPLIT.
                st[4] = cand
                st[2] = 0
            st[2] += 1
            needed = self.split_frames if cand == SPLIT else self.merge_frames
            if st[2] >= needed:
                st[1] = cand
                st[2] = 0
                st[4] = -1
                if cand == SPLIT:
                    split_ops += 1
                    self.total_split_ops += 1
                elif cand == MERGE:
                    merge_ops += 1
                    self.total_merge_ops += 1

        # No evidence this frame: decay pending progress (keep the place memory).
        for key in list(self.cells.keys()):
            if key not in occupied:
                self.cells[key][2] = 0
                self.cells[key][4] = -1

        state_of = np.fromiter((self.cells[k][1] for k in keys.tolist()),
                               dtype=np.int8, count=n)
        shift = np.where(state_of == SPLIT, 1, np.where(state_of == MERGE, -1, 0)).astype(np.int8)

        self.frames_seen += 1
        if self.frames_seen % self.expiry_sweep_every == 0:
            cutoff = self.frames_seen - self.expiry_frames
            for key in [k for k, st in self.cells.items() if st[3] < cutoff]:
                del self.cells[key]

        return HysteresisDecision(shift, split_ops, merge_ops,
                                  cell_mean.astype(np.float32), uniq)

    def _candidate(self, ema: float) -> int:
        if ema > self.split_threshold:
            return SPLIT
        if ema < self.merge_threshold:
            return MERGE
        return KEEP

    # ---------------------------------------------------------------- inspect
    def state_of_point(self, x: float, y: float) -> dict:
        """Explain the ref-cell containing (x, y)."""
        key = int(self.ref_keys(np.array([x]), np.array([y]))[0])
        st = self.cells.get(key)
        cx, cy = self.ref_cell_center(key)
        ema = float(st[0]) if st else 0.0
        state = int(st[1]) if st else KEEP
        cand = self._candidate(ema)
        # Only report a run that is actually counting towards this candidate.
        pending = int(st[2]) if st and int(st[4]) == cand else 0
        return {
            "key": key,
            "center": (round(cx, 3), round(cy, 3)),
            "size_m": round(self.ref_cell, 4),
            "ref_depth": self.ref_depth,
            "state": STATE_NAMES[state],
            "resolution_state": PENDING_STATE_NAMES[state],
            "candidate": STATE_NAMES[cand],
            "ema_importance": round(ema, 4),
            "pending_frames": pending,
            "frames_needed": (self.split_frames if cand == SPLIT
                              else self.merge_frames if cand == MERGE else 0),
            "thresholds": (self.merge_threshold, self.split_threshold),
            "seen": st is not None,
            "last_seen_frame": st[3] if st else -1,
        }

    def status(self) -> dict:
        states = [st[1] for st in self.cells.values()]
        return {
            "ref_depth": self.ref_depth,
            "ref_cell_size_m": round(self.ref_cell, 4),
            "split_threshold": self.split_threshold,
            "merge_threshold": self.merge_threshold,
            "split_frames": self.split_frames,
            "merge_frames": self.merge_frames,
            "ema_alpha": self.ema_alpha,
            "min_depth": self.min_depth,
            "max_depth": self.max_depth,
            "tracked_cells": len(self.cells),
            "split_cells": int(sum(1 for s in states if s == SPLIT)),
            "merge_cells": int(sum(1 for s in states if s == MERGE)),
            "total_split_ops": self.total_split_ops,
            "total_merge_ops": self.total_merge_ops,
            "frames_seen": self.frames_seen,
        }
