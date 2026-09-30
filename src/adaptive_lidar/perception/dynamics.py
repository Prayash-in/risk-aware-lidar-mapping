"""Temporal change detection -> per-point dynamic score (dynamics, §8).

Two independent, documented signals, both needed for honest evaluation:

1. `DynamicScorer` compares the current 2.5D occupancy/height field with a
   short memory of previous frames and flags cells whose geometry moved
   (change detection). This works with *any* semantics, including `unknown`, and
   is what a real vehicle can do without labels.
2. Class-based dynamism (vehicle / pedestrian / cyclist) comes from
   `SemanticLabeler.dynamic_labels` and is combined by the controller.

The score decays over `decay_frames` so a moving object keeps a trailing
bubble instead of flickering on and off.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ChangeField:
    """Quantized 2.5D signature of one frame (occupancy + height)."""

    cell_size: float
    keys: np.ndarray        # (L,) packed (ix, iy)
    z_mean: np.ndarray      # (L,)
    z_max: np.ndarray       # (L,)
    timestamp: float = 0.0


class DynamicScorer:
    """Occupancy/height change detection with a bounded memory."""

    def __init__(self, cell_size: float = 0.5, height_tol: float = 0.15,
                 z_tol: float = 0.30, decay_frames: int = 4, memory_frames: int = 3,
                 max_score: float = 1.0):
        self.cell_size = float(cell_size)
        self.height_tol = float(height_tol)
        self.z_tol = float(z_tol)
        self.decay_frames = max(1, int(decay_frames))
        self.memory_frames = max(1, int(memory_frames))
        self.max_score = float(max_score)
        self._history: list[ChangeField] = []
        self._heat: np.ndarray | None = None   # (keys,) decayed dynamic evidence
        self._heat_keys: np.ndarray | None = None
        self.last_change: np.ndarray | None = None  # (L,) change in [0, 1]

    # ------------------------------------------------------------------ field
    def _signature(self, xyz: np.ndarray, frame_id: int, timestamp: float) -> ChangeField:
        v = self.cell_size
        ix = np.floor(xyz[:, 0] / v).astype(np.int64)
        iy = np.floor(xyz[:, 1] / v).astype(np.int64)
        keys = (ix + (1 << 20)) * (1 << 21) + (iy + (1 << 20))
        order = np.argsort(keys, kind="stable")
        sk = keys[order]
        uniq, first, counts = np.unique(sk, return_index=True, return_counts=True)
        z = xyz[order, 2].astype(np.float64)
        g = np.searchsorted(first, np.arange(sk.shape[0], dtype=np.int64), side="right") - 1
        sum_z = np.bincount(g, weights=z, minlength=uniq.shape[0])
        mean = sum_z / np.maximum(counts, 1)
        z_max = np.maximum.reduceat(z, first)
        return ChangeField(v, uniq, mean.astype(np.float32), z_max.astype(np.float32),
                           float(timestamp))

    # ----------------------------------------------------------------- update
    def update(self, xyz: np.ndarray, frame_id: int = 0, timestamp: float = 0.0,
               dynamic_mask: np.ndarray | None = None) -> np.ndarray:
        """Return a per-point dynamic score in [0, 1].

        dynamic_mask: optional per-point class-based dynamism (from semantics);
            when provided the two signals are combined by a weighted maximum.
        """
        xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
        n = xyz.shape[0]
        if n == 0:
            return np.zeros(0, np.float32)
        sig = self._signature(xyz, frame_id, timestamp)

        change = np.zeros(sig.keys.shape[0], dtype=np.float32)
        if self._history:
            prev = self._history[-1]
            common, ia, ib = np.intersect1d(sig.keys, prev.keys,
                                            assume_unique=True, return_indices=True)
            if common.shape[0]:
                d_mean = np.abs(sig.z_mean[ia] - prev.z_mean[ib])
                d_max = np.abs(sig.z_max[ia] - prev.z_max[ib])
                moved = ((d_mean > self.height_tol) | (d_max > self.z_tol))
                change[ia[moved]] = 1.0
            # Cells that *appeared* between the two frames are also evidence of
            # change. Vanished cells leave a decaying ghost in ``_heat`` and are
            # scored again if they reappear (e.g. after occlusion).
            appeared = np.setdiff1d(sig.keys, prev.keys, assume_unique=True)
            if appeared.shape[0]:
                change[np.searchsorted(sig.keys, appeared)] = 1.0

        # Decaying heat over the cell grid.
        if self._heat is None or self._heat_keys is None:
            self._heat_keys, self._heat = sig.keys.copy(), change.copy()
        else:
            allk = np.union1d(self._heat_keys, sig.keys)
            heat = np.zeros(allk.shape[0], dtype=np.float32)
            m_old = np.searchsorted(allk, self._heat_keys)
            heat[m_old] = np.maximum(heat[m_old], self._heat * self._decay())
            m_new = np.searchsorted(allk, sig.keys)
            heat[m_new] = np.maximum(heat[m_new], change)
            self._heat_keys, self._heat = allk, heat

        self._history.append(sig)
        if len(self._history) > self.memory_frames:
            self._history.pop(0)
        self.last_change = change

        # Per-point score: look the point's own cell heat up.
        ix = np.floor(xyz[:, 0] / self.cell_size).astype(np.int64)
        iy = np.floor(xyz[:, 1] / self.cell_size).astype(np.int64)
        pkeys = (ix + (1 << 20)) * (1 << 21) + (iy + (1 << 20))
        pos = np.searchsorted(self._heat_keys, pkeys)
        pos = np.clip(pos, 0, self._heat_keys.shape[0] - 1)
        score = np.where(self._heat_keys[pos] == pkeys, self._heat[pos], 0.0)

        if dynamic_mask is not None:
            class_score = np.asarray(dynamic_mask, dtype=np.float32).reshape(-1)
            score = np.maximum(score, class_score)
        return np.clip(score, 0.0, self.max_score).astype(np.float32)

    def _decay(self) -> float:
        return float(1.0 - 1.0 / self.decay_frames)

    def reset(self) -> None:
        self._history.clear()
        self._heat = self._heat_keys = None
        self.last_change = None
