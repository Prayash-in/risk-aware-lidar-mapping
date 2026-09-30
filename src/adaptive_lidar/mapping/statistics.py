"""Per-cell 2.5D statistics accumulation (vectorized).

These are the statistics §14 requires per map cell: point count, height
mean/variance/min/max, dominant semantic class with vote confidence, and
optional per-factor importance means. Everything is computed with numpy
`bincount` reductions over a point-to-cell assignment, so the cost is
O(N) in the number of returns, independent of the number of cells.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CellStatistics:
    """Vectorized 2.5D statistics for a set of occupied cells."""

    count: np.ndarray                 # (L,) int64
    z_mean: np.ndarray                # (L,) float64
    z_var: np.ndarray                 # (L,) float64
    z_min: np.ndarray                 # (L,) float64
    z_max: np.ndarray                 # (L,) float64
    semantic: np.ndarray              # (L,) int32 dominant class
    confidence: np.ndarray            # (L,) float64 dominant vote fraction
    importance: np.ndarray | None = None      # (L,) mean total importance
    factors: dict = field(default_factory=dict)  # name -> (L,) mean factor value

    @property
    def num_cells(self) -> int:
        return int(self.count.shape[0])

    def nbytes(self) -> int:
        total = int(self.count.nbytes + self.z_mean.nbytes + self.z_var.nbytes
                    + self.z_min.nbytes + self.z_max.nbytes
                    + self.semantic.nbytes + self.confidence.nbytes)
        if self.importance is not None:
            total += int(self.importance.nbytes)
        for arr in self.factors.values():
            total += int(arr.nbytes)
        return total


def accumulate(
    group: np.ndarray,
    z: np.ndarray,
    semantic: np.ndarray,
    num_groups: int,
    num_classes: int = 9,
    group_starts: np.ndarray | None = None,
    factors: dict | None = None,
) -> CellStatistics:
    """Reduce per-point arrays into per-cell statistics.

    group: (N,) int cell index in [0, num_groups)
    factors: optional dict of name -> (N,) per-point values in [0, 1]; their
      per-cell means are exposed for the controller and for documentation.
    """
    group = np.asarray(group, dtype=np.int64)
    z = np.asarray(z, dtype=np.float64)
    n = z.shape[0]
    count = np.bincount(group, minlength=num_groups).astype(np.int64)
    safe = np.maximum(count, 1)
    sum_z = np.bincount(group, weights=z, minlength=num_groups)
    sum_z2 = np.bincount(group, weights=z * z, minlength=num_groups)
    mean = sum_z / safe
    var = np.clip(sum_z2 / safe - mean * mean, 0.0, None)
    var[count == 0] = 0.0

    if group_starts is not None:
        # z is already grouped-sorted, so the first index of each group is a
        # valid `reduceat` boundary and the min/max pass is O(N).
        z_min = np.minimum.reduceat(z, group_starts)
        z_max = np.maximum.reduceat(z, group_starts)
    else:
        z_min = np.full(num_groups, np.nan)
        z_max = np.full(num_groups, np.nan)
        for g in np.unique(group):
            z_min[g] = z[group == g].min()
            z_max[g] = z[group == g].max()
    z_min = np.nan_to_num(z_min)
    z_max = np.nan_to_num(z_max)

    sem = np.clip(np.asarray(semantic, dtype=np.int64), 0, num_classes - 1)
    # A single bincount over the packed (group, class) index: same result as
    # np.add.at on a (L, C) matrix, but far faster for large scans.
    flat = np.bincount(group * num_classes + sem,
                       minlength=num_groups * num_classes).reshape(num_groups, num_classes)
    votes = flat
    dominant = np.argmax(votes, axis=1).astype(np.int32)
    confidence = np.max(votes, axis=1) / safe

    imp = None
    fac_out: dict = {}
    for name, values in (factors or {}).items():
        arr = np.asarray(values, dtype=np.float64)
        m = (np.bincount(group, weights=arr, minlength=num_groups) / safe)
        fac_out[name] = m.astype(np.float32)
        if name == "total":
            imp = m.astype(np.float32)

    return CellStatistics(
        count=count, z_mean=mean, z_var=var, z_min=z_min, z_max=z_max,
        semantic=dominant, confidence=confidence, importance=imp, factors=fac_out,
    )


def height_score(z_var: np.ndarray, scale: float = 0.25) -> np.ndarray:
    """Vertical-structure term in [0, 1]: 1 - exp(-σ²/scale) (§14).

    High variance means the cell straddles real 3D structure (curb face,
    vehicle body, vegetation) rather than a flat surface.
    """
    v = np.asarray(z_var, dtype=np.float64)
    return 1.0 - np.exp(-np.clip(v, 0.0, None) / max(scale, 1e-6))


def coverage_score(count: np.ndarray, target: float = 6.0) -> np.ndarray:
    """Saturating point-density term in [0, 1]: min(1, n / target)."""
    return np.clip(np.asarray(count, dtype=np.float64) / max(target, 1e-6), 0.0, 1.0)
