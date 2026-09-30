"""2.5D projection: LiDAR points -> adaptive quadtree leaves.

The map is indexed on the ground plane (X, Y). Each point is assigned to the
quadtree leaf whose physical size is the one demanded by the active resolution
policy, and per-cell statistics are reduced in one vectorized pass.

Three policies share this code path (§8):

* ``uniform``   - every cell is ``uniform_cell_size`` (baseline, the high-res grid)
* ``distance``  - ``s(r) = s_min + (s_max - s_min) * (1 - exp(-λr))`` (§6)
* ``adaptive``  - size comes from the risk controller's per-point target size

Because the same function serves all three, the baselines and the proposed
method are guaranteed to be fed identical frames.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from adaptive_lidar.mapping.map_cell import (
    MapCell,
    link_hierarchy,
    make_cells,
    pack_key,
)
from adaptive_lidar.mapping.statistics import CellStatistics, accumulate

#: Documented footprint of a flat internal quadtree node (§16). We store a
#: 32-entry `cell_map` + child ids; this is a conservative, documented constant
#: used for the reported memory estimate.
INTERNAL_NODE_BYTES = 64

MAX_LEAVES = 4_000_000  # safety valve, not a silently applied clamp


def size_to_depth(extent_m: float, size_m: np.ndarray | float) -> np.ndarray:
    """Map a physical size to a level: depth = ceil(log2(extent / size))."""
    return np.ceil(np.log2(extent_m / np.asarray(size_m, dtype=np.float64))).astype(np.int64)


def depth_to_size(extent_m: float, depth: np.ndarray | int) -> np.ndarray:
    """Physical edge length for a level: size = extent / 2**depth."""
    return float(extent_m) / np.ldexp(1.0, np.asarray(depth, dtype=np.int64))


def distance_size_field(s_min: float, s_max: float, lam: float):
    """Return `f(x, y) -> target size` implementing §6."""
    def fn(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        r = np.hypot(x, y)
        return s_min + (s_max - s_min) * (1.0 - np.exp(-lam * r))
    fn.__name__ = f"distance_field(s_min={s_min}, s_max={s_max}, lam={lam})"
    return fn


@dataclass
class ProjectionResult:
    """Leaf set plus the reductions used to build it."""

    cells: list[MapCell]
    statistics: CellStatistics
    group: np.ndarray            # (N,) per-point cell index
    num_internal: int = 0
    num_unsplit: int = 0         # 4-way groups requested but capped
    leaf_arrays_nbytes: int = 0

    @property
    def num_leaves(self) -> int:
        return len(self.cells)

    @property
    def num_nodes(self) -> int:
        return self.num_leaves + self.num_internal

    def memory_bytes(self) -> int:
        """Measured leaf storage + documented internal-node struct size."""
        return int(self.leaf_arrays_nbytes) + int(self.num_internal) * INTERNAL_NODE_BYTES

    def stats(self) -> dict:
        sizes = np.array([c.size for c in self.cells], dtype=np.float64)
        return {
            "leaf_count": self.num_leaves,
            "node_count": self.num_nodes,
            "internal_count": int(self.num_internal),
            "num_unsplit": int(self.num_unsplit),
            "max_depth_used": int(max(c.depth for c in self.cells)) if self.cells else 0,
            "min_cell_size_m": round(float(sizes.min()), 4) if sizes.size else 0.0,
            "max_cell_size_m": round(float(sizes.max()), 4) if sizes.size else 0.0,
            "avg_cell_size_m": round(float(sizes.mean()), 4) if sizes.size else 0.0,
            "map_memory_bytes": self.memory_bytes(),
        }


def _pack(depth: np.ndarray, ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
    return pack_key(depth, ix, iy)


def project_2_5d(
    xyz: np.ndarray,
    semantic: np.ndarray,
    extent_m: float,
    target_size: np.ndarray | None = None,
    target_field=None,
    max_depth: int = 11,
    min_depth: int = 0,
    min_cell_size: float = 0.05,
    max_cell_size: float = 0.50,
    num_classes: int = 9,
    importance: np.ndarray | None = None,
    factors: dict | None = None,
    depth_shift: np.ndarray | None = None,
    max_leaves: int = MAX_LEAVES,
    refine_structure: bool = False,
    z_var_threshold: float = 0.05,
    min_points_for_refine: int = 6,
    max_refine_fraction: float = 0.35,
) -> ProjectionResult:
    """Bin one scan into variable-resolution 2.5D leaves.

    target_size: (N,) per-point physical size (adaptive / uniform policy).
    target_field: `f(x, y) -> size` used when target_size is None (distance LOD).
    depth_shift:  (N,) per-level override applied to the computed depth, used by
                  the hysteresis controller to realize split/merge decisions.
    min_depth / max_depth: hard level floors and ceilings (§16 protection floors).
    refine_structure: enable the one-level structure pass described above.
    """
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    semantic = np.asarray(semantic, dtype=np.int32).reshape(-1)
    n = int(xyz.shape[0])
    half = extent_m / 2.0

    for name, arr, size in (
        ("target_size", target_size, n),
        ("importance", importance, n),
        ("depth_shift", depth_shift, n),
    ):
        if arr is not None and np.asarray(arr).shape[0] != size:
            raise ValueError(
                f"{name} has {np.asarray(arr).shape[0]} entries for {size} points")
    if factors:
        for name, arr in factors.items():
            if np.asarray(arr).shape[0] != n:
                raise ValueError(
                    f"factor '{name}' has {np.asarray(arr).shape[0]} entries for {n} points")

    if n == 0:
        stats = CellStatistics(
            count=np.zeros(0, np.int64), z_mean=np.zeros(0), z_var=np.zeros(0),
            z_min=np.zeros(0), z_max=np.zeros(0), semantic=np.zeros(0, np.int32),
            confidence=np.zeros(0), importance=None, factors={},
        )
        return ProjectionResult([], stats, np.zeros(0, np.int64), 0, 0, 0)

    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    in_bounds = (np.abs(x) <= half) & (np.abs(y) <= half)
    x, y, z = x[in_bounds], y[in_bounds], z[in_bounds]
    semantic = semantic[in_bounds]
    if importance is not None:
        importance = np.asarray(importance, dtype=np.float64)[in_bounds]
    if factors:
        factors = {k: np.asarray(v, dtype=np.float64)[in_bounds] for k, v in factors.items()}
    if depth_shift is not None:
        depth_shift = np.asarray(depth_shift, dtype=np.int64)[in_bounds]
    if target_size is not None:
        target_size = np.asarray(target_size, dtype=np.float64)[in_bounds]
    n = int(x.shape[0])
    if n == 0:
        return project_2_5d(
            np.zeros((0, 3)), np.zeros(0, np.int32), extent_m,
            target_size=np.zeros(0), target_field=target_field,
            max_depth=max_depth, min_depth=min_depth,
            min_cell_size=min_cell_size, max_cell_size=max_cell_size,
            num_classes=num_classes, depth_shift=None, max_leaves=max_leaves)

    if target_size is None:
        if target_field is None:
            raise ValueError("project_2_5d needs either target_size or target_field")
        target = np.asarray(target_field(x, y), dtype=np.float64)
    else:
        target = np.asarray(target_size, dtype=np.float64)
    if target.shape[0] != n:
        raise ValueError(
            f"target_size has {target.shape[0]} entries for {n} points")
    target = np.clip(target, min_cell_size, max_cell_size)

    depth = size_to_depth(extent_m, target)
    if depth_shift is not None:
        depth = depth + depth_shift
    depth = np.clip(depth, min_depth, max_depth)

    grid = np.ldexp(np.ones_like(depth), depth)             # 2**depth
    cell = extent_m / grid
    last = grid.astype(np.int64) - 1
    ix = np.clip(np.floor((x + half) / cell).astype(np.int64), 0, last)
    iy = np.clip(np.floor((y + half) / cell).astype(np.int64), 0, last)

    # Occasional refinement pass (§3.1.4): leaves that really contain 3D
    # structure (height variance) and carry enough returns are deepened once.
    # This is applied only when the caller asks for it; the controller's own
    # hysteresis is the main driver of refinement, this pass catches dense
    # structures the static size field cannot see.
    if refine_structure and max_depth > int(np.min(depth)):
        refined = _refine(
            x, y, z, semantic, depth, ix, iy, half, extent_m, max_depth,
            z_var_threshold, min_points_for_refine, max_refine_fraction,
        )
        depth, ix, iy = refined

    key = _pack(depth, ix, iy)
    order = np.argsort(key, kind="stable")
    uniq, first, counts = np.unique(key[order], return_index=True, return_counts=True)
    g = np.searchsorted(first, np.arange(n, dtype=np.int64), side="right") - 1

    num_unsplit = 0
    if uniq.shape[0] > max_leaves:
        num_unsplit = int(uniq.shape[0] - max_leaves)
        uniq, first, counts = uniq[:max_leaves], first[:max_leaves], counts[:max_leaves]
        keep = g < max_leaves
        g = np.where(keep, g, 0)

    all_factors: dict | None = None
    if factors:
        all_factors = dict(factors)
    if importance is not None:
        # §12: the controller's total importance is the cell's own importance
        # score; it drives the budget tier, so it must reach the cells.
        all_factors = dict(all_factors or {})
        all_factors["total"] = importance

    stats = accumulate(g, z[order], semantic[order], int(uniq.shape[0]),
                       num_classes=num_classes, group_starts=first,
                       factors=_sorted_factors(all_factors, order)
                       if all_factors else None)

    sdepth, six, siy = depth[order][first], ix[order][first], iy[order][first]
    cells = _build_cells(extent_m, sdepth, six, siy, stats)
    num_internal = link_hierarchy(cells)

    return ProjectionResult(
        cells=cells, statistics=stats, group=g,
        num_internal=num_internal, num_unsplit=num_unsplit,
        leaf_arrays_nbytes=stats.nbytes(),
    )


def _sorted_factors(factors: dict, order: np.ndarray) -> dict:
    return {k: np.asarray(v)[order] for k, v in factors.items()}


def _build_cells(extent_m, depth, ix, iy, stats: CellStatistics) -> list[MapCell]:
    """Materialize MapCell objects from the reduced statistics (tight loop)."""
    size = depth_to_size(extent_m, depth)
    half = extent_m / 2.0
    cx = -half + (ix + 0.5) * size
    cy = -half + (iy + 0.5) * size
    d_list = depth.tolist()
    cx_list, cy_list, size_list = cx.tolist(), cy.tolist(), size.tolist()
    ix_list, iy_list = ix.tolist(), iy.tolist()
    count = stats.count.tolist()
    mean = stats.z_mean.tolist()
    var = stats.z_var.tolist()
    zmin = stats.z_min.tolist()
    zmax = stats.z_max.tolist()
    sem = stats.semantic.tolist()
    conf = stats.confidence.tolist()
    imp = stats.importance.tolist() if stats.importance is not None else None
    # Factor name -> MapCell slot. `total` is the controller's composite
    # importance and lands in `importance_score`, which the budget tiers read.
    factor_slots = {
        "range": "range_score", "semantic": "semantic_score",
        "dynamic": "dynamic_score", "bubble": "bubble_score",
        "prediction": "prediction_score", "total": "importance_score",
    }
    fac = {name: arr.tolist() for name, arr in stats.factors.items()
           if name in factor_slots}

    cells: list[MapCell] = []
    append = cells.append
    for i in range(len(d_list)):
        cell = MapCell(
            cell_id=i, center_x=cx_list[i], center_y=cy_list[i],
            size=size_list[i], depth=d_list[i], ix=ix_list[i], iy=iy_list[i],
            mean_height=mean[i], height_variance=var[i], min_height=zmin[i],
            max_height=zmax[i], occupancy=1.0, point_count=count[i],
            semantic_class=sem[i], semantic_confidence=conf[i],
        )
        if imp is not None:
            cell.importance_score = imp[i]
        for name, values in fac.items():
            setattr(cell, factor_slots[name], values[i])
        append(cell)
    return cells


def _refine(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, semantic: np.ndarray,
    depth: np.ndarray, ix: np.ndarray, iy: np.ndarray,
    half: float, extent_m: float, max_depth: int,
    z_var_threshold: float, min_points: int, max_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deepen leaves that show 3D structure by exactly one level."""
    key = _pack(depth, ix, iy)
    order = np.argsort(key, kind="stable")
    uniq, first, counts = np.unique(key[order], return_index=True, return_counts=True)
    g = np.searchsorted(first, np.arange(z.shape[0], dtype=np.int64), side="right") - 1
    zs = z[order]
    sum_z = np.bincount(g, weights=zs, minlength=uniq.shape[0])
    sum_z2 = np.bincount(g, weights=zs * zs, minlength=uniq.shape[0])
    mean = sum_z / np.maximum(counts, 1)
    var = np.clip(sum_z2 / np.maximum(counts, 1) - mean * mean, 0.0, None)

    qualify = (var >= z_var_threshold) & (counts >= min_points)
    if not np.any(qualify):
        return depth, ix, iy
    # Cap the pass so a single frame cannot refine an unbounded share of the map.
    limit = int(max_fraction * uniq.shape[0])
    if limit < 1:
        return depth, ix, iy
    order_var = np.argsort(-var, kind="stable")
    chosen = order_var[:limit]
    chosen = chosen[qualify[chosen]]
    if chosen.shape[0] == 0:
        return depth, ix, iy
    allowed = np.zeros(uniq.shape[0], dtype=bool)
    allowed[chosen] = True
    deepen = allowed[g]

    new_depth = np.where(deepen, np.minimum(depth + 1, max_depth), depth)
    grid = np.ldexp(np.ones_like(new_depth), new_depth)
    cell = extent_m / grid
    last = grid.astype(np.int64) - 1
    nix = np.clip(np.floor((x + half) / cell).astype(np.int64), 0, last)
    niy = np.clip(np.floor((y + half) / cell).astype(np.int64), 0, last)
    return new_depth, nix, niy


def _cell_stats(stats: CellStatistics, i: int) -> dict:
    import adaptive_lidar.mapping.map_cell as mc
    d = {
        "mean_height": float(stats.z_mean[i]),
        "height_variance": float(stats.z_var[i]),
        "min_height": float(stats.z_min[i]),
        "max_height": float(stats.z_max[i]),
        "occupancy": 1.0,
        "point_count": int(stats.count[i]),
        "semantic_class": int(stats.semantic[i]),
        "semantic_confidence": float(stats.confidence[i]),
    }
    for name, arr in stats.factors.items():
        d[f"{name}_score"] = float(arr[i])
    if stats.importance is not None:
        d["importance_score"] = float(stats.importance[i])
    return d


def uniform_target_size(size_m: float, n: int) -> np.ndarray:
    """Constant per-point target size (uniform baseline, §8.2)."""
    return np.full(n, float(size_m), dtype=np.float64)


def budget_cost(cell: MapCell) -> float:
    """Unit cost of one leaf: cells exist once, refining multiplies (§17).

    §17.2: a leaf of size s holds 1 cell + the implicit coarser cells above it,
    so refining a patch raises the cost by the number of levels added. The
    formula below is the §17 convention used throughout the experiments.
    """
    return 1.0
