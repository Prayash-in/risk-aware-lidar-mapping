"""Spatial quadtree over 2.5D map cells.

This is the *real* tree API (insert / find_leaf / query_region / split / merge /
get_leaves) that a spatial structure is expected to expose. The per-frame
projection in `projection_2_5d.py` rebuilds the leaf set every frame with
vectorized numpy reductions for speed; `Quadtree` provides the same structure
with explicit parent/child links for single-cell operations, unit tests and
tooling that need them.

Split/merge here are structural operations on the node graph, and they are
*never* used to delete information: merging a group of leaves produces the
aggregated 2.5D statistics of the coarser cell (count, height moments, semantic
vote), which is exactly the information the budget stage needs to stay correct.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from adaptive_lidar.mapping.map_cell import (
    STATE_COARSE,
    STATE_FINE,
    STATE_MERGE_PENDING,
    STATE_SPLIT_PENDING,
    MapCell,
)


@dataclass
class SplitResult:
    """Outcome of a split attempt (audit-friendly, never silent)."""

    ok: bool
    cell_id: int
    reason: str
    children: tuple = ()


@dataclass
class Quadtree:
    """Node graph keyed by (depth, ix, iy), rooted at depth 0."""

    extent_m: float
    cells: dict = field(default_factory=dict)   # key -> MapCell
    root_key: tuple = (0, 0, 0)
    _next_id: int = 0

    # ------------------------------------------------------------------ build
    @classmethod
    def from_cells(cls, cells: list[MapCell], extent_m: float) -> "Quadtree":
        tree = cls(extent_m=extent_m)
        for c in cells:
            tree.cells[c.key] = c
            tree._next_id = max(tree._next_id, c.cell_id + 1)
        return tree

    def new_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    # --------------------------------------------------------------- geometry
    def size_at(self, depth: int) -> float:
        """Physical cell size at a level: size = extent / 2**depth."""
        return self.extent_m / float(1 << int(depth))

    def depth_for_size(self, size_m: float) -> int:
        return int(np.ceil(np.log2(self.extent_m / float(size_m))))

    def cell_at(self, x: float, y: float, depth: int) -> MapCell | None:
        size = self.size_at(depth)
        half = self.extent_m / 2.0
        ix = int((x + half) // size)
        iy = int((y + half) // size)
        return self.cells.get((int(depth), ix, iy))

    def find_leaf(self, x: float, y: float) -> MapCell | None:
        """Finest occupied cell containing (x, y), or None.

        Rarely, parent and child both exist (during coarsening the hierarchy is
        still resolving), so the search runs deepest-first: the returned cell is
        the most detailed one.
        """
        for depth in range(self.max_depth_used(), -1, -1):
            cell = self.cell_at(x, y, depth)
            if cell is not None:
                return cell
        return None

    def max_depth_used(self) -> int:
        return max((k[0] for k in self.cells), default=0)

    def get_leaves(self) -> list[MapCell]:
        return list(self.cells.values())

    def num_nodes(self) -> int:
        return len(self.cells)

    # ---------------------------------------------------------------- queries
    def query_region(self, x_min: float, y_min: float, x_max: float, y_max: float,
                     max_results: int = 100_000) -> list[MapCell]:
        """Cells whose footprint intersects the axis-aligned region."""
        out: list[MapCell] = []
        for cell in self.cells.values():
            cx0, cy0, cx1, cy1 = cell.bounds
            if cx1 < x_min or cx0 > x_max or cy1 < y_min or cy0 > y_max:
                continue
            out.append(cell)
            if len(out) >= max_results:
                break
        return out

    def cells_within(self, cx: float, cy: float, radius: float) -> list[MapCell]:
        return self.query_region(cx - radius, cy - radius, cx + radius, cy + radius)

    # ----------------------------------------------------------- mutations
    def can_split(self, cell: MapCell, min_size: float, max_depth: int) -> SplitResult:
        if cell.depth >= max_depth:
            return SplitResult(False, cell.cell_id, "at_max_depth")
        if cell.size <= min_size * 1.0001:
            return SplitResult(False, cell.cell_id, "at_min_size")
        if not cell.is_leaf:
            return SplitResult(False, cell.cell_id, "already_internal")
        if any(k in self.cells for k in self._child_keys(cell.key)):
            return SplitResult(False, cell.cell_id, "children_exist")
        return SplitResult(True, cell.cell_id, "ok")

    def split(self, cell: MapCell, min_size: float = 0.05, max_depth: int = 11) -> SplitResult:
        """Replace one leaf with its four children (statistics are inherited).

        Child statistics are provisional: they carry the parent's aggregates
        until the next projection re-accumulates per-point values, which is
        why `projection_2_5d` remains the authoritative builder.
        """
        check = self.can_split(cell, min_size, max_depth)
        if not check.ok:
            return check
        depth = cell.depth + 1
        size = self.size_at(depth)
        ix, iy = cell.ix, cell.iy
        children: list[MapCell] = []
        for dy in (0, 1):
            for dx in (0, 1):
                cx = -self.extent_m / 2.0 + (ix * 2 + dx + 0.5) * size
                cy = -self.extent_m / 2.0 + (iy * 2 + dy + 0.5) * size
                child = MapCell(
                    cell_id=self.new_id(), center_x=cx, center_y=cy, size=size,
                    depth=depth, ix=ix * 2 + dx, iy=iy * 2 + dy,
                    parent=cell.cell_id, point_count=0,
                    range_score=cell.range_score, semantic_score=cell.semantic_score,
                    dynamic_score=cell.dynamic_score, bubble_score=cell.bubble_score,
                    prediction_score=cell.prediction_score,
                    importance_score=cell.importance_score,
                    resolution_state=cell.resolution_state,
                )
                self.cells[child.key] = child
                children.append(child)
        self.cells.pop(cell.key, None)
        return SplitResult(True, cell.cell_id, "split", tuple(c.cell_id for c in children))

    def _child_keys(self, key: tuple[int, int, int]) -> list[tuple[int, int, int]]:
        d, ix, iy = key
        return [(d + 1, ix * 2 + dx, iy * 2 + dy) for dy in (0, 1) for dx in (0, 1)]

    def merge(self, key: tuple[int, int, int]) -> MapCell | None:
        """Collapse the four children of `key` into `key`, aggregating stats.

        Returns the merged cell, or None when the four children are not all
        present (never partially merged: that would lose data).
        """
        child_keys = self._child_keys(key)
        if not all(k in self.cells for k in child_keys):
            return None
        kids = [self.cells[k] for k in child_keys]
        depth, ix, iy = key
        size = self.size_at(depth)
        n_total = sum(k.point_count for k in kids)
        if n_total > 0:
            w = np.array([max(k.point_count, 0) for k in kids], dtype=np.float64)
            if w.sum() <= 0:
                w = np.ones(4, dtype=np.float64)
            w = w / w.sum()
            mean = float(np.dot(w, [k.mean_height for k in kids]))
            var = float(np.dot(w, [k.height_variance for k in kids]) + np.dot(w, [(k.mean_height - mean) ** 2 for k in kids]))
            zmin = min(k.min_height for k in kids)
            zmax = max(k.max_height for k in kids)
        else:
            mean = float(np.mean([k.mean_height for k in kids]))
            var = float(np.mean([k.height_variance for k in kids]))
            zmin, zmax = 0.0, 0.0
        parent = self.cells.get(key)
        merged = MapCell(
            cell_id=parent.cell_id if parent is not None else self.new_id(),
            center_x=-self.extent_m / 2.0 + (ix + 0.5) * size,
            center_y=-self.extent_m / 2.0 + (iy + 0.5) * size,
            size=size, depth=depth, ix=ix, iy=iy,
            mean_height=mean, height_variance=var, min_height=zmin, max_height=zmax,
            occupancy=1.0, point_count=n_total,
            semantic_class=max(kids, key=lambda k: k.semantic_confidence).semantic_class,
            semantic_confidence=float(np.max([k.semantic_confidence for k in kids])),
            range_score=float(np.max([k.range_score for k in kids])),
            semantic_score=float(np.max([k.semantic_score for k in kids])),
            dynamic_score=float(np.max([k.dynamic_score for k in kids])),
            bubble_score=float(np.max([k.bubble_score for k in kids])),
            prediction_score=float(np.max([k.prediction_score for k in kids])),
            importance_score=float(np.max([k.importance_score for k in kids])),
        )
        for k in child_keys:
            self.cells.pop(k, None)
        self.cells[key] = merged
        return merged

    # ------------------------------------------------------ resolution state
    def mark(self, key: tuple[int, int, int], state: str) -> None:
        cell = self.cells.get(key)
        if cell is not None:
            cell.resolution_state = state

    def states(self) -> dict:
        out = {STATE_COARSE: 0, STATE_SPLIT_PENDING: 0, STATE_FINE: 0, STATE_MERGE_PENDING: 0}
        for cell in self.cells.values():
            out[cell.resolution_state] = out.get(cell.resolution_state, 0) + 1
        return out

    def to_dicts(self) -> list[dict]:
        return [c.to_dict() for c in self.cells.values()]
