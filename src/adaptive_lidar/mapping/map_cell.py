"""2.5D map cell: one occupied quadtree leaf with its statistics and risk state.

2.5D means: the cell is addressed by its ground-plane rectangle (x, y, size)
and stores height statistics (μ, σ², min, max) plus semantic/dynamic
information. There are no 3D voxels anywhere in this repository.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from adaptive_lidar.types import ImportanceScore, resolution_tier

#: Documented footprint of a map cell in a flat/struct-of-arrays store.
#: key(8) + center(8) + size(4) + depth(1) + flags(1) + 4 stats(16) + count(4)
#: + semantic(4) + confidence(4) + importance(4) = 54 B -> padded to 64 B.
CELL_BYTES = 64

#: Resolution states used by the hysteresis controller (§16).
STATE_COARSE = "coarse"
STATE_SPLIT_PENDING = "split_pending"
STATE_FINE = "fine"
STATE_MERGE_PENDING = "merge_pending"


@dataclass(slots=True)
class MapCell:
    """An occupied 2.5D quadtree leaf."""

    cell_id: int
    center_x: float
    center_y: float
    size: float                 # physical edge length in meters
    depth: int                  # size == extent / 2**depth
    ix: int = 0                 # column index of the leaf inside its level
    iy: int = 0                 # row index of the leaf inside its level
    parent: int = -1
    children: tuple = ()        # (cell_id, cell_id, cell_id, cell_id); empty if leaf
    is_leaf: bool = True

    # --- 2.5D statistics ---
    mean_height: float = 0.0
    height_variance: float = 0.0
    min_height: float = 0.0
    max_height: float = 0.0
    occupancy: float = 0.0      # 0..1, fraction of the cell footprint with returns
    point_count: int = 0

    # --- semantics / risk state ---
    semantic_class: int = 0
    semantic_confidence: float = 0.0
    range_score: float = 0.0
    semantic_score: float = 0.0
    dynamic_score: float = 0.0
    bubble_score: float = 0.0
    prediction_score: float = 0.0
    importance_score: float = 0.0
    resolution_state: str = STATE_FINE

    @property
    def key(self) -> tuple[int, int, int]:
        """Structural key (depth, ix, iy); siblings share (depth, ix//2, iy//2)."""
        return (self.depth, self.ix, self.iy)

    @property
    def parent_key(self) -> tuple[int, int, int]:
        """Key of the coarser cell that contains this one (depth 0 -> itself)."""
        if self.depth == 0:
            return self.key
        return (self.depth - 1, self.ix >> 1, self.iy >> 1)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        h = self.size / 2.0
        return (self.center_x - h, self.center_y - h, self.center_x + h, self.center_y + h)

    @property
    def tier(self) -> str:
        return resolution_tier(self.size)

    @property
    def importance(self) -> ImportanceScore:
        return ImportanceScore(
            range=self.range_score, semantic=self.semantic_score,
            dynamic=self.dynamic_score, bubble=self.bubble_score,
            prediction=self.prediction_score, total=self.importance_score,
        )

    def memory_bytes(self) -> int:
        return CELL_BYTES

    def to_dict(self) -> dict:
        """Serializable snapshot (JSON-ready, no numpy scalars)."""
        return {
            "cell_id": int(self.cell_id),
            "x": round(float(self.center_x), 3),
            "y": round(float(self.center_y), 3),
            "size": round(float(self.size), 4),
            "resolution": round(float(self.size), 4),
            "depth": int(self.depth),
            "tier": self.tier,
            "resolution_state": self.resolution_state,
            "is_leaf": bool(self.is_leaf),
            "parent": int(self.parent),
            "children": [int(c) for c in self.children],
            "mean_height": round(float(self.mean_height), 3),
            "height_variance": round(float(self.height_variance), 5),
            "min_height": round(float(self.min_height), 3),
            "max_height": round(float(self.max_height), 3),
            "occupancy": round(float(self.occupancy), 3),
            "point_count": int(self.point_count),
            "semantic_class": int(self.semantic_class),
            "semantic_confidence": round(float(self.semantic_confidence), 3),
            "range_score": round(float(self.range_score), 3),
            "semantic_score": round(float(self.semantic_score), 3),
            "dynamic_score": round(float(self.dynamic_score), 3),
            "bubble_score": round(float(self.bubble_score), 3),
            "prediction_score": round(float(self.prediction_score), 3),
            "importance_score": round(float(self.importance_score), 3),
        }


# Half-extent bookkeeping lives in the builder, which knows the map extent; the
# (ix, iy) indices are stored on the cell so `key` needs no global state.


def make_cells(
    extent_m: float, keys: list[tuple[int, int, int]], stats: list[dict]
) -> list[MapCell]:
    """Build `MapCell` objects from (depth, ix, iy) keys + statistics dicts."""
    out: list[MapCell] = []
    for i, (depth, ix, iy) in enumerate(keys):
        size = extent_m / float(1 << depth)
        cx = -extent_m / 2.0 + (ix + 0.5) * size
        cy = -extent_m / 2.0 + (iy + 0.5) * size
        s = dict(stats[i]) if i < len(stats) else {}
        out.append(MapCell(
            cell_id=i, center_x=cx, center_y=cy, size=size, depth=depth,
            ix=ix, iy=iy, **s,
        ))
    return out


def pack_key(depth, ix, iy):
    """Pack (depth, ix, iy) into one int64 key (bit fields, stable for sort)."""
    import numpy as np
    return ((np.asarray(depth, dtype=np.int64) << 56)
            | (np.asarray(ix, dtype=np.int64) << 28)
            | np.asarray(iy, dtype=np.int64))


def count_internal_nodes(cells: list[MapCell]) -> int:
    """Exact number of structural internal nodes implied by a leaf set (vectorized).

    Every ancestor of every leaf is an internal node unless it is itself a
    materialized leaf. Used both by `link_hierarchy` and by the budget report,
    where the coarsened leaf set no longer has materialized parents.
    """
    import numpy as np

    if not cells:
        return 0
    depth = np.fromiter((c.depth for c in cells), dtype=np.int64, count=len(cells))
    ix = np.fromiter((c.ix for c in cells), dtype=np.int64, count=len(cells))
    iy = np.fromiter((c.iy for c in cells), dtype=np.int64, count=len(cells))
    leaf_keys = pack_key(depth, ix, iy)
    internal = 0
    for d in range(0, int(depth.max())):
        mask = depth > d
        if not np.any(mask):
            continue
        shift = depth[mask] - d
        anc = pack_key(np.full(ix[mask].shape, d, dtype=np.int64),
                       ix[mask] >> shift, iy[mask] >> shift)
        uniq = np.unique(anc)
        internal += int(np.count_nonzero(~np.isin(uniq, leaf_keys)))
    return internal


def link_hierarchy(cells: list[MapCell]) -> int:
    """Fill in `parent`/`children` links from the (depth, ix, iy) keys.

    Returns the number of internal nodes in the implied hierarchy, counted
    exactly: every ancestor of every leaf is an internal node unless it is
    itself a materialized leaf. The map is rebuilt from scratch every frame, so
    links are derived from this frame's keys, never inherited from the previous
    one.
    """
    import numpy as np

    if not cells:
        return 0
    index = {c.key: c.cell_id for c in cells}
    by_id = {c.cell_id: c for c in cells}

    for c in cells:
        if c.depth == 0:
            continue
        pid = index.get(c.parent_key)
        if pid is None or pid == c.cell_id:
            continue
        c.parent = pid
        parent = by_id[pid]
        children = list(parent.children)
        while len(children) < 4:
            children.append(-1)
        children[(c.iy & 1) * 2 + (c.ix & 1)] = c.cell_id
        parent.children = tuple(children)
        parent.is_leaf = False

    return count_internal_nodes(cells)
