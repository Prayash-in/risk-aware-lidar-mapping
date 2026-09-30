"""Compute budget and graceful degradation (§17).

Computational capacity is a limited resource, so the map is not allowed to cost
more than its budget. Cost model (transparent, in abstract units):

    cost(cell) = 2^depth · (1 + point_count / 8) · update_rate(cell)

* `2^depth` — refinement cost grows with the level, so splitting is expensive
  exactly where it should be: only when the risk controller asked for it.
* `1 + n/8` — aggregation cost scales with the returns inside the cell.
* `update_rate` — multi-rate refresh tier priced from importance, aligned to
  the hysteresis thresholds: 1.0 hot (I >= split), 0.5 warm, 0.15 cold
  (I < merge). The prototype rebuilds the full map every frame, but it
  *enforces the structure* (which cells exist at which level and rate); the
  model prices that structure. This assumption is explicit, not hidden.

Enforcement (`enforce_budget`):

* `allowed = pct · (unconstrained cost of the current frame)`, so 50% means
  "halve what full detail costs right now" — a pure function of this frame.
* While over budget, merge the lowest-importance sibling group into its parent.
  Merging is a *structural coarsening*: coverage and all 2.5D statistics are
  preserved (the parent aggregates its children), nothing is deleted and
  nothing is randomized.
* Floors: cells never coarsen past `min_depth`, and cells holding a live track
  never coarsen past `footprint_min_depth` (tracked objects keep resolution
  over static scenery).
* If the floors are reached and the budget is still exceeded, the frame is
  reported `feasible: false` with the measured overage. Graceful degradation
  has honest limits and this is where they are surfaced.

Deterministic: identical input always yields an identical output.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from adaptive_lidar.mapping.map_cell import CELL_BYTES, MapCell, count_internal_nodes
from adaptive_lidar.mapping.projection_2_5d import INTERNAL_NODE_BYTES
from adaptive_lidar.types import resolution_tier

HOT_RATE = 1.0
WARM_RATE = 0.5
COLD_RATE = 0.15


def update_rate(importance: np.ndarray | list[float],
                split_threshold: float = 0.60,
                merge_threshold: float = 0.30) -> np.ndarray:
    """Per-leaf refresh-rate tier from importance (vectorized)."""
    imp = np.asarray(importance, dtype=np.float64).reshape(-1)
    return np.where(imp >= split_threshold, HOT_RATE,
                    np.where(imp >= merge_threshold, WARM_RATE, COLD_RATE)).astype(np.float64)


def leaf_cost(depth: int, point_count: int, rate: float = 1.0) -> float:
    """Cost of one leaf in abstract units (see the module docstring)."""
    return (2.0 ** int(depth)) * (1.0 + float(point_count) / 8.0) * float(rate)


@dataclass
class BudgetReport:
    """Outcome of one enforcement pass — every field is measured."""

    pre_cost: float = 0.0
    post_cost: float = 0.0
    allowed: float = 0.0
    feasible: bool = True
    pre_leaves: int = 0
    post_leaves: int = 0
    post_internal: int = 0
    post_memory_bytes: int = 0
    merged_groups: int = 0
    merges: int = 0
    floor_reached: bool = False
    protected_groups: int = 0
    latency_s: float = 0.0
    budget_pct: float = 100.0
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "pre_cost": round(self.pre_cost, 1),
            "post_cost": round(self.post_cost, 1),
            "allowed": round(self.allowed, 1),
            "feasible": bool(self.feasible),
            "pre_leaves": int(self.pre_leaves),
            "post_leaves": int(self.post_leaves),
            "post_internal": int(self.post_internal),
            "post_memory_bytes": int(self.post_memory_bytes),
            "merged_groups": int(self.merged_groups),
            "floor_reached": bool(self.floor_reached),
            "protected_groups": int(self.protected_groups),
            "budget_pct": self.budget_pct,
            "latency_ms": round(self.latency_s * 1000.0, 3),
            "notes": list(self.notes),
        }


class BudgetManager:
    """Enforces a percentage budget by deterministic, importance-ordered merging."""

    def __init__(self, budget_pct: float = 100.0, min_depth: int = 8,
                 footprint_min_depth: int = 9, max_passes: int = 4,
                 hot_rate: float = HOT_RATE, warm_rate: float = WARM_RATE,
                 cold_rate: float = COLD_RATE,
                 split_threshold: float = 0.60, merge_threshold: float = 0.30):
        if not 0 < budget_pct <= 100:
            raise ValueError("budget_pct must be in (0, 100]")
        self.budget_pct = float(budget_pct)
        self.min_depth = int(min_depth)
        self.footprint_min_depth = int(footprint_min_depth)
        self.max_passes = max(1, int(max_passes))
        self.hot_rate, self.warm_rate, self.cold_rate = hot_rate, warm_rate, cold_rate
        self.split_threshold = float(split_threshold)
        self.merge_threshold = float(merge_threshold)
        self.last_report: BudgetReport | None = None

    # ------------------------------------------------------------------ rates
    def rates(self, importance: np.ndarray) -> np.ndarray:
        return update_rate(importance, self.split_threshold, self.merge_threshold)

    def rate_for(self, importance: float) -> float:
        return float(self.rates(np.array([importance]))[0])

    def total_cost(self, cells: list[MapCell]) -> float:
        """Cost of a whole leaf set (vectorized: one numpy pass, not per cell)."""
        if not cells:
            return 0.0
        n = len(cells)
        depths = np.fromiter((c.depth for c in cells), dtype=np.int64, count=n)
        counts = np.fromiter((c.point_count for c in cells), dtype=np.float64, count=n)
        imps = np.fromiter((c.importance_score for c in cells), dtype=np.float64, count=n)
        return float(self._cost(depths, counts, imps).sum())

    def _cost(self, depths: np.ndarray, counts: np.ndarray,
              importances: np.ndarray) -> np.ndarray:
        """Per-leaf cost in abstract units (vectorized)."""
        rates = update_rate(importances, self.split_threshold, self.merge_threshold)
        return (np.ldexp(1.0, depths) * (1.0 + counts / 8.0) * rates)

    # ------------------------------------------------------------- enforcement
    def enforce(self, cells: list[MapCell], extent_m: float,
                protected: set[tuple[int, int, int]] | None = None,
                budget_pct: float | None = None) -> tuple[list[MapCell], BudgetReport]:
        """Return the coarsened leaf set plus a measured `BudgetReport`.

        protected: (depth, ix, iy) keys of cells holding live tracks; their
        groups are not merged below `footprint_min_depth`.
        """
        t0 = time.perf_counter()
        pct = self.budget_pct if budget_pct is None else float(budget_pct)
        prot: set[tuple[int, int, int]] = set(protected or ())
        pre_cost = self.total_cost(cells)
        allowed = pre_cost * pct / 100.0
        report = BudgetReport(pre_cost=pre_cost, allowed=allowed,
                              pre_leaves=len(cells), budget_pct=pct)

        if pct >= 100.0 or not cells:
            # Nothing to coarsen: the leaf set is returned untouched.
            report.post_cost = pre_cost
            report.post_leaves = len(cells)
            report.post_internal = count_internal_nodes(cells)
            report.post_memory_bytes = (len(cells) * CELL_BYTES
                                        + report.post_internal * INTERNAL_NODE_BYTES)
            report.feasible = True
            report.latency_s = time.perf_counter() - t0
            self.last_report = report
            return list(cells), report

        # Working representation: mutable dicts (Python-speed merging, ~5k cells).
        n_leaves = len(cells)
        depths = np.fromiter((c.depth for c in cells), dtype=np.int64, count=n_leaves)
        counts = np.fromiter((c.point_count for c in cells), dtype=np.float64, count=n_leaves)
        imps_arr = np.fromiter((c.importance_score for c in cells), dtype=np.float64,
                               count=n_leaves)
        cost_arr = self._cost(depths, counts, imps_arr)
        live: list[dict | None] = [
            self._as_dict(c, float(cost_arr[i])) for i, c in enumerate(cells)
        ]
        total = pre_cost
        next_id = max(c.cell_id for c in cells) + 1

        for _ in range(self.max_passes):
            if total <= allowed:
                break
            groups: dict[tuple[int, int, int], list[int]] = {}
            for i, leaf in enumerate(live):
                if leaf is None or leaf["d"] <= self.min_depth:
                    continue
                pk = (leaf["d"] - 1, leaf["ix"] >> 1, leaf["iy"] >> 1)
                groups.setdefault(pk, []).append(i)
            if not groups:
                report.floor_reached = True
                break

            def mean_imp(item):
                pk, members = item
                n = sum(live[i]["n"] for i in members if live[i] is not None)
                if n <= 0:
                    return 0.0
                tot = sum(live[i]["imp"] * live[i]["n"] for i in members if live[i] is not None)
                return tot / n

            for (pd, pix, piy), members in sorted(groups.items(), key=mean_imp):
                if total <= allowed:
                    break
                members = [i for i in members if live[i] is not None]
                if not members:
                    continue
                guarded = any((live[i]["d"], live[i]["ix"], live[i]["iy"]) in prot
                              for i in members)
                if guarded and pd < self.footprint_min_depth:
                    report.protected_groups += 1
                    continue
                if any(prot & {(live[i]["d"], live[i]["ix"], live[i]["iy"])}
                       for i in members):
                    prot.add((pd, pix, piy))
                parent = self._merge_children(
                    [live[i] for i in members], pd, pix, piy, extent_m, next_id)
                next_id += 1
                total += parent["cost"] - sum(live[i]["cost"] for i in members)
                for i in members:
                    live[i] = None
                live.append(parent)
                report.merged_groups += 1

        out: list[MapCell] = []
        for leaf in live:
            if leaf is None:
                continue
            out.append(self._as_cell(leaf, extent_m))
        # Re-index so cell ids stay contiguous and parents/children are valid.
        for new_id, cell in enumerate(out):
            cell.cell_id = new_id
        report.post_leaves = len(out)
        report.merges = report.pre_leaves - report.post_leaves
        report.post_cost = self.total_cost(out)
        report.post_internal = count_internal_nodes(out)
        report.post_memory_bytes = (len(out) * CELL_BYTES
                                    + report.post_internal * INTERNAL_NODE_BYTES)
        report.feasible = report.post_cost <= allowed + 1e-9
        if not report.feasible:
            report.notes.append(
                f"budget {pct:g}% not reachable without violating the depth floor "
                f"(min_depth={self.min_depth}); measured usage is "
                f"{report.post_cost:.0f} vs allowed {allowed:.0f} units")
        report.latency_s = time.perf_counter() - t0
        self.last_report = report
        return out, report

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _as_dict(cell: MapCell, cost: float | None = None) -> dict:
        rate = cell.importance_score  # placeholder; the tier is applied in _cost
        return {
            "d": int(cell.depth), "ix": int(cell.ix), "iy": int(cell.iy),
            "cx": float(cell.center_x), "cy": float(cell.center_y),
            "size": float(cell.size), "n": int(cell.point_count),
            "zmin": float(cell.min_height), "zmax": float(cell.max_height),
            "zmean": float(cell.mean_height), "zvar": float(cell.height_variance),
            "sem": int(cell.semantic_class), "conf": float(cell.semantic_confidence),
            "imp": float(cell.importance_score),
            "rate": float(rate),
            "cost": float(cost) if cost is not None else 0.0,
        }

    def _merge_children(self, members: list[dict], pd: int, pix: int, piy: int,
                        extent_m: float, new_id: int) -> dict:
        """Aggregate children into their parent (2.5D statistics preserved)."""
        n_total = sum(m["n"] for m in members)
        half = extent_m / 2.0
        size = extent_m / float(1 << pd)
        cx = -half + (pix + 0.5) * size
        cy = -half + (piy + 0.5) * size
        w = np.array([m["n"] for m in members], dtype=np.float64)
        w = w / w.sum() if w.sum() > 0 else np.full(len(members), 1.0 / len(members))
        zmean = float(np.dot(w, [m["zmean"] for m in members]))
        second = float(np.dot(w, [m["zvar"] + m["zmean"] ** 2 for m in members]))
        votes: dict[int, float] = {}
        for m in members:
            votes[m["sem"]] = votes.get(m["sem"], 0.0) + m["n"]
        imp = float(np.dot(w, [m["imp"] for m in members]))
        rate = self.rate_for(imp)
        return {
            "d": pd, "ix": pix, "iy": piy, "cx": cx, "cy": cy, "size": size,
            "n": n_total,
            "zmin": min(m["zmin"] for m in members),
            "zmax": max(m["zmax"] for m in members),
            "zmean": zmean,
            "zvar": max(0.0, second - zmean * zmean),
            "sem": max(votes.items(), key=lambda kv: kv[1])[0],
            "conf": float(np.dot(w, [m["conf"] for m in members])),
            "imp": imp, "rate": rate,
            "cost": leaf_cost(pd, n_total, rate),
        }

    @staticmethod
    def _as_cell(leaf: dict, extent_m: float) -> MapCell:
        return MapCell(
            cell_id=0, center_x=leaf["cx"], center_y=leaf["cy"], size=leaf["size"],
            depth=leaf["d"], ix=leaf["ix"], iy=leaf["iy"],
            mean_height=leaf["zmean"], height_variance=leaf["zvar"],
            min_height=leaf["zmin"], max_height=leaf["zmax"], occupancy=1.0,
            point_count=leaf["n"], semantic_class=leaf["sem"],
            semantic_confidence=leaf["conf"], importance_score=leaf["imp"],
        )

    def tier_histogram(self, cells: list[MapCell]) -> dict:
        hist = {"high": 0, "medium": 0, "coarse": 0}
        for c in cells:
            hist[resolution_tier(c.size)] += 1
        return hist
