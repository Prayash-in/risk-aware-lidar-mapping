"""Compute budget: cost model, deterministic importance-ordered coarsening."""

from __future__ import annotations

import numpy as np
import pytest

from adaptive_lidar.controller.budget_manager import (
    COLD_RATE,
    HOT_RATE,
    WARM_RATE,
    BudgetManager,
    leaf_cost,
    update_rate,
)
from adaptive_lidar.mapping import project_2_5d, uniform_target_size

EXTENT = 100.0


def _cells(n=3000, seed=0, importance=None, size=0.10):
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(-49, 49, (n, 3)).astype(np.float32)
    sem = rng.integers(0, 9, n).astype(np.int32)
    res = project_2_5d(xyz, sem, EXTENT,
                       target_size=uniform_target_size(size, n), max_depth=11,
                       importance=importance)
    return res.cells


# ------------------------------------------------------------------ cost model
def test_cost_formula_matches_the_documented_law():
    # cost = 2**depth * (1 + n/8) * rate
    assert leaf_cost(9, 8, 1.0) == pytest.approx(512 * 2.0)
    assert leaf_cost(10, 0, 0.5) == pytest.approx(1024 * 1.0 * 0.5)
    assert leaf_cost(3, 40, 0.15) == pytest.approx(8 * 6.0 * 0.15)


def test_cost_grows_with_depth_and_point_count():
    assert leaf_cost(10, 4) > leaf_cost(9, 4)
    assert leaf_cost(9, 40) > leaf_cost(9, 4)


def test_rate_tiers_follow_importance():
    r = update_rate([0.9, 0.6, 0.45, 0.3, 0.1, 0.0])
    assert list(r) == [HOT_RATE, HOT_RATE, WARM_RATE, WARM_RATE, COLD_RATE, COLD_RATE]


def test_total_cost_is_the_sum_of_the_per_leaf_costs():
    cells = _cells(n=1500, importance=np.full(1500, 0.8))
    m = BudgetManager()
    manual = sum(leaf_cost(c.depth, c.point_count, m.rate_for(c.importance_score))
                 for c in cells)
    assert m.total_cost(cells) == pytest.approx(manual, rel=1e-9)


def test_total_cost_of_an_empty_map_is_zero():
    assert BudgetManager().total_cost([]) == 0.0


def test_budget_pct_is_validated():
    with pytest.raises(ValueError):
        BudgetManager(budget_pct=0)
    with pytest.raises(ValueError):
        BudgetManager(budget_pct=101)


# ----------------------------------------------------------------- enforcement
def test_full_budget_changes_nothing():
    cells = _cells()
    m = BudgetManager(budget_pct=100)
    out, rep = m.enforce(cells, EXTENT)
    assert out == cells
    assert rep.pre_cost == rep.post_cost
    assert rep.merged_groups == 0
    assert rep.feasible and not rep.floor_reached
    assert rep.pre_leaves == rep.post_leaves == len(cells)


@pytest.mark.parametrize("pct", [75.0, 50.0, 25.0])
def test_tighter_budgets_reduce_cost_monotonically(pct):
    base = _cells(n=4000, importance=np.linspace(0.0, 1.0, 4000))
    pre = BudgetManager().total_cost(base)
    out, rep = BudgetManager(budget_pct=pct).enforce(base, EXTENT)
    assert rep.post_cost < pre
    assert rep.post_cost <= rep.allowed * 1.02     # small tolerance for group granularity
    assert rep.feasible
    assert rep.merged_groups > 0
    assert len(out) < len(base)


def test_enforcement_never_deletes_a_leaf():
    cells = _cells(n=2500, importance=np.full(2500, 0.5))
    out, rep = BudgetManager(budget_pct=25).enforce(cells, EXTENT)
    assert rep.post_leaves == len(out)
    assert out                                   # non-empty
    assert sum(c.point_count for c in out) == sum(c.point_count for c in cells)


def test_merging_preserves_the_2_5d_statistics():
    cells = _cells(n=2000, importance=np.full(2000, 0.4))
    out, _ = BudgetManager(budget_pct=40).enforce(cells, EXTENT)
    n_before = sum(c.point_count for c in cells)
    n_after = sum(c.point_count for c in out)
    assert n_after == n_before
    # Every returned cell must still be a valid, physically sized quadtree cell.
    for c in out:
        assert c.size == pytest.approx(EXTENT / 2 ** c.depth, rel=1e-6)
        assert c.is_leaf
        assert c.parent == -1 or c.parent >= 0     # ids re-indexed, links consistent
    ids = {c.cell_id for c in out}
    assert ids == set(range(len(out)))


def test_enforcement_is_deterministic():
    cells = _cells(n=2000, seed=5, importance=np.linspace(0, 1, 2000))
    a, ra = BudgetManager(budget_pct=50).enforce(cells, EXTENT)
    b, rb = BudgetManager(budget_pct=50).enforce(cells, EXTENT)
    assert [c.key for c in a] == [c.key for c in b]
    assert ra.post_cost == rb.post_cost


def test_importance_order_means_hot_regions_are_coarsened_last():
    """Within one map, the budget must take detail from the cold field first."""
    n = 3000
    imp = np.full(n, 0.05)                        # cold everywhere ...
    xyz = np.zeros((n, 3), dtype=np.float32)
    xyz[:400, 0] = np.linspace(-1, 1, 400)         # ... except a high-risk patch
    xyz[400:, 0] = np.linspace(-45, 45, n - 400)
    imp[:400] = 0.95
    cells = project_2_5d(xyz, np.ones(n, np.int32), EXTENT,
                         target_size=uniform_target_size(0.09765625, n),
                         max_depth=11, importance=imp).cells
    out, rep = BudgetManager(budget_pct=50).enforce(cells, EXTENT)
    assert rep.feasible
    assert rep.merged_groups > 0

    near = [c.depth for c in out if abs(c.center_x) < 2.0]
    far = [c.depth for c in out if abs(c.center_x) > 30.0]
    assert near and far
    mean_near = sum(near) / len(near)
    mean_far = sum(far) / len(far)
    assert mean_near > mean_far, (
        f"risky patch (mean depth {mean_near:.2f}) must stay finer than the "
        f"cold field (mean depth {mean_far:.2f})")


def test_rate_tier_makes_risky_leaves_expensive():
    """Cost accounting must reflect the multi-rate tier, not a flat rate."""
    cells = _cells(n=1200)
    for c in cells:
        c.importance_score = 0.9 if c.center_x < 0 else 0.1
    m = BudgetManager()
    hot = sum(leaf_cost(c.depth, c.point_count, HOT_RATE) for c in cells if c.center_x < 0)
    cold = sum(leaf_cost(c.depth, c.point_count, COLD_RATE) for c in cells if c.center_x >= 0)
    assert m.total_cost(cells) == pytest.approx(hot + cold, rel=1e-9)


def test_protected_footprint_is_held_at_the_footprint_floor():
    """A tracked footprint may be coarsened down to the floor, never past it."""
    cells = _cells(n=1500, importance=np.full(1500, 0.1))
    m = BudgetManager(budget_pct=25, min_depth=8, footprint_min_depth=9)
    target = max(cells, key=lambda c: c.depth)
    guard = {(target.depth, target.ix, target.iy)}
    out, rep = m.enforce(cells, EXTENT, protected=guard)
    assert rep.protected_groups > 0
    floor_key = (m.footprint_min_depth, target.ix >> (target.depth - m.footprint_min_depth),
                 target.iy >> (target.depth - m.footprint_min_depth))
    surviving = {c.key for c in out}
    assert target.key in surviving or floor_key in surviving
    # Nothing below the footprint floor may exist on that footprint.
    assert not any(k[0] < m.footprint_min_depth
                   and k[1] >> (m.footprint_min_depth - k[0]) == floor_key[1]
                   and k[2] >> (m.footprint_min_depth - k[0]) == floor_key[2]
                   for k in surviving)


def test_infeasible_budget_reports_the_floor_honestly():
    # A budget this small cannot be met without violating the depth floor.
    cells = _cells(n=800, importance=np.full(800, 0.1))
    out, rep = BudgetManager(budget_pct=25, min_depth=10).enforce(cells, EXTENT)
    assert rep.feasible is False
    assert rep.notes and "min_depth" in rep.notes[0]
    assert rep.post_cost > rep.allowed


def test_report_is_serializable_and_documented():
    cells = _cells(n=800)
    _, rep = BudgetManager(budget_pct=60).enforce(cells, EXTENT)
    d = rep.to_dict()
    for key in ("pre_cost", "post_cost", "allowed", "feasible", "pre_leaves",
                "post_leaves", "merged_groups", "floor_reached", "budget_pct"):
        assert key in d
    assert d["budget_pct"] == 60
    assert rep.latency_s >= 0.0


def test_per_call_budget_overrides_the_manager_default():
    cells = _cells(n=1000, importance=np.full(1000, 0.5))
    out, rep = BudgetManager(budget_pct=100).enforce(cells, EXTENT, budget_pct=50)
    assert rep.budget_pct == 50
    assert rep.merged_groups > 0
