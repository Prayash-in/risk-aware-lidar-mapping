"""2.5D mapping: cells, statistics, projection and the spatial quadtree."""

from __future__ import annotations

import numpy as np
import pytest

from adaptive_lidar.mapping import (
    MapCell,
    Quadtree,
    accumulate,
    distance_size_field,
    link_hierarchy,
    project_2_5d,
    size_to_depth,
    uniform_target_size,
)
from adaptive_lidar.mapping.projection_2_5d import depth_to_size
from adaptive_lidar.types import resolution_tier

EXTENT = 100.0


def _cloud(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(-49, 49, size=(n, 3)).astype(np.float32)
    sem = rng.integers(0, 9, size=n).astype(np.int32)
    return xyz, sem


# ------------------------------------------------------------------ size law
def test_size_follows_extent_over_two_to_the_depth():
    # §: size = extent / 2**depth — depth 6 and depth 10 must differ.
    assert size_to_depth(EXTENT, 1.5625)[()] == 6
    assert size_to_depth(EXTENT, 0.09765625)[()] == 10
    assert np.isclose(depth_to_size(EXTENT, 6)[()], 1.5625)
    assert np.isclose(depth_to_size(EXTENT, 10)[()], 0.09765625)


def test_resolution_tiers_follow_the_shared_table():
    assert resolution_tier(0.05) == "high"
    assert resolution_tier(0.10) == "high"
    assert resolution_tier(0.20) == "medium"
    assert resolution_tier(0.50) == "coarse"


# --------------------------------------------------------------- statistics
def test_accumulate_matches_manual_statistics():
    group = np.array([0, 0, 1, 1, 1, 2])
    z = np.array([1.0, 3.0, 0.0, 2.0, 4.0, 7.0])
    sem = np.array([1, 1, 6, 6, 6, 4])
    starts = np.array([0, 2, 5])
    s = accumulate(group, z, sem, 3, num_classes=9, group_starts=starts)
    assert list(s.count) == [2, 3, 1]
    assert np.isclose(s.z_mean[0], 2.0) and np.isclose(s.z_var[0], 1.0)
    assert np.isclose(s.z_mean[1], 2.0) and np.isclose(s.z_var[1], 8.0 / 3.0)
    assert s.z_min[2] == 7.0 and s.z_max[2] == 7.0
    assert list(s.semantic) == [1, 6, 4]
    assert np.isclose(s.confidence[1], 1.0)


# --------------------------------------------------------------- projection
def test_uniform_projection_uses_exactly_the_uniform_size():
    xyz, sem = _cloud()
    res = project_2_5d(xyz, sem, EXTENT, target_size=uniform_target_size(0.10, len(xyz)),
                       max_depth=11)
    assert res.num_leaves > 0
    assert all(c.size == pytest.approx(depth_to_size(EXTENT, 10)[()], rel=1e-3)
               for c in res.cells)
    assert res.stats()["min_cell_size_m"] == res.stats()["max_cell_size_m"]


def test_distance_projection_refines_near_and_coarsens_far():
    xyz, sem = _cloud()
    res = project_2_5d(xyz, sem, EXTENT,
                       target_field=distance_size_field(0.05, 0.50, 0.12), max_depth=11)
    near = [c for c in res.cells if np.hypot(c.center_x, c.center_y) < 5.0]
    far = [c for c in res.cells if np.hypot(c.center_x, c.center_y) > 30.0]
    assert near and far
    assert min(c.size for c in near) < min(c.size for c in far)
    assert res.stats()["leaf_count"] < len(xyz)   # aggregation actually happens


def test_projection_cells_carry_2_5d_statistics():
    xyz = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 2.0]], np.float32)
    sem = np.array([4, 4], np.int32)
    res = project_2_5d(xyz, sem, EXTENT, target_size=uniform_target_size(0.10, 2),
                       max_depth=11)
    assert len(res.cells) == 1
    cell = res.cells[0]
    assert cell.point_count == 2
    assert cell.min_height == 0.0 and cell.max_height == 2.0
    assert cell.height_variance > 0.0
    assert cell.occupancy == 1.0


def test_projection_aggregates_points_into_fewer_cells():
    xyz = np.zeros((500, 3), np.float32)
    xyz[:, 0] = np.linspace(0.0, 0.5, 500)
    sem = np.ones(500, np.int32)
    res = project_2_5d(xyz, sem, EXTENT, target_size=uniform_target_size(1.0, 500),
                       max_depth=11)
    assert res.num_leaves < 10
    assert res.statistics.count.sum() == 500


def test_projection_respects_depth_bounds():
    xyz, sem = _cloud()
    res = project_2_5d(xyz, sem, EXTENT, target_size=uniform_target_size(0.001, len(xyz)),
                       max_depth=9, min_depth=4)
    assert all(4 <= c.depth <= 9 for c in res.cells)


def test_projection_drops_out_of_bounds_points():
    xyz = np.array([[-500.0, 0.0, 0.0], [1.0, 1.0, 0.0]], np.float32)
    sem = np.array([1, 1], np.int32)
    res = project_2_5d(xyz, sem, EXTENT, target_size=uniform_target_size(0.2, 2))
    assert res.num_leaves == 1
    assert res.cells[0].point_count == 1


def test_projection_on_empty_cloud():
    res = project_2_5d(np.zeros((0, 3)), np.zeros(0, np.int32), EXTENT,
                       target_size=uniform_target_size(0.2, 0))
    assert res.num_leaves == 0 and res.memory_bytes() == 0


def test_importance_and_factors_reach_the_cells():
    # The budget tiers read `importance_score`; it must not be silently lost.
    xyz = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [10.0, 10.0, 0.0]], np.float32)
    sem = np.array([1, 1, 6], np.int32)
    imp = np.array([1.0, 0.9, 0.1])
    res = project_2_5d(xyz, sem, EXTENT,
                       target_size=uniform_target_size(0.2, 3),
                       importance=imp,
                       factors={"range": np.array([1.0, 1.0, 0.2]),
                                "bubble": np.array([0.5, 0.5, 0.0])})
    assert len(res.cells) == 2
    near = min(res.cells, key=lambda c: c.center_x)
    far = max(res.cells, key=lambda c: c.center_x)
    assert near.importance_score == pytest.approx(0.95, abs=1e-3)
    assert far.importance_score == pytest.approx(0.10, abs=1e-3)
    assert near.range_score == pytest.approx(1.0, abs=1e-3)
    assert near.bubble_score == pytest.approx(0.5, abs=1e-3)
    assert res.statistics.importance is not None


def test_target_size_length_is_validated():
    with pytest.raises(ValueError):
        project_2_5d(np.zeros((3, 3)), np.zeros(3, np.int32), EXTENT,
                     target_size=np.zeros(2))


def test_structure_refinement_pass_splits_structured_cells():
    # Two heights in the same cell: real 3D structure -> one extra level.
    rng = np.random.default_rng(3)
    base = rng.uniform(-1, 1, size=(600, 2))
    xyz = np.hstack([base, rng.uniform(0, 0.01, (600, 1))]).astype(np.float32)
    xyz[::2, 2] = 1.5
    sem = np.ones(600, np.int32)
    plain = project_2_5d(xyz, sem, EXTENT, target_size=uniform_target_size(1.0, 600),
                         max_depth=11)
    refined = project_2_5d(xyz, sem, EXTENT, target_size=uniform_target_size(1.0, 600),
                           max_depth=11, refine_structure=True, z_var_threshold=0.05,
                           min_points_for_refine=4)
    assert refined.num_leaves > plain.num_leaves


# ---------------------------------------------------------------- hierarchy
def test_leaves_link_to_parents_when_a_coarser_leaf_exists():
    coarse = MapCell(cell_id=0, center_x=0.0, center_y=0.0, size=2.0, depth=5,
                     ix=0, iy=0, point_count=10)
    child = MapCell(cell_id=1, center_x=-0.5, center_y=-0.5, size=1.0, depth=6,
                    ix=0, iy=0, point_count=4)
    internal = link_hierarchy([coarse, child])
    assert coarse.is_leaf is False
    assert list(coarse.children).count(1) == 1
    assert child.parent == 0
    # Depths 0..4 hold no data of their own -> 5 structural internal nodes.
    assert internal == 5


def test_internal_node_count_is_exact():
    # A depth-9 leaf implies 9 ancestors that hold no data of their own.
    cell = MapCell(cell_id=0, center_x=0.0, center_y=0.0,
                   size=depth_to_size(EXTENT, 9)[()], depth=9, ix=3, iy=7)
    assert link_hierarchy([cell]) == 9


def test_parent_key_walks_one_level_up():
    cell = MapCell(cell_id=0, center_x=0.0, center_y=0.0, size=1.0, depth=6, ix=5, iy=9)
    assert cell.parent_key == (5, 2, 4)


# ----------------------------------------------------------------- quadtree
def _tiny_tree():
    cells = project_2_5d(*_cloud(800, seed=1), EXTENT,
                        target_size=uniform_target_size(0.5, 800), max_depth=11).cells
    return Quadtree.from_cells(cells, EXTENT)


def test_quadtree_find_leaf_returns_the_occupied_cell():
    tree = _tiny_tree()
    cell = tree.get_leaves()[0]
    found = tree.find_leaf(cell.center_x, cell.center_y)
    assert found is not None and found.key == cell.key


def test_quadtree_find_leaf_prefers_finest_when_hierarchy_overlaps():
    # During coarsening a parent may momentarily coexist with finer children;
    # find_leaf must return the most detailed cell, not the first (coarse) one.
    from adaptive_lidar.mapping.map_cell import MapCell
    tree = _tiny_tree()
    cell = tree.get_leaves()[0]
    d, ix, iy = cell.key
    parent = MapCell(cell_id=999, center_x=cell.center_x, center_y=cell.center_y,
                     size=cell.size * 2.0, depth=d - 1 if d > 0 else 0,
                     ix=ix // 2, iy=iy // 2)
    tree.cells[parent.key] = parent
    found = tree.find_leaf(cell.center_x, cell.center_y)
    assert found is not None and found.key == cell.key


def test_quadtree_query_region_returns_intersecting_footprints():
    tree = _tiny_tree()
    hits = tree.query_region(-2.0, -2.0, 2.0, 2.0)
    assert hits, "a populated region must return cells"
    for c in hits:
        x0, y0, x1, y1 = c.bounds          # footprint must overlap the query
        assert x1 >= -2.0 and x0 <= 2.0 and y1 >= -2.0 and y0 <= 2.0
    # A far-away region with no data must return nothing.
    assert tree.query_region(49.0, 49.0, 50.0, 50.0) == []


def test_quadtree_split_creates_four_children_and_conserves_size():
    tree = _tiny_tree()
    cell = tree.get_leaves()[0]
    before = tree.num_nodes()
    res = tree.split(cell, min_size=0.05, max_depth=11)
    assert res.ok and len(res.children) == 4
    assert tree.num_nodes() == before + 3
    for cid in res.children:
        child = next(c for c in tree.get_leaves() if c.cell_id == cid)
        assert np.isclose(child.size, cell.size / 2.0)
        assert child.depth == cell.depth + 1
    assert cell.key not in tree.cells


def test_quadtree_split_respects_floors_and_reports_reason():
    tree = _tiny_tree()
    cell = tree.get_leaves()[0]
    res = tree.split(cell, min_size=cell.size, max_depth=cell.depth)
    assert not res.ok and res.reason in ("at_min_size", "at_max_depth")


def test_quadtree_merge_aggregates_children_into_the_parent():
    tree = _tiny_tree()
    cell = tree.get_leaves()[0]
    res = tree.split(cell, min_size=0.05, max_depth=11)
    merged = tree.merge(cell.key)
    assert merged is not None
    assert merged.key == cell.key
    assert np.isclose(merged.size, cell.size)
    assert all(k not in tree.cells for k in
               [(cell.depth + 1, cell.ix * 2 + dx, cell.iy * 2 + dy)
                for dx in (0, 1) for dy in (0, 1)])
    assert res.ok


def test_quadtree_merge_refuses_incomplete_groups():
    tree = _tiny_tree()
    assert tree.merge((0, 0, 0)) is None


def test_quadtree_states_are_reported():
    tree = _tiny_tree()
    tree.mark(tree.get_leaves()[0].key, "split_pending")
    states = tree.states()
    assert states["split_pending"] == 1
    assert sum(states.values()) == tree.num_nodes()
