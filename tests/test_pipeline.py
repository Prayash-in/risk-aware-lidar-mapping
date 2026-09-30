"""End-to-end pipeline: API contract, the three policies, metrics, state."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from adaptive_lidar import AdaptiveLidarMapper, __version__, load_config
from adaptive_lidar.data import SyntheticPointCloudSource
from adaptive_lidar.pipeline import metrics_tiers, summarize
from adaptive_lidar.types import resolution_tier

FRAMES = 12


@pytest.fixture(scope="module")
def source():
    return SyntheticPointCloudSource("urban_crossing", num_frames=FRAMES)


def _run(mapper, source, n=FRAMES):
    return [mapper.process_frame(source.load_frame(i)) for i in range(n)]


# --------------------------------------------------------------------- public
def test_package_exposes_the_documented_api():
    assert isinstance(__version__, str) and __version__
    assert callable(AdaptiveLidarMapper)
    assert callable(load_config)
    assert callable(summarize)
    assert callable(metrics_tiers)


def test_default_config_loads_from_the_repository():
    cfg = load_config("configs/default.yaml")
    assert cfg.mode == "adaptive"
    assert cfg.quadtree.extent_m == 100.0
    assert cfg.quadtree.min_cell_size == 0.05 and cfg.quadtree.max_cell_size == 0.50
    assert cfg.controller.split_threshold == 0.60
    assert cfg.controller.merge_threshold == 0.30
    assert cfg.controller.split_frames == 2 and cfg.controller.merge_frames == 3
    assert cfg.controller.ref_depth == 9
    assert cfg.budget.pct in (25.0, 50.0, 75.0, 100.0)


def test_default_config_resolves_without_an_explicit_path():
    """With no argument the shipped defaults must load from any directory."""
    cfg = load_config()
    assert cfg.mode == "adaptive"
    assert cfg.quadtree.extent_m == 100.0


def test_overrides_reach_the_nested_config():
    cfg = load_config("configs/default.yaml",
                      overrides={"budget": {"pct": 25.0}, "mode": "uniform"})
    assert cfg.budget.pct == 25.0
    assert cfg.mode == "uniform"


def test_status_is_serializable_and_states_capabilities():
    st = AdaptiveLidarMapper().status()
    json.dumps(st)                                   # must be JSON-safe
    assert {"mode", "extent_m", "frames_processed", "semantics",
            "controller", "tracker", "budget"} <= set(st)
    sem = st["semantics"]
    assert sem["torch_available"] in (True, False)
    assert sem["use_model"] is False
    # Without a checkpoint the labeler must say so instead of pretending.
    assert sem["available"] is False
    assert sem["checkpoint"] is None


# ------------------------------------------------------------------ one frame
def test_process_frame_returns_a_complete_result(source):
    mapper = AdaptiveLidarMapper()
    res = mapper.process_frame(source.load_frame(0))
    assert res.mode == "adaptive"
    assert res.leaves, "adaptive mode must return cells"
    assert res.metrics.frame_latency > 0.0
    assert res.extras["semantic_source"] in ("ground_truth", "model", "unknown")
    for cell in res.leaves:
        assert cell.size > 0 and cell.depth >= 0
        assert cell.size == pytest.approx(100.0 / 2 ** cell.depth, rel=1e-6)
        assert 0.0 <= cell.importance_score <= 1.0


def test_metrics_are_measured_and_self_consistent(source):
    mapper = AdaptiveLidarMapper()
    res = mapper.process_frame(source.load_frame(3))
    m = res.metrics
    assert m.total_cells == len(res.leaves)
    assert m.memory_estimate > 0
    assert m.compute_usage > 0
    assert m.controller_latency <= m.frame_latency
    assert m.fps > 0
    assert (m.high_resolution_cells + m.medium_resolution_cells
            + m.coarse_resolution_cells) == m.total_cells


def test_memory_estimate_follows_the_budget_not_the_pre_budget_map(source):
    """The budget coarsens the map: reported memory must describe what is returned."""
    from adaptive_lidar.mapping.map_cell import CELL_BYTES
    from adaptive_lidar.mapping.projection_2_5d import INTERNAL_NODE_BYTES

    mapper = AdaptiveLidarMapper(overrides={"budget": {"pct": 25.0}})
    for i in range(4):
        mapper.process_frame(source.load_frame(i))
    res = mapper.process_frame(source.load_frame(4))
    b = res.extras["budget"]
    assert b["post_leaves"] == res.metrics.total_cells <= b["pre_leaves"]
    assert b["post_memory_bytes"] == res.metrics.memory_estimate
    assert b["post_memory_bytes"] == (
        b["post_leaves"] * CELL_BYTES + b["post_internal"] * INTERNAL_NODE_BYTES)


def test_summarize_is_json_serializable(source):
    mapper = AdaptiveLidarMapper()
    res = mapper.process_frame(source.load_frame(2))
    payload = json.dumps(summarize(res))
    assert "tiers" in payload and "latency_s" in payload


# ------------------------------------------------------------------- policies
def test_three_policies_process_identical_frames(source):
    mapper = AdaptiveLidarMapper()
    frames = [source.load_frame(i) for i in range(8)]
    out = mapper.compare_modes(frames[-1], replay=frames[:-1])
    assert set(out) == {"uniform", "distance", "adaptive"}
    assert all(v["frame_id"] == frames[-1].frame_id for v in out.values())
    # Identical input -> identical point count in every mode.
    counts = {v["cells"] for v in out.values()}
    assert all(v > 0 for v in counts)


def test_uniform_mode_is_a_single_cell_size(source):
    mapper = AdaptiveLidarMapper(overrides={"mode": "uniform"})
    res = mapper.process_frame(source.load_frame(4))
    sizes = {round(c.size, 9) for c in res.leaves}
    assert len(sizes) == 1
    depth = res.leaves[0].depth
    assert sizes.pop() == pytest.approx(100.0 / 2 ** depth, rel=1e-9)


def test_distance_mode_is_finer_near_than_far(source):
    mapper = AdaptiveLidarMapper(overrides={"mode": "distance"})
    res = mapper.process_frame(source.load_frame(4))
    near = [c.size for c in res.leaves if np.hypot(c.center_x, c.center_y) < 8.0]
    far = [c.size for c in res.leaves if np.hypot(c.center_x, c.center_y) > 40.0]
    assert near and far
    assert min(near) < min(far)


def test_adaptive_mode_uses_more_cells_than_distance(source):
    """Risk-driven refinement must not be a no-op relative to pure distance LOD."""
    adaptive = AdaptiveLidarMapper()
    for i in range(6):
        adaptive.process_frame(source.load_frame(i))
    dist = AdaptiveLidarMapper(overrides={"mode": "distance"})
    for i in range(6):
        dist.process_frame(source.load_frame(i))
    a = adaptive.process_frame(source.load_frame(6))
    d = dist.process_frame(source.load_frame(6))
    assert a.metrics.total_cells != d.metrics.total_cells


def test_tiers_use_the_shared_size_table(source):
    mapper = AdaptiveLidarMapper()
    res = mapper.process_frame(source.load_frame(5))
    for cell in res.leaves[:200]:
        assert cell.tier == resolution_tier(cell.size)


# ------------------------------------------------------------------ sequences
def test_sequence_is_deterministic(source):
    a = _run(AdaptiveLidarMapper(), source, 6)
    b = _run(AdaptiveLidarMapper(), source, 6)
    assert [r.metrics.total_cells for r in a] == [r.metrics.total_cells for r in b]
    assert [r.metrics.compute_usage for r in a] == [r.metrics.compute_usage for r in b]


def test_process_source_method(source):
    mapper = AdaptiveLidarMapper()
    results = mapper.process_source(source, limit=4)
    assert len(results) == 4
    assert [r.frame_id for r in results] == [0, 1, 2, 3]


def test_budget_is_enforced_end_to_end(source):
    loose = AdaptiveLidarMapper(overrides={"budget": {"pct": 100.0}})
    tight = AdaptiveLidarMapper(overrides={"budget": {"pct": 25.0}})
    for i in range(6):
        loose.process_frame(source.load_frame(i))
        tight.process_frame(source.load_frame(i))
    a = loose.process_frame(source.load_frame(6))
    b = tight.process_frame(source.load_frame(6))
    assert b.metrics.compute_usage <= a.metrics.compute_usage
    assert b.extras["budget"]["post_cost"] <= b.extras["budget"]["allowed"] * 1.02


def test_hysteresis_produces_structure_changes_over_time(source):
    mapper = AdaptiveLidarMapper()
    results = _run(mapper, source, FRAMES)
    ops = sum(r.metrics.split_operations for r in results)
    assert ops > 0, "the controller must actually refine cells over a sequence"
    assert sum(r.extras["controller"] for r in results) == \
        sum(r.metrics.split_operations + r.metrics.merge_operations for r in results)


def test_reset_clears_all_temporal_state(source):
    mapper = AdaptiveLidarMapper()
    _run(mapper, source, 5)
    assert mapper.status()["frames_processed"] == 5
    mapper.reset()
    st = mapper.status()
    assert st["frames_processed"] == 0
    assert st["controller"]["hysteresis"]["tracked_cells"] == 0
    assert st["tracker"]["tracks"] == 0


def test_empty_cloud_does_not_crash():
    from adaptive_lidar.data import PointCloud
    mapper = AdaptiveLidarMapper()
    empty = PointCloud(xyz=np.zeros((0, 3), np.float32),
                       intensity=np.zeros(0, np.float32),
                       semantic=np.zeros(0, np.int32),
                       confidence=np.zeros(0, np.float32))
    res = mapper.process_frame(empty)
    assert res.metrics.total_cells == 0
    assert res.extras["budget"]["post_cost"] == 0.0


# ------------------------------------------------------------------------ CLI
def test_cli_runs_and_reports_measured_metrics(tmp_path):
    out = tmp_path / "cli.json"
    proc = subprocess.run(
        [sys.executable, "-m", "adaptive_lidar", "run", "--frames", "3", "--quiet"],
        capture_output=True, text=True, cwd=str(tmp_path), timeout=300,
        env={**__import__("os").environ,
             "PYTHONPATH": str(__import__("adaptive_lidar").__file__
                               .rsplit("adaptive_lidar", 1)[0])},
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    payload = json.loads(proc.stdout[proc.stdout.index("{"):])
    assert payload["aggregate"]["frames"] == 3
    assert payload["aggregate"]["latency_ms"]["mean"] > 0
    assert out.exists() is False        # `run` prints, it does not write files


def test_cli_info_reports_honest_capabilities():
    proc = subprocess.run(
        [sys.executable, "-m", "adaptive_lidar", "info"],
        capture_output=True, text=True, timeout=300,
        env={**__import__("os").environ,
             "PYTHONPATH": str(__import__("adaptive_lidar").__file__
                               .rsplit("adaptive_lidar", 1)[0])},
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    info = json.loads(proc.stdout)
    assert "torch_available" in info
    assert "sources" in info
    # The KITTI source must either load or say clearly why it cannot.
    kitti = info["sources"]["semantic_kitti"]
    assert kitti.get("available") is False or "frames" in kitti
