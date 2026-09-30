"""RandLA-Net helpers that must work without torch, plus optional network tests."""

from __future__ import annotations

import numpy as np
import pytest

from adaptive_lidar.perception.randlanet.model import (
    DEFAULT_CONFIG,
    build_graph,
    is_torch_available,
    knn_graph,
    point_features,
    subsample_indices,
)

NEEDS_TORCH = pytest.mark.skipif(
    not is_torch_available(), reason="torch is an optional dependency"
)


def _pts(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-25.0, 25.0, (n, 3)).astype(np.float32)


# ---------------------------------------------------------------- point_features
def test_point_features_has_d_in_columns():
    pts = _pts(40)
    f = point_features(pts, np.ones(40, np.float32))
    assert f.shape == (40, DEFAULT_CONFIG["d_in"])


def test_point_features_true_channel_order():
    xyz = np.array([[1.0, 2.0, 3.0], [-1.0, -2.0, 3.0]], np.float32)
    f = point_features(xyz, np.array([0.5, 0.5], np.float32))
    np.testing.assert_allclose(f[:, :3], xyz)
    np.testing.assert_allclose(f[:, 3], [0.5, 0.5])
    expected_range = np.hypot(xyz[:, 0], xyz[:, 1])
    np.testing.assert_allclose(f[:, 4], expected_range)
    # channel 5 is the local height above/below the frame mean z
    np.testing.assert_allclose(f[:, 5], xyz[:, 2] - xyz[:, 2].mean(),
                               rtol=1e-6)


def test_point_features_default_intensity_is_one():
    f = point_features(_pts(8))
    np.testing.assert_allclose(f[:, 3], 1.0)


def test_point_features_extra_channels_are_appended():
    extra = np.arange(8, dtype=np.float32).reshape(8, 1)
    f = point_features(_pts(8), extra=extra)
    assert f.shape == (8, DEFAULT_CONFIG["d_in"] + 1)
    np.testing.assert_allclose(f[:, -1], extra[:, 0])


# -------------------------------------------------------------------- knn_graph
def test_knn_graph_shape_and_self_first():
    pts = _pts(50)
    idx = knn_graph(pts, 8)
    assert idx.shape == (50, 8)
    assert (idx[:, 0] == np.arange(50)).all()


def test_knn_graph_k_capped_by_n():
    pts = _pts(4)
    idx = knn_graph(pts, 16)
    assert idx.shape == (4, 4)
    for row in idx.tolist():
        assert sorted(row) == [0, 1, 2, 3]


def test_knn_graph_neighbours_are_sorted_by_distance():
    pts = _pts(60)
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    idx = knn_graph(pts, 7)
    def row_dist(i):
        return d[i, idx[i]]
    for i in range(60):
        assert row_dist(i)[0] == pytest.approx(0.0, abs=1e-4)
        assert (np.diff(row_dist(i)) >= -1e-4).all()


# ----------------------------------------------------------------- subsampling
def test_subsample_indices_deterministic_and_subset():
    a = subsample_indices(100, 4, np.random.default_rng(0))
    b = subsample_indices(100, 4, np.random.default_rng(0))
    np.testing.assert_array_equal(a, b)
    assert a.shape[0] == 25
    assert len(set(a.tolist())) == 25
    assert (a >= 0).all() and (a < 100).all()


def test_subsample_indices_keeps_at_least_one_and_respects_n():
    assert subsample_indices(0, 4).shape[0] == 0
    assert subsample_indices(1, 4).shape[0] == 1
    assert subsample_indices(10, 4).shape[0] == 2
    np.testing.assert_array_equal(subsample_indices(16, 1), np.arange(16))


# -------------------------------------------------------------------- build_graph
def check_graph(graph, n, cfg):
    levels = len(cfg["encoder_dims"])
    assert set(graph) == {
        f"knn_{i}" for i in range(levels)
    } | {f"sub_{i}" for i in range(levels - 1)}
    n_prev = n
    for i in range(levels - 1):
        n_next = graph[f"sub_{i}"].shape[0]
        assert graph[f"knn_{i}"].shape == (n_prev, cfg["k_neighbors"][i])
        assert 0 < n_next < n_prev
        n_prev = n_next
    assert graph[f"knn_{levels - 1}"].shape[0] == n_prev
    assert graph[f"knn_{levels - 1}"].shape[1] == cfg["k_neighbors"][-1]


def test_build_graph_matches_config_shapes():
    n = 2000
    graph = build_graph(_pts(n), DEFAULT_CONFIG)
    check_graph(graph, n, DEFAULT_CONFIG)
    # reduction follows the subsampling schedule: 2000 -> 500 -> 125 -> 31
    assert [g.shape[0] for g in (graph["sub_0"], graph["sub_1"], graph["sub_2"])]\
        == [500, 125, 31]
    # the final encoder level keeps N3 points (no further sampling below it)
    assert graph["knn_3"].shape[0] == graph["sub_2"].shape[0]


def test_build_graph_knn_matches_direct_construction():
    pts = _pts(300)
    graph = build_graph(pts, DEFAULT_CONFIG)
    np.testing.assert_array_equal(graph["knn_0"], knn_graph(pts, DEFAULT_CONFIG["k_neighbors"][0]))


def test_graph_indices_are_unique_and_in_range():
    n = 500
    graph = build_graph(_pts(n), DEFAULT_CONFIG)
    for key in ("sub_0", "sub_1", "sub_2"):
        keep = graph[key]
        assert len(set(keep.tolist())) == len(keep)
        assert (keep >= 0).all() and (keep < n).all()


# --------------------------------------------------------- network (optional)
@NEEDS_TORCH
def test_module_passes_a_forward_shape():
    import torch
    from adaptive_lidar.perception.randlanet.model import randla_net_module
    torch.manual_seed(0)
    model = randla_net_module(DEFAULT_CONFIG).eval()
    n = 1024
    pts = _pts(n)
    graph = build_graph(pts, DEFAULT_CONFIG)
    feats = torch.from_numpy(point_features(pts).T[None]).float()
    tg = {
        k: (torch.from_numpy(v.T[None]).long()
            if k.startswith("knn_") else torch.from_numpy(v[None]).long())
        for k, v in graph.items()
    }
    with torch.no_grad():
        out = model(feats, tg)
    assert out.shape == (1, DEFAULT_CONFIG["d_out"], n)


@NEEDS_TORCH
def test_module_default_config_none_is_equivalent():
    from adaptive_lidar.perception.randlanet.model import randla_net_module
    a = randla_net_module(None)
    b = randla_net_module(DEFAULT_CONFIG)
    assert set(dict(a.named_parameters())) == set(dict(b.named_parameters()))