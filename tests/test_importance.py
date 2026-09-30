"""Importance factors, ablations and the risk/prediction bubble laws."""

from __future__ import annotations

import numpy as np
import pytest

from adaptive_lidar.controller.importance import (
    DEFAULT_SEMANTIC_IMPORTANCE,
    ImportanceScorer,
)
from adaptive_lidar.prediction import (
    DEFAULT_R_MIN,
    PredictionConfig,
    bubble_radius,
    constant_velocity_path,
    corridor_length,
    path_endpoints,
)
from adaptive_lidar.types import TrackedObject


def _track(label="vehicle", x=5.0, y=0.0, vx=8.0, vy=0.0, radius=None, path=None):
    cfg = PredictionConfig()
    return TrackedObject(
        object_id=1, label=label, x=x, y=y, vx=vx, vy=vy, confidence=0.9,
        timestamp=0.0,
        bubble_radius=cfg.bubble_radius(label, float(np.hypot(vx, vy)))
        if radius is None else radius,
        predicted_path=constant_velocity_path(x, y, vx, vy,
                                              cfg.pred_horizon, cfg.pred_samples)
        if path is None else path,
    )


# -------------------------------------------------------------------- factors
def test_range_factor_decays_with_distance():
    s = ImportanceScorer()
    xyz = np.array([[1.0, 0, 0], [10.0, 0, 0], [50.0, 0, 0]], np.float32)
    f = s.range_factor(xyz)
    assert f[0] > f[1] > f[2]
    assert np.all((f >= 0) & (f <= 1))
    assert f[0] == pytest.approx(np.exp(-1 / 30.0), abs=1e-5)


def test_semantic_factor_uses_the_class_table():
    s = ImportanceScorer()
    f = s.semantic_factor(np.array([1, 3, 6, 99]))
    assert f[0] == pytest.approx(DEFAULT_SEMANTIC_IMPORTANCE[1])
    assert f[1] < f[0]                      # vegetation cheaper than road
    assert f[2] > f[1]                      # vehicle dearer than vegetation
    assert 0.0 <= f[3] <= 1.0               # unknown class stays in range


def test_bubble_factor_is_one_inside_and_falls_off_outside():
    s = ImportanceScorer()
    t = _track(label="pedestrian", x=0.0, y=0.0, vx=0.0, vy=0.0, radius=4.0)
    xyz = np.array([[0.0, 0, 0], [3.0, 0, 0], [5.0, 0, 0], [9.0, 0, 0]], np.float32)
    f = s.bubble_factor(xyz, [t])
    assert f[0] == pytest.approx(1.0)
    assert f[1] > f[2] > f[3]
    assert f[3] == 0.0                      # beyond 1.5 r


def test_dynamic_factor_is_raised_inside_the_track_halo():
    s = ImportanceScorer(halo_radius_m=3.0, halo_value=0.7)
    t = _track(x=0.0, y=0.0, vx=0.0, vy=0.0)
    xyz = np.array([[0.0, 0, 0], [2.5, 0, 0], [6.0, 0, 0]], np.float32)
    f = s.dynamic_factor(xyz, np.zeros(3, np.float32), [t])
    assert f[0] == pytest.approx(0.7)
    assert f[1] == pytest.approx(0.7)
    assert f[2] == pytest.approx(0.0)


def test_prediction_factor_lights_up_the_future_corridor():
    s = ImportanceScorer()
    # Vehicle at the origin moving +x at 10 m/s: it will be near x = 20 in 2 s.
    t = _track(label="vehicle", x=0.0, y=0.0, vx=10.0, vy=0.0, radius=8.0)
    xyz = np.array([[20.0, 0, 0], [-40.0, 0, 0]], np.float32)
    f = s.prediction_factor(xyz, [t])
    assert f[0] > f[1]
    assert 0.0 <= f[1] <= 1.0


def test_total_is_a_weighted_average_in_unit_range():
    s = ImportanceScorer()
    xyz = np.random.default_rng(0).uniform(-40, 40, (500, 3)).astype(np.float32)
    labels = np.random.default_rng(1).integers(0, 9, 500).astype(np.int32)
    t = _track(radius=6.0)
    f = s.score(xyz, labels, np.zeros(500, np.float32), [t])
    assert np.all((f.total >= 0.0) & (f.total <= 1.0))
    w = s.weights
    manual = (w["w_r"] * f.range + w["w_d"] * f.dynamic + w["w_s"] * f.semantic
              + w["w_b"] * f.bubble + w["w_p"] * f.prediction)
    assert np.allclose(f.total, manual, atol=1e-6)


def test_default_weights_sum_to_one():
    s = ImportanceScorer()
    assert sum(s.weights.values()) == pytest.approx(1.0)
    assert s.weights["w_s"] == pytest.approx(0.25)
    assert s.weights["w_b"] == pytest.approx(0.25)


def test_ablation_renormalizes_the_remaining_weights():
    s = ImportanceScorer(use_range=True, use_semantic=False, use_dynamic=False,
                         use_bubble=False, use_prediction=False)
    assert s.weights["w_r"] == pytest.approx(1.0)
    assert s.weights["w_s"] == 0.0
    xyz = np.array([[2.0, 0, 0]], np.float32)
    f = s.score(xyz, np.array([1], np.int32), np.zeros(1, np.float32), None)
    assert f.total[0] == pytest.approx(f.range[0], abs=1e-6)


def test_every_ablation_keeps_the_total_in_range():
    rng = np.random.default_rng(7)
    xyz = rng.uniform(-40, 40, (300, 3)).astype(np.float32)
    labels = rng.integers(0, 9, 300).astype(np.int32)
    t = _track(radius=8.0)
    for flags in ({"use_semantic": False}, {"use_dynamic": False},
                  {"use_bubble": False}, {"use_prediction": False}):
        base = dict(use_range=True, use_semantic=True, use_dynamic=True,
                    use_bubble=True, use_prediction=True)
        base.update(flags)
        s = ImportanceScorer(**base)
        f = s.score(xyz, labels, rng.random(300).astype(np.float32), [t])
        assert np.all((f.total >= 0) & (f.total <= 1)), flags
        assert sum(s.weights.values()) == pytest.approx(1.0)


def test_factor_means_are_exposed_for_ablations():
    s = ImportanceScorer()
    xyz = np.random.default_rng(2).uniform(-30, 30, (200, 3)).astype(np.float32)
    f = s.score(xyz, np.ones(200, np.int32), None, None)
    summary = f.summary()
    assert set(summary) == {"range", "dynamic", "semantic", "bubble",
                            "prediction", "total"}
    assert 0.0 <= summary["total"] <= 1.0


# ----------------------------------------------------------------- bubble law
def test_bubble_radius_follows_the_documented_linear_law():
    cfg = PredictionConfig()
    # r = r_min(label) + kappa * |v| * T
    r = cfg.bubble_radius("vehicle", 10.0)
    assert r == pytest.approx(DEFAULT_R_MIN["vehicle"] + cfg.kappa * 10.0 * cfg.horizon)
    assert cfg.bubble_radius("vehicle", 1.0) < r        # grows with speed
    assert cfg.bubble_radius("vehicle", 0.0) == pytest.approx(DEFAULT_R_MIN["vehicle"])


def test_class_bubble_minima_are_ordered_by_exposure():
    cfg = PredictionConfig()
    assert (cfg.bubble_radius("pedestrian", 0.0)
            < cfg.bubble_radius("cyclist", 0.0)
            < cfg.bubble_radius("vehicle", 0.0))
    assert bubble_radius("unknown", 0.0) > 0.0


def test_constant_velocity_path_is_horizon_limited():
    path = constant_velocity_path(0.0, 0.0, 10.0, 2.0, horizon=2.0, samples=8)
    assert len(path) == 8
    first, last = path_endpoints(path)
    # Samples are strictly in the future: t = dt .. horizon (t = 0 is not predicted).
    assert first == (2.5, 0.5)                     # t = 0.25 s (horizon/samples)
    assert last == (20.0, 4.0)                     # t = 2.0 s
    speed = float(np.hypot(10.0, 2.0))
    assert corridor_length(path) == pytest.approx(speed * 1.75, rel=1e-3)


def test_prediction_is_disabled_gracefully():
    assert constant_velocity_path(0.0, 0.0, 1.0, 0.0, horizon=0.0) == []
    assert constant_velocity_path(0.0, 0.0, 1.0, 0.0, horizon=2.0, samples=0) == []
    assert path_endpoints([]) == (None, None)


def test_corridor_radius_is_bounded():
    cfg = PredictionConfig()
    assert cfg.corridor_radius(100.0) == pytest.approx(4.0)
    assert cfg.corridor_radius(0.5) == pytest.approx(1.0)
