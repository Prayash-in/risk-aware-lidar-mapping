"""Kalman filter, change detection and multi-object tracking."""

from __future__ import annotations

import numpy as np
import pytest

from adaptive_lidar.data import SyntheticPointCloudSource
from adaptive_lidar.prediction import PredictionConfig
from adaptive_lidar.tracking import KalmanCV, ObjectTracker


# ------------------------------------------------------------------ KalmanCV
def test_kalman_init_zeroes_the_state():
    kf = KalmanCV()
    kf.init(3.0, -2.0)
    assert kf.pos == (3.0, -2.0)
    assert kf.vel == (0.0, 0.0)
    assert kf.speed == 0.0


def test_kalman_recovers_a_constant_velocity():
    kf = KalmanCV(position_noise_m=0.05)
    kf.init(0.0, 0.0)
    dt = 0.1
    truth = []
    for i in range(1, 40):
        kf.predict(dt)
        x, y = 10.0 * i * dt, 2.0 * i * dt
        kf.update(x, y)
        truth.append((x, y))
    assert kf.vel[0] == pytest.approx(10.0, abs=0.3)
    assert kf.vel[1] == pytest.approx(2.0, abs=0.3)
    assert kf.pos[0] == pytest.approx(truth[-1][0], abs=0.4)
    assert kf.speed == pytest.approx(np.hypot(10.0, 2.0), abs=0.5)


def test_kalman_prediction_advances_the_position():
    kf = KalmanCV()
    kf.init(0.0, 0.0)
    for i in range(1, 20):                    # learn vx = 10 m/s
        kf.predict(0.1)
        kf.update(10.0 * i * 0.1, 0.0)
    before = kf.pos[0]
    speed = kf.vel[0]
    kf.predict(0.5)                            # no update: pure dead reckoning
    assert kf.pos[0] == pytest.approx(before + speed * 0.5, rel=0.05)
    assert kf.pos[1] == pytest.approx(0.0, abs=0.3)


def test_kalman_acceleration_is_a_finite_difference():
    kf = KalmanCV()
    kf.init(0.0, 0.0)
    kf.predict(0.1)
    kf.update(1.0, 0.0)
    ax, ay = kf.acceleration(0.1)
    assert ay == 0.0
    assert kf.acceleration(0.0) == (0.0, 0.0)      # guarded against dt = 0


# ------------------------------------------------------------------- tracker
def _moving_object_scan(x0, y0, vx, vy, n_frames, dt=0.1, points=40, cls=6,
                        extent=100.0):
    """A rigid cluster of dynamic points moving at constant velocity."""
    rng = np.random.default_rng(11)
    out = []
    for i in range(n_frames):
        cx, cy = x0 + vx * dt * i, y0 + vy * dt * i
        pts = rng.normal(0.0, 0.25, (points, 2)) + np.array([cx, cy])
        xyz = np.hstack([pts, rng.uniform(0.4, 1.6, (points, 1))]).astype(np.float32)
        sem = np.full(points, cls, dtype=np.int32)
        out.append((xyz, sem, (i + 1) * dt))
    return out


def test_tracker_confirms_a_moving_vehicle():
    tr = ObjectTracker(extent_m=100.0, coast_frames=5)
    frames = _moving_object_scan(5.0, 0.0, 10.0, 0.0, 12)
    tracks = []
    for xyz, sem, t in frames:
        tracks = tr.update(xyz, sem, dt=0.1, timestamp=t)
    assert tracks, "the object must be confirmed after several frames"
    t0 = tracks[0]
    assert t0.label == "vehicle"
    assert t0.x == pytest.approx(5.0 + 10.0 * 0.1 * 11, abs=1.0)
    assert abs(t0.vx) > 3.0


def test_tracker_assigns_stable_ids():
    tr = ObjectTracker()
    frames = _moving_object_scan(5.0, 0.0, 8.0, 0.0, 10)
    ids = set()
    for xyz, sem, t in frames:
        for obj in tr.update(xyz, sem, dt=0.1, timestamp=t):
            ids.add(obj.object_id)
    assert len(ids) == 1                 # one object -> one id for its whole life


def test_bubble_and_rollout_are_attached_to_every_track():
    tr = ObjectTracker(config=PredictionConfig())
    frames = _moving_object_scan(0.0, 0.0, 10.0, 0.0, 8)
    out = []
    for xyz, sem, t in frames:
        out = tr.update(xyz, sem, dt=0.1, timestamp=t)
    for obj in out:
        r_min, kappa, horizon = PredictionConfig().r_min[obj.label], 1.5, 2.0
        assert obj.bubble_radius == pytest.approx(
            r_min + kappa * np.hypot(obj.vx, obj.vy) * horizon)
        assert len(obj.predicted_path) == PredictionConfig().pred_samples
        assert obj.label_source == "semantics"


def test_ground_truth_labels_are_recorded_as_such():
    """GT provenance must never be silently presented as a model result."""
    tr = ObjectTracker(use_gt_labels=True, label_radius_m=6.0)
    frames = _moving_object_scan(5.0, 0.0, 6.0, 0.0, 10)
    actors = [{"x": 5.0 + 6.0 * 0.1 * i, "y": 0.0, "label": "cyclist"}
              for i in range(10)]
    out = []
    for (xyz, sem, t), a in zip(frames, actors):
        out = tr.update(xyz, sem, dt=0.1, timestamp=t, gt_actors=[a])
    assert out and out[0].label == "cyclist"
    assert out[0].label_source == "ground_truth"


def test_static_object_stops_producing_tracks():
    """A cluster that never changes must not be reported as a new object."""
    tr = ObjectTracker()
    rng = np.random.default_rng(5)
    base = rng.normal(0.0, 0.2, (40, 2))
    for _ in range(12):
        xyz = np.hstack([base, rng.uniform(0.4, 1.6, (40, 1))]).astype(np.float32)
        sem = np.full(40, 6, dtype=np.int32)
        tr.update(xyz, sem, dt=0.1)
    assert not tr.tracks


def test_track_coasts_and_then_dies():
    tr = ObjectTracker(coast_frames=2)
    frames = _moving_object_scan(5.0, 0.0, 5.0, 0.0, 8)
    for xyz, sem, t in frames:
        tr.update(xyz, sem, dt=0.1, timestamp=t)
    assert tr.tracks
    empty = np.zeros((0, 3), np.float32)
    for _ in range(2):
        tr.update(empty, np.zeros(0, np.int32), dt=0.1)
    assert tr.tracks, "still coasting inside the window"
    for _ in range(4):
        tr.update(empty, np.zeros(0, np.int32), dt=0.1)
    assert not tr.tracks


def test_reset_clears_the_tracker():
    tr = ObjectTracker()
    for xyz, sem, t in _moving_object_scan(5.0, 0.0, 5.0, 0.0, 8):
        tr.update(xyz, sem, dt=0.1, timestamp=t)
    assert tr.tracks
    tr.reset()
    assert not tr.tracks
    assert tr.status()["next_id"] == 1


def test_two_objects_are_tracked_separately():
    tr = ObjectTracker()
    rng = np.random.default_rng(3)
    for i in range(14):
        t = (i + 1) * 0.1
        clouds, sems = [], []
        for x0, vx in ((-10.0, 6.0), (12.0, -4.0)):
            cx = x0 + vx * 0.1 * i
            pts = rng.normal(0.0, 0.2, (40, 2)) + np.array([cx, 1.0])
            clouds.append(np.hstack([pts, rng.uniform(0.4, 1.6, (40, 1))]).astype(np.float32))
            sems.append(np.full(40, 6, dtype=np.int32))
        out = tr.update(np.vstack(clouds), np.concatenate(sems), dt=0.1, timestamp=t)
    assert len({o.object_id for o in out}) == 2


def test_change_detection_flags_motion_and_ignores_still_geometry():
    tr = ObjectTracker(change_depth=6)
    rng = np.random.default_rng(2)
    still = np.hstack([rng.normal(0, 0.1, (300, 2)), rng.uniform(0, 0.05, (300, 1))])
    still = still.astype(np.float32)
    sem = np.ones(300, np.int32)
    tr.update(still, sem, dt=0.1)
    # The first frame has no history to compare against, so every occupied cell
    # is reported once as "new" (bootstrap).
    assert tr.change_cells > 0
    tr.update(still, sem, dt=0.1)
    assert tr.change_cells == 0, "an unchanged scene must not flag any cell"
    moved = still.copy()
    moved[:, 0] += 3.0                       # everything shifts -> big change
    tr.update(moved, sem, dt=0.1)
    assert tr.change_cells > 0
    tr.update(moved, sem, dt=0.1)
    assert tr.change_cells == 0, "and it settles again once the history matches"


def test_tracker_status_is_reported():
    tr = ObjectTracker()
    for xyz, sem, t in _moving_object_scan(5.0, 0.0, 5.0, 0.0, 6):
        tr.update(xyz, sem, dt=0.1, timestamp=t)
    st = tr.status()
    for key in ("tracks", "change_cells", "candidates", "latency_ms",
                "gate_m", "coast_frames", "use_gt_labels", "prediction"):
        assert key in st
    assert st["latency_ms"] >= 0.0


# -------------------------------------------------- tracker on the synthetic set
def test_tracker_finds_the_synthetic_actors():
    src = SyntheticPointCloudSource("urban_crossing", num_frames=20)
    tr = ObjectTracker()
    last = []
    for i in range(len(src)):
        cloud = src.load_frame(i)
        last = tr.update(cloud.xyz, cloud.semantic, dt=0.1,
                         timestamp=cloud.timestamp,
                         gt_actors=cloud.meta.get("actors") if i > 0 else None)
    labels = {o.label for o in last}
    assert labels & {"vehicle", "cyclist", "pedestrian"}, (
        f"expected dynamic actors, got {labels}")
    for o in last:
        assert o.bubble_radius > 0.0
        assert o.predicted_path
