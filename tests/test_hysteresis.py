"""Hysteresis state machine: thresholds, persistence, deadband, expiry."""

from __future__ import annotations

import numpy as np
import pytest

from adaptive_lidar.controller.hysteresis import (
    KEEP,
    MERGE,
    SPLIT,
    HysteresisController,
)

EXTENT = 100.0


def _points(x=0.0, y=0.0, n=8):
    return np.array([[x, y, 0.0]] * n, dtype=np.float64)


def _run(ctrl, importance, frames=6, x=0.0, y=0.0):
    """Feed the same ref-cell `frames` times; return the per-frame decisions."""
    out = []
    for _ in range(frames):
        d = ctrl.update(_points(x, y), np.full(8, float(importance)))
        out.append(d)
    return out


def test_thresholds_must_be_ordered():
    with pytest.raises(ValueError):
        HysteresisController(split_threshold=0.2, merge_threshold=0.5)
    with pytest.raises(ValueError):
        HysteresisController(split_frames=0)


def test_split_requires_persistence_before_it_fires():
    ctrl = HysteresisController(split_threshold=0.6, split_frames=2, merge_frames=3)
    first = ctrl.update(_points(), np.full(8, 0.9))
    assert first.split_ops == 0            # frame 1: state seen, candidate recorded
    second = ctrl.update(_points(), np.full(8, 0.9))
    assert second.split_ops == 1           # frame 2: 2 consecutive frames -> SPLIT
    assert np.all(second.shift == 1)
    third = ctrl.update(_points(), np.full(8, 0.9))
    assert third.split_ops == 0            # already split: steady state
    assert np.all(third.shift == 1)


def test_a_single_spike_does_not_split_the_map():
    ctrl = HysteresisController(split_frames=2, ema_alpha=1.0)
    ctrl.update(_points(), np.full(8, 0.0))
    spike = ctrl.update(_points(), np.full(8, 1.0))
    assert spike.split_ops == 0
    after = ctrl.update(_points(), np.full(8, 0.0))
    assert after.split_ops == 0
    assert np.all(after.shift == 0)


def test_deadband_between_thresholds_keeps_the_current_level():
    ctrl = HysteresisController(split_threshold=0.6, merge_threshold=0.3,
                               split_frames=1, merge_frames=1, ema_alpha=1.0)
    ctrl.update(_points(), np.full(8, 0.9))          # -> SPLIT
    held = ctrl.update(_points(), np.full(8, 0.45))  # inside the deadband
    assert held.split_ops == 0 and held.merge_ops == 0
    assert np.all(held.shift == 1)


def test_merge_requires_more_frames_than_split():
    ctrl = HysteresisController(split_threshold=0.6, merge_threshold=0.3,
                               split_frames=2, merge_frames=3, ema_alpha=1.0)
    _run(ctrl, 0.9, frames=3)                          # establish SPLIT
    assert ctrl.status()["split_cells"] == 1
    d1 = ctrl.update(_points(), np.full(8, 0.0))
    d2 = ctrl.update(_points(), np.full(8, 0.0))
    assert d1.merge_ops == 0 and d2.merge_ops == 0    # 3 frames are required
    d3 = ctrl.update(_points(), np.full(8, 0.0))
    assert d3.merge_ops == 1 and np.all(d3.shift == -1)
    assert ctrl.status()["merge_cells"] == 1


def test_pending_progress_is_reset_when_the_evidence_stops():
    ctrl = HysteresisController(split_threshold=0.6, split_frames=3, ema_alpha=1.0)
    ctrl.update(_points(0.0, 0.0), np.full(8, 0.9))
    ctrl.update(_points(0.0, 0.0), np.full(8, 0.9))    # pending = 1
    quiet = ctrl.update(_points(0.0, 0.0), np.full(8, 0.0))
    assert quiet.split_ops == 0
    # Two more high frames must restart the count from scratch (3 needed).
    ctrl.update(_points(0.0, 0.0), np.full(8, 0.9))
    final = ctrl.update(_points(0.0, 0.0), np.full(8, 0.9))
    assert final.split_ops == 0
    assert ctrl.update(_points(0.0, 0.0), np.full(8, 0.9)).split_ops == 1


def test_ema_slows_the_transition():
    ctrl = HysteresisController(ema_alpha=0.5, split_threshold=0.6, split_frames=2)
    # A brand-new high-importance cell still needs the full persistence.
    assert ctrl.update(_points(), np.full(8, 1.0)).split_ops == 0
    assert ctrl.update(_points(), np.full(8, 1.0)).split_ops == 1

    # With alpha = 0.5 a halving of the evidence decays the EMA towards 0.5,
    # which sits inside the deadband, so the level is held.
    slow = HysteresisController(ema_alpha=0.5, split_threshold=0.6, merge_threshold=0.3)
    slow.update(_points(), np.full(8, 1.0))
    slow.update(_points(), np.full(8, 1.0))          # -> SPLIT
    assert slow.status()["split_cells"] == 1
    held = slow.update(_points(), np.full(8, 0.0))   # ema -> 0.5 (deadband)
    assert held.merge_ops == 0
    assert np.all(held.shift == 1)


def test_ema_value_is_exposed_for_introspection():
    ctrl = HysteresisController(ema_alpha=0.5)
    ctrl.update(_points(1.0, 1.0), np.full(8, 0.8))
    ctrl.update(_points(1.0, 1.0), np.full(8, 0.2))
    state = ctrl.state_of_point(1.0, 1.0)
    assert state["ema_importance"] == pytest.approx(0.5, abs=1e-3)
    assert state["ref_depth"] == ctrl.ref_depth
    assert state["size_m"] == pytest.approx(100.0 / 512, abs=1e-4)
    assert state["thresholds"] == (ctrl.merge_threshold, ctrl.split_threshold)


def test_cells_expire_after_the_configured_window():
    # A cell that keeps being observed is never expired.
    live = HysteresisController(expiry_frames=2, expiry_sweep_every=1, ema_alpha=1.0)
    for _ in range(6):
        live.update(_points(0.0, 0.0), np.full(8, 0.5))
    assert live.status()["tracked_cells"] == 1

    # A cell that stops being observed is swept out.
    stale = HysteresisController(expiry_frames=2, expiry_sweep_every=1, ema_alpha=1.0)
    stale.update(_points(0.0, 0.0), np.full(8, 0.5))
    for _ in range(6):
        stale.update(_points(40.0, 40.0), np.full(8, 0.5))
    assert stale.status()["tracked_cells"] == 1
    assert stale.state_of_point(0.0, 0.0)["seen"] is False


def test_reset_clears_all_state():
    ctrl = HysteresisController()
    _run(ctrl, 0.9, frames=4)
    assert ctrl.status()["tracked_cells"] > 0
    ctrl.reset()
    st = ctrl.status()
    assert st["tracked_cells"] == 0 and st["frames_seen"] == 0
    assert st["total_split_ops"] == 0 and st["total_merge_ops"] == 0


def test_empty_frame_is_a_no_op():
    ctrl = HysteresisController()
    d = ctrl.update(np.zeros((0, 3)), np.zeros(0))
    assert d.shift.shape[0] == 0 and d.total_ops == 0
    assert ctrl.status()["frames_seen"] == 0


def test_decision_carries_the_reference_cell_diagnostics():
    ctrl = HysteresisController(ref_depth=9)
    d = ctrl.update(_points(3.0, -4.0), np.full(8, 0.7))
    assert d.ref_keys.shape[0] == d.ref_cell_mean.shape[0]
    assert d.ref_cell_mean[0] == pytest.approx(0.7, abs=1e-6)


def test_state_constants_are_distinct():
    assert len({KEEP, SPLIT, MERGE}) == 3
