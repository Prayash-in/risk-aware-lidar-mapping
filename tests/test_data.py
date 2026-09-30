"""Data backends: synthetic scene and SemanticKITTI reader."""

from __future__ import annotations

import numpy as np
import pytest

from adaptive_lidar.data import (
    SEMANTIC_SOURCE_GT,
    SEMANTIC_SOURCE_UNKNOWN,
    PointCloud,
    SemanticKITTIPointCloudSource,
    SyntheticPointCloudSource,
    make_source,
    write_kitti_fixture,
)
from adaptive_lidar.data.semantic_kitti import DEFAULT_LABEL_MAP


# ------------------------------------------------------------------ synthetic
def test_synthetic_frames_have_points_and_labels():
    src = SyntheticPointCloudSource("urban_crossing", num_frames=4)
    cloud = src.load_frame(0)
    assert cloud.num_points > 1000
    assert cloud.semantic_source == SEMANTIC_SOURCE_GT
    assert set(np.unique(cloud.semantic)).issubset(set(range(9)))


def test_synthetic_is_deterministic():
    a = SyntheticPointCloudSource("urban_crossing", num_frames=3)
    b = SyntheticPointCloudSource("urban_crossing", num_frames=3)
    for i in range(3):
        ca, cb = a.load_frame(i), b.load_frame(i)
        assert ca.num_points == cb.num_points
        assert np.allclose(ca.xyz, cb.xyz)
        assert np.array_equal(ca.semantic, cb.semantic)


def test_synthetic_random_access_replays_history():
    src = SyntheticPointCloudSource("urban_crossing", num_frames=5)
    first = src.load_frame(2)
    src.load_frame(4)
    again = src.load_frame(2)          # must replay deterministically
    assert np.allclose(first.xyz, again.xyz)


def test_synthetic_timestamps_advance():
    src = SyntheticPointCloudSource("urban_crossing", num_frames=4)
    stamps = [src.load_frame(i).timestamp for i in range(4)]
    assert stamps == sorted(stamps)
    assert stamps[1] > stamps[0]


def test_synthetic_range_filter():
    src = SyntheticPointCloudSource("urban_crossing", num_frames=1, max_range_m=12.0)
    cloud = src.load_frame(0)
    assert cloud.range().max() <= 12.0 + 1e-6


def test_scenarios_load():
    for name in ("urban_crossing", "downtown_rush", "demo_60s", "risk_proof"):
        src = SyntheticPointCloudSource(name, num_frames=1)
        assert src.load_frame(0).num_points > 0


# ---------------------------------------------------------------- SemanticKITTI
def test_kitti_missing_dataset_raises_descriptive_error():
    with pytest.raises(FileNotFoundError) as excinfo:
        SemanticKITTIPointCloudSource(root="definitely/not/here", sequence="08")
    assert "SemanticKITTI" in str(excinfo.value)


def test_kitti_fixture_reads_scan_labels_and_pose(tmp_path):
    write_kitti_fixture(str(tmp_path), "08", num_frames=3, points_per_frame=200)
    src = SemanticKITTIPointCloudSource(root=str(tmp_path), sequence="08", num_frames=3)
    cloud = src.load_frame(1)
    assert cloud.num_points == 200
    assert cloud.semantic_source == SEMANTIC_SOURCE_GT
    assert set(np.unique(cloud.semantic)).issubset(set(DEFAULT_LABEL_MAP.values()))
    assert np.isclose(src.get_pose(1)[0, 3], 0.8)


def test_kitti_ego_motion_compensation_aligns_frames(tmp_path):
    write_kitti_fixture(str(tmp_path), "08", num_frames=3, points_per_frame=300)
    src = SemanticKITTIPointCloudSource(root=str(tmp_path), sequence="08", num_frames=3)
    ref = src.load_frame(0)
    moved = src.compensate(1, target_index=0)
    assert np.allclose(moved.xyz, ref.xyz, atol=1e-3)
    # Without compensation the same points are offset by the ego motion.
    assert np.abs(src.load_frame(1).xyz[:, 0] - ref.xyz[:, 0]).max() > 0.5


def test_kitti_stride_points_applies_to_labels_and_points(tmp_path):
    write_kitti_fixture(str(tmp_path), "08", num_frames=2, points_per_frame=300)
    src = SemanticKITTIPointCloudSource(root=str(tmp_path), sequence="08",
                                        num_frames=2, stride_points=120)
    assert src.load_frame(0).num_points == 120


def test_kitti_without_labels_reports_unknown(tmp_path):
    write_kitti_fixture(str(tmp_path), "08", num_frames=1, points_per_frame=100,
                        with_labels=False)
    src = SemanticKITTIPointCloudSource(root=str(tmp_path), sequence="08",
                                        num_frames=1, use_labels=False)
    cloud = src.load_frame(0)
    assert cloud.semantic_source == SEMANTIC_SOURCE_UNKNOWN
    assert np.all(cloud.semantic == 0)


# --------------------------------------------------------------------- common
def test_point_cloud_transform_and_filter():
    cloud = PointCloud(xyz=np.array([[1.0, 0.0, 0.0], [30.0, 0.0, 0.0]], np.float32),
                       intensity=np.ones(2, np.float32),
                       semantic=np.array([1, 6], np.int32),
                       confidence=np.ones(2, np.float32))
    moved = cloud.transform(np.array([[1, 0, 0, 5], [0, 1, 0, 0],
                                      [0, 0, 1, 0], [0, 0, 0, 1]], np.float32))
    assert np.allclose(moved.xyz[0], [6.0, 0.0, 0.0])
    assert cloud.filter_range(10.0).num_points == 1


def test_make_source_dispatch():
    assert make_source("synthetic", num_frames=1).name == "synthetic"
    with pytest.raises(ValueError):
        make_source("nope")
