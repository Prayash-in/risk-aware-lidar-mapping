"""Semantic labelling, dynamic scoring, and the segmenter API.

The RandLA-Net network itself lives in `test_randlanet.py`; this file covers the
perception *pipeline*: how labels arrive, their provenance, the checkpoint gate,
and the fallback that never pretends to be a prediction.
"""

from __future__ import annotations

import numpy as np
import pytest

from adaptive_lidar.data.base import (
    SEMANTIC_SOURCE_GT,
    SEMANTIC_SOURCE_MODEL,
    SEMANTIC_SOURCE_UNKNOWN,
    PointCloud,
)
from adaptive_lidar.perception import DynamicScorer, SemanticLabeler
from adaptive_lidar.perception.randlanet import (
    DEFAULT_CONFIG,
    RandLANetSegmenter,
    SegmenterResult,
    find_checkpoint,
    is_torch_available,
    point_features,
)
from adaptive_lidar.perception.randlanet.inference import CheckpointNotAvailable
from adaptive_lidar.types import DYNAMIC_CLASSES, UNKNOWN

SAMPLE = np.array([
    [-8.0, -1.0, 0.0], [-7.5, -1.2, 0.0], [-7.9, 0.8, 0.0],   # road cluster
    [3.0, 3.0, 1.7], [3.2, 3.1, 1.6], [3.1, 3.4, 1.8],         # pedestrian cluster
    [10.0, 10.0, 2.1], [9.7, 10.2, 2.0], [10.3, 9.8, 2.2],      # far vehicle cluster
    [0.0, 0.3, 0.1],
], dtype=np.float32)

SEMANTICS = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3], np.int32)


def _cloud(semantic=SEMANTICS, source=SEMANTIC_SOURCE_GT):
    return PointCloud(
        xyz=SAMPLE, intensity=np.ones(SAMPLE.shape[0], np.float32),
        semantic=semantic, confidence=np.ones(SAMPLE.shape[0], np.float32),
        semantic_source=source,
    )


class TestSemanticLabeler:
    def test_ground_truth_source_is_preserved(self):
        lab = SemanticLabeler(use_model=False).label(_cloud())
        assert lab.source == SEMANTIC_SOURCE_GT
        np.testing.assert_array_equal(lab.labels, SEMANTICS)
        assert lab.confidence.shape == SAMPLE.shape[:1]

    def test_unknown_source_when_no_labels(self):
        n = SAMPLE.shape[0]
        cloud = PointCloud(
            xyz=SAMPLE, intensity=np.ones(n, np.float32),
            semantic=np.zeros(n, np.int32), confidence=np.ones(n, np.float32),
            semantic_source=SEMANTIC_SOURCE_UNKNOWN,
        )
        lab = SemanticLabeler(use_model=False).label(cloud)
        assert lab.source == SEMANTIC_SOURCE_UNKNOWN
        np.testing.assert_array_equal(lab.labels, np.zeros(n, np.int32))

    def test_use_model_true_but_unavailable_falls_back_truthfully(self):
        lab = SemanticLabeler(use_model=True, checkpoint=None, prefer_ground_truth=False)
        out = lab.label(_cloud())
        # no checkpoint + no torch -> must be ground truth, never a fake prediction
        if not is_torch_available():
            assert out.source == SEMANTIC_SOURCE_GT
        assert "unavailable" in out.note or "ground-truth" in out.note

    def test_semantic_importance_table_is_bounded(self):
        lab = SemanticLabeler()
        imp = lab.importance(np.array([0, 1, 7, 99], np.int32))
        assert (imp >= 0).all() and (imp <= 1.5).all()
        np.testing.assert_allclose(
            imp[:3],
            [lab.semantic_importance[0], lab.semantic_importance[1],
             lab.semantic_importance[7]])

    def test_dynamic_labels_match_dynamic_classes(self):
        lab = SemanticLabeler()
        labels = np.array([2, 1, 3, 0, 4, 5, 6, 7, 8], np.int32)
        dyn = lab.dynamic_labels(labels)
        expected = np.isin(labels, tuple(DYNAMIC_CLASSES))
        np.testing.assert_array_equal(dyn, expected)

    def test_status_reports_use_model(self):
        st = SemanticLabeler(use_model=True).status()
        assert st["use_model"] is True
        assert "available" in st


class TestDynamicScorer:
    def _two_frames(self):
        static = SAMPLE
        moving = np.concatenate([SAMPLE[:-1], [[5.0, 5.0, 0.5]]], axis=0)
        return static, moving

    def test_move_is_detected_and_static_is_not(self):
        scorer = DynamicScorer(cell_size=1.0)
        static, moving = self._two_frames()
        scorer.update(static, frame_id=0)
        s_moving = scorer.update(moving, frame_id=1)
        # the changed cell must score higher than the untouched cluster
        assert s_moving[-1] > 0.0
        assert np.median(s_moving[:-1]) < s_moving[-1]

    def test_score_is_bounded(self):
        scorer = DynamicScorer()
        a = np.concatenate([SAMPLE, [[7.0, 7.0, 2.2]]], axis=0)
        b = np.concatenate([SAMPLE, [[7.0, 7.5, 2.2]]], axis=0)
        scorer.update(a, frame_id=0)
        out = scorer.update(b, frame_id=1)
        assert (out >= 0).all() and (out <= scorer.max_score).all()

    def test_reset_clears_memory(self):
        scorer = DynamicScorer()
        scorer.update(SAMPLE, frame_id=0)
        scorer.update(SAMPLE + 0.01, frame_id=1)
        assert len(scorer._history) == 2
        scorer.reset()
        assert scorer._history == []
        assert scorer._heat is None

    def test_class_based_dynamism_boosts_score(self):
        scorer = DynamicScorer(cell_size=1.0)
        mask = np.zeros(SAMPLE.shape[0], dtype=bool)
        mask[3] = True
        out = scorer.update(SAMPLE, frame_id=0, dynamic_mask=mask)
        assert out[3] == pytest.approx(1.0)
        assert out[0] == pytest.approx(0.0)


class TestSegmenterAPI:
    def test_raises_without_torch_or_checkpoint(self):
        s = RandLANetSegmenter(checkpoint=None)
        assert not s.available
        with pytest.raises(CheckpointNotAvailable):
            s.require()

    def test_status_documents_capability(self):
        s = RandLANetSegmenter(checkpoint=None)
        st = s.status()
        assert st["implemented"] is True
        assert st["architecture"]["d_in"] == DEFAULT_CONFIG["d_in"]
        assert "no checkpoint is shipped" in st["note"]
        assert st["available"] is (is_torch_available() and s.checkpoint_path is not None)

    def test_find_checkpoint_missing_paths(self):
        assert find_checkpoint("") is None
        assert find_checkpoint("does/not/exist.pth") is None

    def test_map_external_to_internal_unknown_by_default(self):
        seg = RandLANetSegmenter(checkpoint=None)
        ext = np.array([24, 40, 252, 1], np.int32)
        out = seg.map_external_to_internal(ext, class_ids=[1, 40, 252])
        assert out.tolist() == [0, 1, 2, 0]

    def test_segmenter_result_describe(self):
        r = SegmenterResult(
            labels=np.array([0, 3, 3], np.int32),
            confidence=np.ones(3, np.float32),
            checkpoint="fake",
        )
        d = r.describe()
        assert d["source"] == SEMANTIC_SOURCE_MODEL
        assert d["points"] == 3

    def test_point_features_exported_column_count(self):
        f = point_features(SAMPLE)
        assert f.shape[1] == DEFAULT_CONFIG["d_in"]


class TestProvenance:
    def test_sources_are_distinct(self):
        assert len({SEMANTIC_SOURCE_UNKNOWN, SEMANTIC_SOURCE_GT,
                    SEMANTIC_SOURCE_MODEL}) == 3

    def test_class_names_cover_unknown(self):
        from adaptive_lidar.types import CLASS_NAMES
        assert CLASS_NAMES[UNKNOWN] == "unknown"