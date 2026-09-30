"""Semantic labelling with truthful provenance.

`SemanticLabeler` decides, per frame, where the semantics come from:

1. RandLA-Net inference (if a checkpoint + torch are available) -> `model_prediction`
2. dataset ground-truth labels                                 -> `ground_truth`
3. nothing available                                           -> `unknown`

The chosen source is stored on the frame and exported with every result, so a
reader can never mistake ground truth for a prediction. It also exposes the
semantic importance table used by the risk controller (§7): a prototype
heuristic, documented as such, not a trained result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from adaptive_lidar.data.base import (
    SEMANTIC_SOURCE_GT,
    SEMANTIC_SOURCE_MODEL,
    SEMANTIC_SOURCE_UNKNOWN,
    PointCloud,
)
from adaptive_lidar.perception.randlanet import CheckpointNotAvailable, RandLANetSegmenter
from adaptive_lidar.types import UNKNOWN


@dataclass
class SemanticResult:
    labels: np.ndarray
    confidence: np.ndarray
    source: str
    checkpoint: str | None = None
    note: str = ""


class SemanticLabeler:
    """Assign per-point semantic classes, recording their provenance."""

    def __init__(self, use_model: bool = False, checkpoint: str | None = None,
                 device: str = "cpu", voxel_size: float = 0.06,
                 num_classes: int = 9, prefer_ground_truth: bool = False,
                 semantic_importance: dict | None = None):
        self.use_model = bool(use_model)
        self.prefer_ground_truth = bool(prefer_ground_truth)
        self.segmenter = RandLANetSegmenter(
            checkpoint=checkpoint, device=device, voxel_size=voxel_size,
            num_classes=num_classes)
        # Prototype heuristic table (config-controlled); not a learned quantity.
        self.semantic_importance = dict(semantic_importance or {
            0: 0.6, 1: 1.0, 2: 0.8, 3: 0.3, 4: 0.5, 5: 0.7, 6: 0.9, 7: 0.9, 8: 0.9,
        })

    # ------------------------------------------------------------------ label
    def label(self, cloud: PointCloud) -> SemanticResult:
        n = cloud.num_points
        want_model = self.use_model and not self.prefer_ground_truth
        if want_model:
            try:
                out = self.segmenter.predict(cloud.xyz, cloud.intensity)
                return SemanticResult(out.labels, out.confidence,
                                      SEMANTIC_SOURCE_MODEL, out.checkpoint,
                                      "RandLA-Net checkpoint inference")
            except CheckpointNotAvailable as exc:
                note = f"model unavailable ({exc.__class__.__name__}); "
            else:  # pragma: no cover - defensive
                note = ""
        else:
            note = ""

        if cloud.semantic_source == SEMANTIC_SOURCE_GT or np.any(cloud.semantic > UNKNOWN):
            note += "using dataset ground-truth labels"
            return SemanticResult(cloud.semantic.copy(), cloud.confidence.copy(),
                                  SEMANTIC_SOURCE_GT, None, note.strip())
        return SemanticResult(np.zeros(n, np.int32), np.zeros(n, np.float32),
                              SEMANTIC_SOURCE_UNKNOWN, None,
                              (note + "no checkpoint and no labels: semantics unknown").strip())

    # ------------------------------------------------------------- controller
    def importance(self, labels: np.ndarray) -> np.ndarray:
        """Per-point semantic importance in [0, 1] from the config table."""
        table = self.semantic_importance
        return np.array([float(table.get(int(v), 0.5)) for v in np.asarray(labels)],
                        dtype=np.float32)

    def dynamic_labels(self, labels: np.ndarray) -> np.ndarray:
        """Boolean mask of dynamic-class points (vehicle/pedestrian/cyclist)."""
        from adaptive_lidar.types import DYNAMIC_CLASSES
        return np.isin(np.asarray(labels), tuple(DYNAMIC_CLASSES))

    def status(self) -> dict:
        info = self.segmenter.status()
        info["use_model"] = self.use_model
        return info
