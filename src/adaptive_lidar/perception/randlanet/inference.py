"""RandLA-Net inference: checkpoint loading, voxelisation, sliding window.

`RandLANetSegmenter` is the only entry point the pipeline needs. It is honest
about its own capability:

* checkpoint found + torch installed -> real network forward pass, labels are
  reported as ``model_prediction`` with a confidence per point;
* otherwise -> it raises `CheckpointNotAvailable` and the caller decides the
  fallback (dataset ground truth, or ``unknown``), which the provenance field
  then records truthfully.

No default behaviour silently pretends to be a prediction.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from adaptive_lidar.data.base import SEMANTIC_SOURCE_MODEL
from adaptive_lidar.perception.randlanet.model import (
    DEFAULT_CONFIG,
    is_torch_available,
    knn_graph,
    point_features,
    randla_net_module,
)
from adaptive_lidar.types import CLASS_NAMES, UNKNOWN


class CheckpointNotAvailable(RuntimeError):
    """Raised when real inference cannot run (no torch, or no checkpoint)."""


@dataclass
class SegmenterResult:
    """Per-point network output."""

    labels: np.ndarray            # (N,) int32, internal taxonomy ids
    confidence: np.ndarray        # (N,) float32 in [0, 1]
    source: str = SEMANTIC_SOURCE_MODEL
    checkpoint: str | None = None
    meta: dict = field(default_factory=dict)

    def describe(self) -> dict:
        return {
            "source": self.source,
            "checkpoint": self.checkpoint,
            "points": int(self.labels.shape[0]),
            **{k: v for k, v in self.meta.items()},
        }


def find_checkpoint(path: str | None) -> str | None:
    """Return an existing checkpoint path, or None."""
    if not path:
        return None
    if os.path.isfile(path):
        return path
    # Also accept a directory containing checkpoint.pth
    cand = os.path.join(path, "checkpoint.pth")
    return cand if os.path.isfile(cand) else None


class RandLANetSegmenter:
    """Lazily-loaded RandLA-Net inference wrapper.

    Parameters
    ----------
    checkpoint: path to a ``.pth``/``.pt`` state dict, or a directory holding
        ``checkpoint.pth``. May be None — the object then reports that it is
        unavailable instead of failing at import time.
    device / voxel_size / num_classes: inference settings; they must match the
        checkpoint, see `docs/RANDLANET.md`.
    """

    def __init__(self, checkpoint: str | None = None, device: str = "cpu",
                 voxel_size: float = 0.06, num_classes: int = 9,
                 k_neighbors: int = 16, config: dict | None = None):
        self.checkpoint_path = find_checkpoint(checkpoint)
        self.device = device
        self.voxel_size = float(voxel_size)
        self.num_classes = int(num_classes)
        self.k_neighbors = int(k_neighbors)
        self.config = dict(DEFAULT_CONFIG)
        self.config.update(config or {})
        self.config["d_out"] = self.num_classes
        self.config["voxel_size"] = self.voxel_size
        self.config["num_neighbors"] = self.k_neighbors
        self._model = None
        self._loaded = False

    # ------------------------------------------------------------- capability
    @property
    def available(self) -> bool:
        """True only when both torch and a checkpoint are usable."""
        return is_torch_available() and self.checkpoint_path is not None

    def status(self) -> dict:
        return {
            "torch_available": is_torch_available(),
            "checkpoint": self.checkpoint_path,
            "available": self.available,
            "device": self.device,
            "voxel_size_m": self.voxel_size,
            "num_classes": self.num_classes,
            "architecture": {
                "d_in": self.config["d_in"], "d_out": self.config["d_out"],
                "encoder_dims": list(self.config["encoder_dims"]),
                "decoder_dims": list(self.config["decoder_dims"]),
                "k_neighbors": list(self.config["k_neighbors"]),
                "subsampling": list(self.config["subsampling"]),
            },
            "implemented": True,
            "weights_available": self.checkpoint_path is not None,
            "note": ("no checkpoint is shipped with this repository; labels are "
                     "ground truth or unknown, never model predictions"),
        }

    def require(self) -> None:
        if not is_torch_available():
            raise CheckpointNotAvailable(
                "torch is not installed, so RandLA-Net inference cannot run. "
                "Install the optional dependency (pip install torch) or use the "
                "ground-truth / unknown semantic path; see docs/RANDLANET.md."
            )
        if self.checkpoint_path is None:
            raise CheckpointNotAvailable(
                "no RandLA-Net checkpoint found. Place a trained checkpoint at "
                "models/randlanet/checkpoint.pth (see models/randlanet/README.md) "
                "or pass checkpoint=<path>. Without it the pipeline reports "
                "ground-truth or unknown semantics, never predictions."
            )

    # ------------------------------------------------------------------ model
    def load(self):  # pragma: no cover - requires torch and a checkpoint
        """Instantiate the network and load its weights (idempotent)."""
        if self._loaded:
            return self._model
        self.require()
        import torch
        model = randla_net_module(self.config)
        state = torch.load(self.checkpoint_path, map_location=self.device,
                           weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            # A checkpoint that does not match the architecture would silently
            # produce random weights; say so instead.
            self.load_report = {
                "missing_keys": list(missing)[:10], "unexpected_keys": list(unexpected)[:10],
            }
            raise CheckpointNotAvailable(
                f"checkpoint {self.checkpoint_path} does not match the network "
                f"({len(missing)} missing keys); see docs/RANDLANET.md for the "
                f"expected architecture")
        model.eval().to(self.device)
        self._model = model
        self._loaded = True
        self.load_report = {
            "missing_keys": [], "unexpected_keys": list(unexpected)[:10],
        }
        return model

    # -------------------------------------------------------------- inference
    def voxel_downsample(self, xyz: np.ndarray, intensity: np.ndarray,
                         semantic: np.ndarray | None = None) -> np.ndarray:
        """Keep one point per voxel (largest intensity), returning indices."""
        v = self.voxel_size
        keys = np.floor(np.asarray(xyz, dtype=np.float64) / v).astype(np.int64)
        keys = (keys - keys.min(axis=0))
        packed = (keys[:, 0] * (keys[:, 1].max() + 1) + keys[:, 1]) * (keys[:, 2].max() + 1) + keys[:, 2]
        order = np.lexsort((-np.asarray(intensity), packed))
        first = np.ones(packed.shape[0], dtype=bool)
        first[order[1:]] = packed[order[1:]] != packed[order[:-1]]
        return np.nonzero(first)[0]

    def predict(self, xyz: np.ndarray, intensity: np.ndarray | None = None,
                semantic_hint: np.ndarray | None = None) -> SegmenterResult:
        """Segment one scan. Raises `CheckpointNotAvailable` if not runnable."""
        self.require()
        xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
        inten = (np.ones(xyz.shape[0], np.float32) if intensity is None
                 else np.asarray(intensity, dtype=np.float32))
        idx = self.voxel_downsample(xyz, inten)
        if idx.shape[0] < 1:  # pragma: no cover - empty frame guard
            raise ValueError("empty point cloud")

        feats = point_features(xyz[idx], inten[idx])
        labels = np.zeros(xyz.shape[0], dtype=np.int32)
        conf = np.zeros(xyz.shape[0], dtype=np.float32)
        probs = self._forward(feats, idx.shape[0])  # (M, num_classes)
        pred = probs.argmax(axis=1).astype(np.int32)
        pmax = probs.max(axis=1).astype(np.float32)

        # Nearest-sample assignment back to every original point.
        tree_idx = knn_graph(xyz[idx], min(self.k_neighbors, idx.shape[0]))[:, 0]
        labels[:] = pred[tree_idx]
        conf[:] = pmax[tree_idx]
        return SegmenterResult(
            labels=labels, confidence=conf, checkpoint=self.checkpoint_path,
            meta={"samples": int(idx.shape[0]), "points": int(xyz.shape[0])},
        )

    def _forward(self, feats: np.ndarray, m: int):  # pragma: no cover - torch
        import torch
        model = self.load()
        graph = self._graph(feats)
        with torch.no_grad():
            x = torch.from_numpy(feats.T[None, ...]).float().to(self.device)
            tgraph = {
                key: torch.from_numpy(val.T[None, ...]).long().to(self.device)
                if key.startswith("knn_") else
                torch.from_numpy(val[None, ...]).long().to(self.device)
                for key, val in graph.items()
            }
            logits = model(x, tgraph)          # (1, num_classes, M)
            return torch.softmax(logits, dim=1)[0].cpu().numpy().T

    def _graph(self, feats: np.ndarray) -> dict:
        """KNN + subsampling graph for one sample, built in plain numpy."""
        from adaptive_lidar.perception.randlanet.model import build_graph
        return build_graph(feats[:, :3], self.config)

    # -------------------------------------------------------------- utilities
    @staticmethod
    def map_external_to_internal(labels: np.ndarray, class_ids: list[int]) -> np.ndarray:
        """Map an external label set onto the internal taxonomy (0 = unknown)."""
        out = np.zeros_like(np.asarray(labels, dtype=np.int32))
        for i, ext in enumerate(class_ids):
            out[np.asarray(labels) == int(ext)] = i
        return out

    @staticmethod
    def describe_labels(labels: np.ndarray) -> dict:
        """Histogram of predicted classes with readable names."""
        uniq, counts = np.unique(np.asarray(labels), return_counts=True)
        return {CLASS_NAMES.get(int(u), str(int(u))): int(c) for u, c in zip(uniq, counts)}
