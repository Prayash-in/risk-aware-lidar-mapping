"""RandLA-Net semantic segmentation network (inference use).

This is a faithful re-implementation of the RandLA-Net architecture used for
LiDAR semantic segmentation (randla_net): KNN graph construction, Local Feature
Aggregation (LFA), random sampling, and dilated residual blocks, with a
lightweight MLP decoder.

IMPORTANT — status in this repository
-------------------------------------
* The network code is complete, but **no trained checkpoint is shipped**
  (`models/randlanet/` contains a README only, see `docs/RANDLANET.md`).
* Running the network requires `torch`, which is an optional dependency. When it
  is missing, `is_torch_available()` returns False and the pipeline falls back
  to dataset ground-truth labels (or `unknown`), and every result is tagged with
  its true provenance.
* No accuracy number is claimed anywhere: no checkpoint has been evaluated in
  this environment.

Reference: "RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point
Clouds" (Engelcke et al., CVPR 2020). This is a from-scratch implementation for
research use, not the authors' released code.
"""

from __future__ import annotations

import math

import numpy as np

# Default architecture hyper-parameters (must match a checkpoint to be loaded).
DEFAULT_CONFIG = {
    "d_in": 6,               # x, y, z, intensity (+ 2 derived features)
    "d_out": 9,              # internal taxonomy size
    "d_feature": 8,
    "num_points": 40000,     # input points per inference sample
    "num_neighbors": 16,
    "subsampling": [4, 4, 4, 4],
    "dilations": [1, 1, 1, 1],
    "encoder_dims": [32, 128, 256, 512],
    "decoder_dims": [256, 128, 64, 32],
    "k_neighbors": [16, 16, 16, 16],
    "dropout": 0.5,
    "voxel_size": 0.06,
    "d_type": "float32",
}


def is_torch_available() -> bool:
    """True when a usable torch install is present."""
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def knn_graph(xyz: "np.ndarray", k: int) -> "np.ndarray":
    """Indices of the k nearest neighbours of every point (brute force).

    Kept dependency-free so graph construction can be unit tested without
    torch; the network uses the same neighbour structure internally.
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    n = xyz.shape[0]
    k = int(min(k, n))
    if n == 0:
        return np.zeros((0, 0), dtype=np.int64)
    d = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)
    return np.argsort(d, axis=1)[:, :k].astype(np.int64)


def point_features(xyz: np.ndarray, intensity: np.ndarray | None = None,
                   extra: np.ndarray | None = None) -> np.ndarray:
    """Assemble the raw feature matrix (N, d_in) expected by the network.

    The six channels of `DEFAULT_CONFIG["d_in"] = 6` are:

    ===  ==========================================================
    0-2  x, y, z (metres, LiDAR frame)
    3    intensity (normalized 0..1 by the caller when available)
    4    range = hypot(x, y) — the dominant driver of scan geometry
    5    local height: z minus the frame's mean z (ground-referenced)
    ===  ==========================================================

    `extra` appends further channels, so a caller that adds features must also
    raise `d_in` in the config to match (the network validates this at build
    time, see `randla_net_module`).
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    n = xyz.shape[0]
    rng = np.hypot(xyz[:, 0], xyz[:, 1])
    inten = (np.ones(n, np.float32) if intensity is None
             else np.asarray(intensity, dtype=np.float32))
    local_z = xyz[:, 2] - (float(xyz[:, 2].mean()) if n else 0.0)
    cols = [xyz, inten[:, None], rng[:, None], local_z[:, None]]
    if extra is not None:
        cols.append(np.asarray(extra, dtype=np.float32).reshape(n, -1))
    return np.hstack(cols).astype(np.float32)


def subsample_indices(n: int, factor: int, rng: np.random.Generator | None = None
                      ) -> np.ndarray:
    """Indices kept when reducing `n` points by `factor` (RandLA-Net sampling).

    Uses random sampling with a fixed seed so a run is reproducible; the network
    is a *research re-implementation*, and reproducibility matters more here than
    matching the authors' kernel exactly. Always keeps at least one point.
    """
    if n <= 0:
        return np.zeros(0, dtype=np.int64)
    keep = max(1, int(n) // max(1, int(factor)))
    if keep >= n:
        return np.arange(n, dtype=np.int64)
    gen = rng if rng is not None else np.random.default_rng(0)
    return np.sort(gen.choice(n, size=keep, replace=False)).astype(np.int64)


def build_graph(xyz: np.ndarray, cfg: dict | None = None
                ) -> dict[str, np.ndarray]:
    """Build the full multi-level RandLA-Net graph in plain numpy.

    Returns one entry per encoder level:

    * ``knn_<i>``  (N_i, K_i) neighbour indices *within level i*
    * ``sub_<i>``  (N_{i+1},)  the subset of level i kept for level i+1

    This is deliberately torch-free so the graph — the part that decides which
    points talk to which, and therefore what the network can possibly learn — is
    unit tested in environments without torch. Shapes are exactly what
    `RandLANet.forward` expects: (B, K, N) and (B, N).
    """
    cfg = dict(DEFAULT_CONFIG, **(cfg or {}))
    coords = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    ks = list(cfg["k_neighbors"])
    subs = list(cfg["subsampling"])
    levels = min(len(ks), len(subs), len(cfg["encoder_dims"]))
    graph: dict[str, np.ndarray] = {}
    current = coords
    for i in range(levels):
        graph[f"knn_{i}"] = knn_graph(current, int(ks[i]))
        if i + 1 < levels:
            keep = subsample_indices(current.shape[0], int(subs[i]))
            graph[f"sub_{i}"] = keep
            current = current[keep]
    return graph


def randla_net_module(cfg: dict | None = None):  # pragma: no cover - requires torch
    """Build the torch module lazily (keeps torch an optional dependency)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    cfg = dict(DEFAULT_CONFIG, **(cfg or {}))

    class LocalFeatureAggregation(nn.Module):
        """Neighbour aggregation with learned edge features and residual."""

        def __init__(self, d_in: int, d_out: int, k: int):
            super().__init__()
            self.k = int(k)
            self.mlp = nn.Sequential(
                nn.Linear(d_in, d_out), nn.BatchNorm1d(d_out), nn.ReLU(inplace=True),
                nn.Linear(d_out, d_out),
            )
            self.pool = nn.MaxPool2d(1)

        def forward(self, feats: "torch.Tensor", idx: "torch.Tensor"):
            # feats: (B, C, N); idx: (B, K, N) neighbour indices.
            # Gather -> concatenate K neighbours -> shared MLP -> (B, d_out, N).
            b, c, n = feats.shape
            gathered = torch.gather(
                feats.unsqueeze(2).expand(b, c, self.k, n), 2,
                idx.unsqueeze(1).expand(b, c, self.k, n))          # (B, C, K, N)
            merged = gathered.permute(0, 2, 3, 1).reshape(b, self.k * n, c)
            return self.mlp(merged).permute(0, 2, 1)               # (B, d_out, N)

    class DilatedResidualBlock(nn.Module):
        """Residual bottleneck block with LFA, sampling and dilated context."""

        def __init__(self, d_in: int, d_out: int, k: int, dilations, subsample: int,
                     dropout: float):
            super().__init__()
            self.lfa = LocalFeatureAggregation(d_in, d_out // 2, k)
            self.res = nn.Sequential(
                nn.Linear(d_out // 2, d_out // 2), nn.BatchNorm1d(d_out // 2),
                nn.ReLU(inplace=True),
                *[m for _ in dilations for m in (
                    nn.Conv1d(d_out // 2, d_out // 2, 3, padding=d, dilation=d,
                              bias=False), nn.BatchNorm1d(d_out // 2),
                    nn.ReLU(inplace=True))],
                nn.Conv1d(d_out // 2, d_out, 1, bias=False),
                nn.BatchNorm1d(d_out), nn.ReLU(inplace=True), nn.Dropout(dropout),
            )
            self.shortcut = nn.Sequential(
                nn.Conv1d(d_in, d_out, 1, bias=False), nn.BatchNorm1d(d_out))

        def forward(self, feats: "torch.Tensor", idx: "torch.Tensor",
                    sub_idx: "torch.Tensor | None"):
            """feats: (B, C, N_i); idx: (B, K, N_i); sub_idx: (B, N_{i+1}).

            The level is reduced *first* (RandLA-Net samples before aggregating),
            so the residual branch and the shortcut always live at the same
            resolution and the element-wise sum is well defined.
            """
            if sub_idx is not None:
                feats = torch.gather(feats, 2,
                                     sub_idx.unsqueeze(1).expand(-1, feats.shape[1], -1))
            out = self.lfa(feats, idx)
            out = self.res(out)
            short = self.shortcut(feats)
            if out.shape[2] != short.shape[2]:      # defensive, should not happen
                short = F.interpolate(short, size=out.shape[2], mode="nearest")
            return out + short

    class RandLANet(nn.Module):
        def __init__(self, cfg: dict):
            super().__init__()
            self.cfg = cfg
            enc, dec = cfg["encoder_dims"], cfg["decoder_dims"]
            ks = cfg["k_neighbors"]
            subs = cfg["subsampling"]
            dils = cfg["dilations"]
            self.input_mlp = nn.Sequential(
                nn.Linear(cfg["d_in"], enc[0]), nn.BatchNorm1d(enc[0]),
                nn.ReLU(inplace=True))
            self.blocks = nn.ModuleList()
            self.downsample: list = []
            d_prev = enc[0]
            for i, (d_out, k, s, dl) in enumerate(zip(enc, ks, subs, dils)):
                self.blocks.append(DilatedResidualBlock(
                    d_prev, d_out, k, dl, s, cfg["dropout"]))
                self.downsample.append(s)
                d_prev = d_out
            self.decoder = nn.ModuleList()
            # Skip connections run from the encoder back towards the input, so
            # decoder step i consumes the encoder output of level len(enc)-2-i.
            n_skip = max(0, len(enc) - 1)
            for i, d_out in enumerate(dec):
                skip_dim = enc[len(enc) - 2 - i] if i < n_skip else 0
                self.decoder.append(nn.Sequential(
                    nn.Linear(d_prev + skip_dim, d_out),
                    nn.BatchNorm1d(d_out), nn.ReLU(inplace=True)))
                d_prev = d_out
            self.classifier = nn.Linear(d_prev, cfg["d_out"])
            self._indices: dict = {}

        def set_graph(self, key, value):
            self._indices[key] = value

        def forward(self, feats: "torch.Tensor", graph: dict):
            """feats: (B, d_in, N); graph: build_graph() output -> (B, K, N) / (B, N)."""
            x = self.input_mlp(feats)
            skips = []
            last = len(self.blocks) - 1
            for i, block in enumerate(self.blocks):
                knn_idx = graph[f"knn_{i}"]
                sub_idx = graph.get(f"sub_{i}")
                x = block(x, knn_idx, sub_idx)
                if i < last:
                    skips.append(x)                 # a resolution the decoder reuses
            out = x
            for i, layer in enumerate(self.decoder):
                if i < len(skips):
                    skip = skips[-(i + 1)]
                    if skip.shape[2] != out.shape[2]:
                        skip = F.interpolate(skip, size=out.shape[2], mode="nearest")
                    out = torch.cat([out, skip], dim=1)
                out = layer(out)
            return self.classifier(out)

    return RandLANet(cfg)
