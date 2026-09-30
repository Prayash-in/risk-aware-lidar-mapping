# Architecture and data model

## Pipeline

Everything flows through `AdaptiveLidarMapper.process_frame` in
`src/adaptive_lidar/pipeline.py`. One mapper instance owns one map and carries
temporal state (dynamics memory, tracks, EMA, hysteresis state).

```
PointCloud
   │
   ▼ 1. preprocess          range filter; ego-frame (+x forward)
   ▼ 2. semantics           RandLA-Net / ground truth / unknown  (provenance kept)
   ▼ 3. dynamics            ChangeField signature vs. short memory -> dynamic score
   ▼ 4. tracking            change-driven clusters -> KalmanCV tracks           │
   ▼ 5. prediction          bubbles + constant-velocity corridor                │
   ▼ 6. controller          factors: range, semantic, dynamic, bubble, prediction
   │                          -> hysteresis decision -> target cell sizes
   ▼ 7. projection_2_5d     variable-resolution quadtree leaves + statistics
   ▼ 8. budget              deterministic coarsening (never deletion) -> leaves
   │
   ▼ FrameResult(leaves, tracks, metrics, extras)
```

The runtime of stages 2-6 is independent of the resolution policy; only stage 7
uses the policy, and only stage 8 writes resolution back down. `compare_modes`
runs the same input through stages 7 of `uniform` / `distance` / `adaptive` with
identical temporal state — that is what makes cell counts comparable.

## 2.5D representation

There are **no 3D voxels**. A map cell is a ground-plane rectangle of edge
`size`, addressed by `(depth, ix, iy)` where `size = extent_m / 2**depth`. It
stores 2.5D statistics: point count, height mean/variance/min/max, a semantic
vote + confidence, and the five risk scores.

| quantity | definition |
|----------|-----------|
| depth    | `log2(extent_m / size)` |
| uniform  | `project_2_5d(..., target_size = uniform_cell_size)` |
| distance | `s(d) = s_min + (s_max - s_min) * exp(-lambda * d)` |
| adaptive | per-point target size from the controller, plus `depth_shift` |

Per-frame projection rebuilds the leaf set with vectorized numpy reductions;
`Quadtree` in `mapping/quadtree.py` is the explicit parent/child graph used for
single-cell operations, split/merge tooling and unit tests. Splits inherit a
parent's statistics provisionally; the next projection re-accumulates.

## Importance model

The controller fuses five factors into a per-point importance in
`controller/importance.py`:

```
total = w_r * I_range + w_d * I_dynamic + w_s * I_semantic
      + w_b * I_bubble + w_p * I_prediction        (enabled weights renormalized)
I_range   = exp(-d / range_scale)
I_dynamic = decaying change-detection heat (DynamicScorer)
I_semantic= config table per class (documented heuristic, not learned)
I_bubble  = 1 inside r_bubble(track), halo-decayed outside
I_prediction = 1 inside the constant-velocity corridor, halo-decayed outside
```

Split/merge thresholds (`split_threshold=0.60`, `merge_threshold=0.30`) and
person-class persistence (`split_frames=2`, `merge_frames=3`) turn factor
evidence into structural split/merge decisions via reference depth 9, with EMA
`alpha=0.5` smoothing. Merge-proposal evidence is tracked per cell, so a cell
that wants to merge never counts its own evidence toward a *split*.

## Compute budget

Each leaf costs a deterministic unit:

```
cost(cell) = 2**depth * (1 + point_count / 8) * rate(tier, cell)
```

rates: hot 1.0, warm 0.5, cold 0.15 (a passive far cell is 6.7× cheaper to keep
than a hot one). The budget manager:

1. computes the total over all leaves (vectorized);
2. if over budget, repeatedly merges the four children of the lowest-importance
   group whose merged statistics preserve the parent's aggregates (no deletion);
3. never treats a partially-present group as mergeable;
4. protects every cell in a track's bubble ancestor chain, down to the
   budget floor level;
5. stops at the floor and reports `floor_reached=True` if the budget is still
   not met (deterministic and auditable).

`BudgetReport` returns pre/post cost, allowed, leaves, merged groups, protected
groups, floor state and a post-budget memory estimate.

## Data model / provenance

- `PointCloud`: one ego-frame scan — N points, `intensity`, `semantic`,
  `confidence`, timestamp, pose, and `semantic_source` (`unknown`,
  `ground_truth`, `model_prediction`). Length validation and fill-in happen in
  `__post_init__`.
- `PointCloudSource`: read-only frame-addressed backend; synthetic and
  SemanticKITTI implement `describe()` with their true source capability.
- `FrameResult.extras["semantic_source"]` records where each frame's labels came
  from; benchmarks export it so GT is never mistaken for prediction.
- Memory figures are a **documented structural estimate**:
  `CELL_BYTES = 64` per leaf in `mapping/map_cell.py` plus
  `post_internal * CELL_BYTES` implied internal nodes. They are labelled
  "Estimated", not RSS.

## Why measured, not simulated

`Metrics` fields (`frame_latency`, `controller_latency`, `fps`) are produced by
`time.perf_counter` inside the pipeline. Budget cost and cell counts are exact
arithmetic on the returned leaves. Benchmarks write the raw JSON/CSV into
`results/` with `"label": "Measured"`; no number is invented or back-filled.