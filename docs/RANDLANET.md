# RandLA-Net: status, expected checkpoint, and honest fallback

## Status

RandLA-Net (Engelcke et al., CVPR 2020) is implemented **from scratch for
inference use** in `src/adaptive_lidar/perception/randlanet/`. The wiring in
this repository is complete, but:

* **No trained checkpoint is shipped.** `models/randlanet/` contains this
  README, not weights.
* **`torch` is an optional dependency.** When it is missing,
  `is_torch_available()` returns `False` and the network is never loaded.
* Because of the above, **no accuracy number is claimed** and no checkpoint
  has been evaluated in this environment.

A wrapper that can be unit-tested without torch:
- `knn_graph()` — brute-force KNN in numpy.
- `point_features()` — (N, 6) feature matrix:
  x, y, z, intensity, range (`hypot(x, y)`), local height (`z - mean z`).
- `subsample_indices()` — seeded random subsampling used by the encoder.
- `build_graph()` — full multi-level `knn_<i>` / `sub_<i>` graph for one sample,
  matching the shapes `RandLANet.forward` expects.
- `randla_net_module(cfg)` — lazy torch module; **not imported** when torch is
  absent (keeps torch optional for the rest of the package).

## Expected architecture (must match a checkpoint)

`DEFAULT_CONFIG` in `model.py` defines the exact tensor shapes a checkpoint must
cover; `RandLANetSegmenter.load()` refuses a state dict that leaves missing
keys, so a mismatched checkpoint fails loudly instead of silently using random
weights:

| key | value |
|-----|-------|
| `d_in` | 6 |
| `d_out` | 9 (internal taxonomy) |
| `encoder_dims` | [32, 128, 256, 512] |
| `decoder_dims` | [256, 128, 64, 32] |
| `k_neighbors` | [16, 16, 16, 16] |
| `subsampling` | [4, 4, 4, 4] |
| `dilations` | [1, 1, 1, 1] |
| `dropout` | 0.5 |
| `voxel_size` | 0.06 m |
| `num_points` | 40000 (sliding-window budget) |

A checkpoint trained from this exact architecture can be loaded from
`models/randlanet/checkpoint.pth` (or any path passed as `--checkpoint` /
`RandLANetSegmenter(checkpoint=...)`). `load_state_dict(strict=False)` treats
extra keys as benign and reports them in `load_report["unexpected_keys"]`.

## Inference path

`predict()`:
1. voxel-downsample the scan by `voxel_size`, keeping the largest-intensity
   point per voxel;
2. build `point_features` and the 4-level `build_graph`;
3. network forward → softmax over the taxonomy;
4. nearest-sample assignment back to every original (non-downsampled) point.

## Honest fallback

`SemanticLabeler` picks one of three sources per frame:

1. model prediction — only when a checkpoint + torch are usable;
2. dataset ground truth — when the frame carries labels;
3. `unknown` — otherwise.

The choice is recorded in `FrameResult.extras["semantic_source"]` and in every
exported result and benchmark, so **ground truth is never reported as a
prediction**.

## Reproducing a checkpoint

The silver-standard target is SemanticKITTI seq 08. Training is out of scope for
this repository (it is an inference/mapping core). A checkpoint exported as a
flat or namespaced `state_dict` with keys matching the modules above is all the
inference wrapper needs.

## Tests

`tests/test_randlanet.py` runs the torch-free helper suite always; the two
forward-shape tests are skipped when `torch` is unavailable:

```bash
python -m pytest tests/test_randlanet.py tests/test_perception.py -v
```