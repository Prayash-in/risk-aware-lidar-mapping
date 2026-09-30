# RandLA-Net weights

This directory is a placeholder for a trained checkpoint.

**Nothing is shipped here** — no weights, no accuracy claims. The inference
wrapper (`RandLANetSegmenter`) refuses to run without a checkpoint and without
`torch`, and the pipeline then truthfully reports `ground_truth` or `unknown`
semantics instead of a fake prediction.

To enable real inference:

1. Install torch: `pip install .[randlanet]`
2. Place a checkpoint whose `state_dict` matches the architecture described in
   `../docs/RANDLANET.md` at `checkpoint.pth` (or pass any path via
   `--checkpoint` / `RandLANetSegmenter(checkpoint=...)`).
3. Run with `--use-model`: `python -m adaptive_lidar run --use-model`.

The checkpoint contract (exact tensor shapes) is defined in
`../src/adaptive_lidar/perception/randlanet/model.py` (`DEFAULT_CONFIG`), and a
mismatched state dict is rejected loudly rather than loaded with missing keys.