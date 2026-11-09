# EventNeRF debris reconstruction

This repository is an independent, compact implementation of the EventNeRF
training objective for a Davis346-style grayscale event stream. It is organized
like `event_snn_torch`: JSON configuration, small Python modules, shell entry
points, GitHub Actions, W&B logging, and a dated `runs/` report.

The scientific reference is **EventNeRF: Neural Radiance Fields From a Single
Colour Event Camera** (Rudnev et al., CVPR 2023). The authors' official code is
<https://github.com/r00tman/EventNeRF>. This repository is not the official
implementation and does not claim parity with the paper's colour Bayer model.

## What is implemented

- Known-pose NeRF volume rendering in the target-body coordinate system.
- The EventNeRF loss
  `log(I(t1)) - log(I(t0)) = sum(polarity * contrast_threshold)`.
- The official implementation's configurable sRGB-to-linear approximation
  `log(rgb^2.2 + epsilon)` is used before the event difference.
- Random temporal windows, ray jitter, event-positive rays, and 10% zero-event
  negative rays after warm-up.
- Grayscale supervision matching the current synthetic Davis346 data.
- RGB frames are **never imported by training code**. They are evaluation-only.
- One log-intensity affine map is fitted on `validation_rgb` frames and frozen
  before `test_rgb`, avoiding test-set calibration leakage.
- Density-mesh extraction, metric-frame bidirectional Chamfer, accuracy,
  completeness, and F-score at 0.10/0.25/0.50 m. No ICP or similarity alignment
  is applied.
- Novel-image PSNR, SSIM, foreground PSNR/MAE, and LPIPS-Alex when available.
- W&B logging in online, offline, or disabled mode.

## Dataset compatibility and the important limitation

The original `20261031_debris_rotation_v3_davis346` data contains a rotating
target, moving camera, Earth, and stars. That violates EventNeRF's static-scene
and constant-background assumptions, and its source RGB frames were intentionally
removed after event generation. Training directly on those files would not be a
valid EventNeRF evaluation.

`space_event_blender` therefore receives only new files:

- `generate_eventnerf_dataset.py`
- `blender_eventnerf_scene.py`
- `prepare_eventnerf_scene.py`
- `examples/configs/eventnerf_debris_davis346.json`
- `examples/bash/generate_eventnerf_dataset.sh`

They preserve the same debris geometry, Davis346 intrinsics/resolution, 128 Hz
source rendering, 100–200 deg/s relative view rate, and sampled solar lighting.
For method validity, the target and illumination are static, the background is
constant, and the camera performs the equivalent orbit. Existing generators and
datasets are not modified.

## Run on the RTX 4070 PC

Generate one deterministic 1.2-turn scene:

```bash
cd /home/azumayudai/Desktop/ws_blender/space_event_blender
bash examples/bash/generate_eventnerf_dataset.sh
```

Train and evaluate using the already available PyTorch 2.8 CUDA image:

```bash
cd /home/azumayudai/Desktop/ws_snn/event_nerf
WANDB_MODE=online bash scripts/train_4070.sh
```

Use `WANDB_MODE=offline` when no W&B API key is available. The GitHub workflow
uses the repository's `WANDB_API_KEY` secret and the persistent 4070 self-hosted
runner.

The default output is `runs/1110/` and includes checkpoints, CSV/JSON metrics,
the reconstructed PLY mesh, point clouds, training curves, held-out photometric
views, density slices, and 3D error plots.

## Data leakage contract

Training may read only:

- `events.npz`
- `transforms.json` poses and intrinsics
- event thresholds from `meta/metadata.json`

The following are evaluation-only and are deliberately loaded only by
`event_nerf.evaluate`:

- `frames/*.png`
- `bbox_masks/*.png`
- `gt_target_mesh.ply`

This is a per-scene optimization method, so the events cover the trajectory that
is reconstructed. "Held-out RGB" means the intensity images were never training
targets; it does not mean their timestamps were absent from the event stream.

## License and citation

The repository uses CC BY-NC-SA 4.0 to match the official EventNeRF code. If this
implementation contributes to a publication, cite Rudnev et al. (CVPR 2023) and
clearly identify this independent grayscale debris adaptation.
