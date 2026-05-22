# MTPNet Research Prototype Design

Date: 2026-05-22

## Purpose

MTPNet is a new standalone research project for ROGII-style wellbore geology prediction. The first milestone is a strong research prototype, not a notebook-only experiment: it must prove that a CNN/MTP model can learn local multi-trajectory stratigraphic inversion from log-misfit heatmaps, while keeping the project structure ready for later fold-safe OOF, Docker, Kaggle packaging, and sequential tracking.

The prototype is based on the MTP/MDN direction in `research_101.md` and the bundled resources:

- geosteering inversion is multi-modal, not a single deterministic TVT regression;
- the core input is a 2D horizontal-log versus typewell-log mismatch heatmap;
- the core output is `K` possible future trajectories plus logits/probabilities;
- the core training objective is MTP loss: best-mode regression plus mode classification;
- later inference should roll forward sequentially and keep multiple realizations.

## Project Boundary

`MTPNet/` is the standalone project root. The existing `old/` and `gbm/` folders are reference material only. We may reuse ideas, metrics, Docker/Makefile patterns, and small helper logic from them, but MTPNet must not import `old.rogii` or depend on the old package layout.

The initial source package will be:

```text
MTPNet/
  pyproject.toml
  Makefile
  configs/
    mtp_smoke.yml
    mtp_v0.yml
  mtpnet/
    __init__.py
    config.py
    io.py
    heatmap.py
    windows.py
    model.py
    loss.py
    train.py
    eval.py
    cli.py
  tests/
```

`track.py`, `infer.py`, Docker files, and fold-safe OOF orchestration are planned extensions, not required for the first runnable prototype.

## Data Design

The prototype reads Kaggle-style well files:

```text
<well_id>__horizontal_well.csv
<well_id>__typewell.csv
sample_submission.csv
```

Config controls the data source and sample size:

```yaml
data:
  data_dir: data
  train_dir: data/train
  test_dir: data/test
  k_wells: -1
  copy_from:
```

`k_wells` semantics:

- `-1`: use all wells in the selected train directory;
- positive integer: use the first `k` wells after stable sorting by well id;
- `0`: invalid, fail fast with a clear error.

MTPNet can be self-contained by copying data from `../old/data` into `MTPNet/data`. The code must still work when `data_dir` points elsewhere, so Docker and Kaggle runs can mount or copy data without changing source files.

## Window Builder

The first prototype trains on local windows rather than full wells. Each window is built around a known start point, compressed along the horizontal well, and cropped around a typewell vertical center.

Default v0 window settings:

```yaml
window:
  rows_per_step: 32
  history_steps: 8
  future_steps: 16
  vertical_bins: 64
  vertical_radius_ft: 160.0
  stride_steps: 4
  max_windows_per_well: 64
```

For each window:

- compress horizontal GR and TVT by averaging every `rows_per_step`;
- choose a typewell crop centered near the current TVT anchor;
- create a `[C, H, W]` tensor, where `H=vertical_bins` and `W=history_steps + future_steps`;
- create a target path of length `future_steps` in vertical-bin coordinates;
- keep metadata: `well_id`, row range, center TVT, compression settings, and target TVT values.

The smoke config may reduce `k_wells`, `max_windows_per_well`, epochs, and model size, but must exercise the same code path as full v0.

## Input Channels

The minimum channel set is intentionally small but extensible:

1. `gr_diff`: horizontal GR minus typewell GR.
2. `abs_gr_diff`: absolute GR mismatch.
3. `history_mask`: drawn path for known history steps.
4. `history_sdf`: signed/normalized distance to the known history path.
5. `finite_mask`: valid-data mask for interpolated or missing GR values.

The channel builder should be registry-like enough to add later channels without rewriting the dataset:

- local NCC score;
- horizontal/typewell broadcast channels;
- base/B2/A path SDF priors;
- formation or nearby-well priors.

For v0, missing GR is interpolated then edge-filled, and `finite_mask` preserves where the original values were finite.

## Model

The model is a compact PyTorch CNN encoder plus MTP heads:

```text
input [B, C, H, W]
  -> Conv/BN/GELU blocks with pooling
  -> flattened feature vector
  -> dense head
  -> path_head:  [B, K, future_steps]
  -> logit_head: [B, K]
```

Default mode settings:

```yaml
model:
  k_modes: 8
  conv_channels: [16, 32, 64, 128]
  hidden_dims: [512, 1024]
  dropout: 0.05
```

The path output is vertical-bin coordinates. Later versions may add an auxiliary TVT-offset head, but v0 keeps one primary target so the prototype stays debuggable.

## Loss

MTP loss selects the closest predicted trajectory for each sample and only regresses that mode:

```text
error[k] = mean(abs(pred_path[k] - target_path))
best_k = argmin(error)
loss = path_loss(pred_path[best_k], target_path)
     + alpha_cls * cross_entropy(logits, best_k)
     + smooth_lambda * second_diff_penalty(pred_path[best_k])
```

Default loss settings:

```yaml
loss:
  path_loss: mae
  alpha_cls: 0.2
  smooth_lambda: 0.01
```

The implementation must return a metrics dictionary with regression loss, classification loss, smoothness loss, best-mode indices, and per-sample best-mode error for evaluation.

## Commands

The first runnable commands:

```bash
make copy-data
make smoke
make train CONFIG=configs/mtp_v0.yml
make eval RUN_DIR=artifacts/mtp_v0
```

Equivalent Python entry points:

```bash
python -m mtpnet.cli copy-data --source ../old/data --target data
python -m mtpnet.cli train --config configs/mtp_smoke.yml
python -m mtpnet.cli eval --run-dir artifacts/mtp_smoke
```

`copy-data` is a convenience command. Training and evaluation must work without it when the config points directly at an existing data directory.

## Evaluation

The research prototype focuses on window-level proof, not final leaderboard score. It writes artifacts under the configured run directory:

```text
artifacts/<run_name>/
  config_resolved.yml
  checkpoints/best.pt
  metrics.json
  window_predictions.parquet
```

Required metrics:

- `top1_rmse_bins`;
- `weighted_mean_rmse_bins`;
- `oracle_topk_rmse_bins`;
- `oracle_top3_rmse_bins`;
- `best_mode_mae_bins`;
- `classification_accuracy_best_mode`;
- mode usage histogram;
- train/valid split sizes and well counts.

The first success criterion is not that top1 beats B2. The prototype is successful if the target is often represented inside the predicted top-K paths and oracle top-K is clearly better than weighted/top1 single-path outputs. That proves the multi-modal formulation is alive.

## Validation Split

For the research prototype, the default split is stable well-level holdout:

```yaml
validation:
  valid_fraction: 0.2
  seed: 42
```

Windows from the same well must not appear in both train and validation. Full fold-safe OOF is a later milestone, but this first split must already respect well boundaries.

## Error Handling

The CLI should fail fast with clear messages when:

- no horizontal well files are found;
- a horizontal well has no matching typewell;
- `k_wells` is `0`;
- a well has too few known or hidden points to build any window;
- a config references an unknown input channel;
- PyTorch is not installed.

Wells that cannot produce windows may be skipped with a warning, but if all wells are skipped the command must fail.

## Testing

Initial tests should cover the non-GPU core:

- config loading and `k_wells` semantics, including `-1`;
- well file discovery and typewell matching;
- heatmap tensor shape and finite-mask behavior on tiny synthetic data;
- MTP loss chooses the closest mode and backpropagates;
- a one-batch train step returns finite losses.

The smoke command is also part of verification: it should run quickly on a small `k_wells` setting and write a checkpoint plus metrics.

## Extension Path

The prototype is intentionally shaped so these can be added without redesign:

1. Fold-safe OOF training over all wells.
2. Sequential multi-realization tracker.
3. Base/B2/A SDF prior channels from old artifacts.
4. Guarded blend and danger diagnostics.
5. Dockerfile and Kaggle train/infer packaging.
6. Test-time submission generation.

The first implementation must avoid hardcoding notebook constants in model internals. Window shape, `K`, channels, data size, and run paths belong in config.
