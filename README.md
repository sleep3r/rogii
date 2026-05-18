# ROGII - Wellbore Geology Prediction

Local framework for the Kaggle competition
`rogii-wellbore-geology-prediction`.

The goal is to predict hidden `TVT` intervals in horizontal wells. The repo is
intentionally small: one training pipeline, one inference pipeline, two configs.

## Setup

```bash
make install-deps
```

Kaggle auth uses the access-token flow:

```bash
mkdir -p ~/.kaggle
echo KGAT_xxx > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
uv run kaggle competitions files -c rogii-wellbore-geology-prediction
```

## Data

```bash
make unzip-data
```

Expected layout:

```text
data/
  train/
  test/
  sample_submission.csv
```

`make train-local` runs `ensure-data`, so it unzips the Kaggle archive if
`data/train` or `data/test` is missing.

## Training

Fast smoke test on the public sample:

```bash
make quick-train
```

Main local training:

```bash
make train-local
```

Equivalent command:

```bash
uv run python -m rogii --config configs/stack.yml
```

The main `stack.yml` pipeline is the local, reproducible version of the public
DWT-style baseline:

1. Build fold-safe OOF features by well: each validation fold gets a
   `KaggleTopContext` built only from that fold's training wells.
2. Train residuals from `last_known_tvt`.
3. Build an OOF matrix from three LightGBM and three CatBoost base models.
4. Fit non-negative hill-climb blend weights.
5. Tune PF_ANCC delta postprocess around the DWT-notebook optimum and optional
   Savitzky-Golay smoothing on OOF.
6. Rebuild full-train context, fit final full-context models on all train wells,
   and save an inference artifact under `artifacts/stack`.
7. Append a machine-readable row to `artifacts/runs.csv`.

The feature table includes PF_ANCC/PF_Z, beam paths, multi-scale NCC,
multi-radius DTW, stochastic DTW uncertainty, DWT-lowpass DTW, spatial formation
planes, dense ANCC calibration, and offset GR residuals (`tda*`, `tdbc*`,
`tdsc*`, `tdpf*`, `tddtw*`). After feature changes, old `artifacts/stack`
models are invalid; run `make train-local` again before `make submit`.

Feature cache keys include a schema version and the `KaggleTopContext` key, so
fold-safe OOF features and full-context inference features cannot reuse the same
pickle by accident.

Spatial distance features `kg_form_knn_dist` and `kg_dense_ancc_dist` are
stored in normalized KD-tree units, not feet. They are meant as relative
neighborhood-confidence features.

Artifacts:

```text
artifacts/stack/
  model.pkl
  features.json
  metrics.json
  config.yml
  source_config.yml
```

`metrics.json` includes OOF model scores, blend weights, selected postprocess
parameters, per-well diagnostics, hidden-length slices, and feature schema
metadata. `artifacts/runs.csv` is the compact experiment registry for comparing
serious runs.

## Kaggle

Remote Kaggle training is intentionally disabled:

```bash
make train-kaggle
```

The full `stack.yml` train run exceeds Kaggle's 9-hour CPU notebook limit. The
working path is local training followed by an inference-only Kaggle run.

Inference-only Kaggle run from a local artifact:

```bash
make submit MESSAGE="baseline infer"
```

That command publishes the trained local artifact as a private Kaggle dataset,
pushes a private Kaggle script, waits for `submission.csv`, validates it, and
also stops before the competition submit API call. Submit the notebook version
manually from the Kaggle UI when you are happy with it.

Dry runs:

```bash
make submit-dry MESSAGE="infer dry run"
```

Monitor kernels:

```bash
make status-train
make logs-train
make status-submit
make logs-submit
```

## Configs

```text
configs/quick.yml  # tiny public-sample smoke test
configs/stack.yml  # current baseline experiment and submit artifact source
```

`configs/stack.yml` is the working baseline. To try ideas, edit it directly or
copy it temporarily while experimenting. Once an experiment wins, make that the
new `stack.yml`.

## Research Notes

Public notebook/discussion mining lives outside the model path:

```bash
make mine-code
make mine-discussions
make model-bundle
```

The generated model bundle is written to `.kaggle_mining/model_bundle.md`.
It includes `COMPETITION.md`, `CHANGELOG.md`, mined public notebooks,
discussion-derived ideas, current configs, metrics, and the current solution
code snapshot.

## Checks

```bash
make format
make check
uv run python -m compileall rogii
```

Experiment history and leaderboard results are tracked in `CHANGELOG.md`.
