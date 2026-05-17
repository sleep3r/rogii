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

`make train` and `make infer` run `ensure-data`, so they unzip the Kaggle archive
if `data/train` or `data/test` is missing.

## Training

Fast smoke test on the public sample:

```bash
make quick-train
```

Main local training:

```bash
make train
```

Equivalent command:

```bash
uv run python -m rogii --config configs/stack.yml
```

The main pipeline is:

1. Build per-row features from trajectory, GR, `TVT_input`, typewell logs, and
   notebook-style alignment signals.
2. Train residuals from `last_known_tvt`.
3. Fit grouped OOF fold models by well.
4. Build an OOF matrix from LightGBM, XGBoost, and CatBoost base models.
5. Fit non-negative hill-climb blend weights.
6. Tune PF_ANCC delta postprocess and optional Savitzky-Golay smoothing on OOF.
7. Save `submission.csv` plus artifacts under `artifacts/stack`.

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

## Inference

Run local inference from a trained artifact:

```bash
make infer
```

Default model directory is `artifacts/stack`. Override it when needed:

```bash
make infer INFER_MODEL_DIR=artifacts/quick SUBMISSION=artifacts/infer_quick/submission.csv
```

Inference uses `<model-dir>/config.yml`, so it preserves the postprocess selected
during training.

## Kaggle

Dry-run inference submission packaging:

```bash
make submit-infer-dry MESSAGE="baseline dry run"
```

End-to-end inference-only submit:

```bash
make submit MESSAGE="baseline infer"
```

That command publishes the trained local artifact as a private Kaggle dataset,
pushes a private Kaggle script, waits for `submission.csv`, validates it, and
submits the produced notebook version.

Remote Kaggle training without competition submit:

```bash
make train-kaggle MESSAGE="baseline remote train"
```

Monitor kernels:

```bash
make status-kaggle
make logs-kaggle
make status-infer
make logs-infer
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
make research-brief
```

The generated brief is written to `.kaggle_mining/research_brief.md`.

## Checks

```bash
make format
make check
uv run python -m compileall rogii
```

Experiment history and leaderboard results are tracked in `CHANGELOG.md`.
