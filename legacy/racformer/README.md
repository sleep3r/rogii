# RAC-Former v1

Residual Anchored C-field Transformer for TVT prediction in horizontal oil/gas wells.

## Core idea

```
pred_tvt = prior_tvt + cumsum(segment_slopes) + direct_residual
```

- `prior_tvt`: deterministic baseline (b2 if available, hengck fallback)
- `segment_slopes`: K=16 learned ft/row corrections, materialized via the same
  cumulative-sum design matrix as the residual oracle
- `direct_residual`: per-step ft adjustment, anchor-zeroed (disabled in
  sanity config until v2)

All output heads are zero-initialized so at step 0 `pred_tvt == prior_tvt`
exactly — every gain over the prior is learned residual.

## Quickstart

```bash
cd racformer
make install
make train-smoke              # CPU, 5 wells, 3 epochs (sanity wiring check)
make train-sanity             # Full segment-only run, 5 folds, 80 epochs
make infer
```

Or from the repo root:

```bash
make -C racformer train-sanity
```

## ClearML

Dataset is already uploaded as
`ROGII/Wellbore / rogii-wellbore-geology-prediction / 20260519_s3`.

```bash
make clearml-data-local-path  # pre-warm the local cache
make train-sanity              # tracking.enabled=true in YAML by default
```

To pull the dataset directly from ClearML inside the run, flip
`data_clearml.enabled: true` in the config (or export
`RACFORMER_DATA_CLEARML_ENABLED=true`).

To disable tracking for a one-off run:

```bash
RACFORMER_CLEARML_ENABLED=false make train-sanity
```

## Remote GPU (spacebridge)

```bash
make check-server-env
make train-server INSTANCE=el        # or xc — see portainer.yml
```

## Docker

```bash
make docker-build
make docker-run CONFIG=racformer/configs/racformer_v1.yml
```

## Layout

```
racformer/
  config.py            # dataclasses, YAML loader, env overrides
  dataset.py           # feature engineering, WellSample, RACDataset, oracle
  model.py             # InputProj → ConvStem → Encoder → SegmentQueryDecoder → 5 heads
  loss.py              # 9 normalized loss terms
  train.py             # 5-fold GroupKFold, AdamW, cosine LR, EMA, ClearML hooks
  infer.py             # TTA × 5-fold ensemble → submission CSV
  clearml_tracking.py  # slim ClearML Task wrapper (NullTracker fallback)
  clearml_data.py      # ClearML Dataset upload/download CLI
  configs/
    racformer_smoke.yml     # CPU smoke
    racformer_sanity.yml    # Segment-only, ClearML on, all aux losses off
    racformer_v1.yml        # Full model
  Dockerfile           # PyTorch CUDA 12.4 + uv, multi-stage
  Makefile             # all training / Docker / spacebridge commands
  pyproject.toml       # racformer deps (torch, clearml, spacebridge, ...)
  portainer.yml        # GPU instance keys for spacebridge
```

## See also

- `../old/` — legacy CatBoost pipeline that produced the b2 prior
  (`b2_guarded_submit`) and the schema10 OOF baseline. RAC-Former consumes
  these as `prior_tvt`.
