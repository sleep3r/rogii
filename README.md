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

Upload the unpacked data to ClearML Dataset:

```bash
make upload-clearml-data
```

Defaults:

```text
PROJECT_NAME=ROGII/Wellbore
OUTPUT_URI=s3://s3-basket-cold.wb.ru/ds-experiments
CLEARML_DATA_PROJECT=$(PROJECT_NAME)
CLEARML_DATA_NAME=rogii-wellbore-geology-prediction
CLEARML_DATA_VERSION=20260519_s3
CLEARML_DATA_OUTPUT_URI=$(OUTPUT_URI)
```

The upload includes the unpacked `train/`, `test/`, public sample directories,
`sample_submission.csv`, and the PPTX. Kaggle zip archives are excluded.
`make upload-clearml-data` requires an `s3://` output URI, because the ClearML
fileserver is too slow for this dataset size. To resolve the dataset locally:

```bash
make clearml-data-local-path
```

## Training

Fast smoke test on the public sample:

```bash
make quick-train
```

Replay the public drift-targeting + NCC tree notebook path:

```bash
make drift-ncc-quick
make drift-ncc-train
```

See `DRIFT_NCC_NOTEBOOK.md` for the Kaggle notebook mapping and the private
artifact caveat.

Run the deterministic local FormationPlaneKNN surface solver experiment:

```bash
make formation-plane-knn-quick
make formation-plane-knn
```

See `FORMATION_PLANE_KNN.md` for outputs and validation notes.

Run the A2 target-free selector/scorer on generated FormationPlaneKNN OOF
candidates:

```bash
make formation-selector-quick
make formation-selector
```

Pass an OOF schema10 file for safety blends when available:

```bash
make formation-selector FORMATION_SELECTOR_SCHEMA10=artifacts/schema10_oof_raw.parquet
```

Run the B-lite GR/NCC scorer over A candidates:

```bash
make formation-b-lite-quick
make formation-b-lite
```

Run the B2 constrained reranker over existing A + B-lite artifacts:

```bash
make formation-b2-quick
make formation-b2
```

If the old schema10 OOF parquet is gone, rebuild a compatible row-level OOF
baseline from the saved model artifact:

```bash
make export-oof-baseline \
  OOF_BASELINE_MODEL_DIR=artifacts/clearml/c11ac4df327f49f3b91ad293c69bc91e \
  OOF_BASELINE_OUTPUT=artifacts/oof_baseline/schema10_oof.parquet
```

Full A2/B2 chain on the current full artifacts:

```bash
make formation-b2 \
  FORMATION_B2_INPUT=artifacts/formation_plane_knn_a2_full/oof_candidates.parquet \
  FORMATION_B2_B_SCORES=artifacts/formation_b_lite_a2_full/b_candidate_scores.parquet \
  FORMATION_B2_OUTPUT=artifacts/formation_b2_a2_full_schema10 \
  FORMATION_B2_SCHEMA10=artifacts/oof_baseline/schema10_oof.parquet \
  FORMATION_B2_SCHEMA10_COLUMN=schema10_oof_pp
```

Search guarded B2 policies and freeze the current bounded submit candidate:

```bash
make formation-b2-guarded \
  FORMATION_B2_OUTPUT=artifacts/formation_b2_a2_full_schema10 \
  FORMATION_B2_GUARDED_INPUT=artifacts/formation_plane_knn_a2_full/oof_candidates.parquet \
  FORMATION_B2_GUARDED_OUTPUT=artifacts/formation_b2_danger_guard_a2_full_schema10 \
  FORMATION_B2_GUARDED_SCHEMA10=artifacts/oof_baseline/schema10_oof.parquet \
  FORMATION_B2_GUARDED_SCHEMA10_COLUMN=schema10_oof_pp
```

Replay the frozen inference policy and verify exact parity against the guarded
OOF artifact:

```bash
make formation-b2-infer-oof-replay \
  FORMATION_B2_INFER_INPUT=artifacts/formation_plane_knn_a2_full/oof_candidates.parquet \
  FORMATION_B2_INFER_CHOICES=artifacts/formation_b2_a2_full_schema10/b2_selector_choices.csv \
  FORMATION_B2_INFER_METADATA=artifacts/formation_b2_a2_full_schema10/b2_candidate_metadata.parquet \
  FORMATION_B2_INFER_OUTPUT=artifacts/formation_b2_infer_oof_replay \
  FORMATION_B2_INFER_SCHEMA10=artifacts/oof_baseline/schema10_oof.parquet \
  FORMATION_B2_INFER_SCHEMA10_COLUMN=schema10_oof_pp \
  FORMATION_B2_INFER_REFERENCE=artifacts/formation_b2_danger_guard_a2_full_schema10/guarded_predictions.parquet
```

Apply the frozen B2 correction to hidden test rows after producing a base
submission from the same c11/schema10 artifact:

```bash
make formation-b2-infer-test \
  FORMATION_B2_TEST_BASE_SUBMISSION=artifacts/kaggle_submit_output/submission.csv \
  FORMATION_B2_TEST_OUTPUT=artifacts/formation_b2_test_submit
```

Main local training:

```bash
make train-local
```

Equivalent command:

```bash
uv run python -m rogii --config configs/stack.yml
```

Server training through spacebridge:

```bash
cp portainer.yml.example portainer.yml
# edit portainer.yml: ClearML config path, Registry username, Portainer instance, GPU ids
make check-server-env
make train-server INSTANCE=gpu CONFIG=configs/stack_gpu.yml
```

With Colima, make sure Docker has the Buildx CLI plugin. If `docker buildx
version` fails, install and link it:

```bash
brew install docker-buildx
mkdir -p ~/.docker/cli-plugins
ln -sfn "$(brew --prefix)/opt/docker-buildx/bin/docker-buildx" ~/.docker/cli-plugins/docker-buildx
docker buildx version
```

`make train-server` builds the Docker image, injects `~/clearml.conf` as a
BuildKit secret, pushes the image, and starts it through Portainer. By default
it passes `config=$(CONFIG)`, enables ClearML tracking, and enables
`data.clearml`, so the container downloads the ClearML Dataset instead of
packing local `data/` into the image. The Docker context excludes `.venv`,
`artifacts`, mining output, and `data/`.

For GPU server runs, pass `CONFIG=configs/stack_gpu.yml` explicitly. It is the
same stack as `configs/stack.yml`, but with parallel feature prep and explicit
GPU model params:

- features: `num_workers: 8`;
- CatBoost: `task_type=GPU`, `devices=0`;
- LightGBM: `device_type=gpu`, `gpu_device_id=0`, `gpu_use_dp=false`,
  `max_bin=63`.

Training logs print `Model backend config` before every model fit, so the actual
backend is visible in ClearML/Portainer logs. The Docker image installs
`clinfo`, writes an NVIDIA OpenCL ICD file, and runs `rogii.gpu_preflight`
before training. If CUDA/OpenCL is not visible to the container, the job fails in
the first seconds with `nvidia-smi`, `clinfo`, or LightGBM smoke-test output
instead of dying after feature preparation.

The command intentionally does not pass `--image-name`; spacebridge reads
`portainer.yml` and handles Harbor naming/tagging from `REGISTRY_USERNAME`.

Useful overrides:

```bash
make train-server INSTANCE=gpu CONFIG=configs/stack_gpu.yml SERVER_NOTES=fold_avg_v1
make train-server INSTANCE=gpu CONFIG=configs/stack.yml SERVER_NOTES=cpu_models
make train-server INSTANCE=gpu CLEARML_TASK_NAME=rogii_fold_avg
```

Keep `SERVER_NOTES` and other spacebridge `key=value` args without spaces; the
spacebridge parser is intentionally simple.

ClearML is opt-in for normal local runs:

```bash
uv run python -m rogii --config configs/quick.yml --clearml_enabled=true
ROGII_CLEARML_ENABLED=true make train-local
```

The ClearML task records the resolved config, scalar metrics, `metrics.json`,
`features.json`, `config.yml`, `submission.csv`, and every file under the run
output directory. On server runs `CLEARML_LOG_MODEL=true` by default, so
`model.pkl` is uploaded as well. All of this uses the task
`output_uri`, which defaults to `s3://s3-basket-cold.wb.ru/ds-experiments`.

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

The feature table includes PF_ANCC/PF_Z, beam paths, multi-scale NCC,
multi-radius DTW, stochastic DTW uncertainty, DWT-lowpass DTW, spatial formation
planes, dense ANCC calibration, and offset GR residuals (`tda*`, `tdbc*`,
`tdsc*`, `tdpf*`, `tddtw*`). After feature changes, old `artifacts/stack`
models are invalid; run `make train-local` again before `make submit`.

Feature cache has two layers. Context-free alignment/PF/typewell/GR features are
cached once per well and reused across fold-safe contexts. Context-dependent
spatial/dense ANCC and aggregate signal features include the
`KaggleTopContext` key, so fold-safe OOF and full-context inference cannot reuse
the wrong pickle by accident. `features.num_workers` controls per-well
process-based feature preparation; keep it at `1` for stage-level profiling and
raise it on the server when memory allows.

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
metadata. ClearML is the experiment registry for comparing serious runs.

## Kaggle

Remote Kaggle training is intentionally disabled:

```bash
make train-kaggle
```

The full `stack.yml` train run exceeds Kaggle's 9-hour CPU notebook limit. The
working path is local training followed by an inference-only Kaggle run.

Inference-only Kaggle run from the current c11 ClearML artifact:

```bash
make submit MESSAGE="c11 b2 guarded infer"
```

By default, `make submit` downloads model artifacts from ClearML task
`c11ac4df327f49f3b91ad293c69bc91e` into
`artifacts/clearml/<CML_ID>/`, publishes that directory as a private Kaggle
dataset, pushes a private Kaggle script, waits for `submission.csv`, validates
it, and stops before the competition submit API call. Submit the notebook
version manually from the Kaggle UI when you are happy with it.

`make submit` also applies the frozen B2 guarded correction from
`configs/formation_b2_guarded_submit.yml` inside the Kaggle inference kernel.
To submit the plain base model instead:

```bash
make submit B2_SUBMIT_CONFIG=
```

Use another ClearML run:

```bash
make submit CML_ID=<clearml_task_id> MESSAGE="new run infer"
```

Fetch the model without touching Kaggle:

```bash
make fetch-clearml-model CML_ID=<clearml_task_id>
```

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
