# Changelog and Experiment Log

This file tracks both code changes and modeling experiments. Every meaningful run
should leave enough context here to answer three questions later: what changed,
what happened, and what we learned.

## How To Add A Run

Use this shape for new entries:

```text
### EXP-YYYYMMDD-N - Short name

- Command/config:
- Data:
- Local result:
- Kaggle result:
- What changed:
- Takeaway:
- Next:
```

Use `Kaggle result: pending` until the submission is scored.

## 2026-05-17

### Framework Bootstrap

- Created the local ROGII competition framework with `uv`, `pyproject.toml`,
  configs, package CLI, and Makefile commands.
- Replaced the original large `train.py` flow with package modules under
  `rogii/` and `python -m rogii`.
- Added structured config files:
  - `configs/quick.yml` for tiny public-sample checks;
  - `configs/stack.yml` for the main GBM stack training path;
  - `configs/hgb.yml` for the legacy sklearn HGB baseline;
  - `configs/submit.yml` for Kaggle notebook runs without local CV;
  - `configs/best.yml` as the selected default submit config.
- Added timed run logging, artifact saving, feature snapshots, metrics JSON,
  and submission writing.

Result: the project can train locally, generate `submission.csv`, and preserve
metrics in `artifacts/<run>/metrics.json`.

### Kaggle and Submission Tooling

- Added data download and unzip targets.
- Fixed Kaggle auth expectations around `~/.kaggle/kaggle.json` versus access
  token setup.
- Added end-to-end Kaggle code competition submit flow:
  - package the current source into a Kaggle script workspace;
  - push/run the Kaggle notebook;
  - wait for completion;
  - download and validate `submission.csv`;
  - submit the produced notebook version.
- Added dry-run packaging for submit validation.

Result: `make submit` is the intended end-to-end path, while
`make submit-kaggle-dry` validates packaging without pushing/submitting.

### Research Database

- Added local research mining skills:
  - `kaggle-code-miner` for public notebooks;
  - `kaggle-discussion-miner` for Kaggle discussions;
  - `kaggle-research-brief` for a Markdown synthesis.
- Reworked discussion mining to use Kaggle's official remote MCP endpoint
  `https://www.kaggle.com/mcp`, specifically `list_forum_topics` and
  `get_forum_topic`.
- Added Makefile targets:
  - `make mine-code`;
  - `make mine-discussions`;
  - `make research-brief`;
  - `make research-db`.

Result:

- `kernel` sources in SQLite: 241.
- `discussion` sources in SQLite: 36.
- Top idea buckets currently include `model`, `reported-score`, `feature`,
  `validation`, `alignment`, `data`, `solution`, and `rules`.
- Markdown brief regenerated at `.kaggle_mining/research_brief.md`.

Takeaway: public signals point toward alignment-heavy approaches:
DTW/DWT/correlation against typewells, spatial/geological context, boosting or
ensembles on residuals, and careful validation by wells.

### EXP-20260517-1 - Public-Sample Quick Run

- Command/config: `make quick-train` / `configs/quick.yml`.
- Data: public example split, 3 train wells and 3 public test wells.
- Local result:
  - rows: 14,151;
  - wells: 3;
  - CV folds: 3;
  - CV RMSE: 10.11490;
  - flat baseline RMSE on same rows: 11.40534;
  - final train RMSE: 3.07895;
  - best residual blend weight: 0.75.
- Kaggle result: not submitted.
- What changed: used the compact HGB setup with typewell features and public
  top-solution style signals enabled.
- Takeaway: good smoke test. Not representative enough for model quality
  because it uses only the public sample wells.
- Next: use full `configs/hgb.yml` for real local validation.

### EXP-20260517-2 - Full HGB Local Validation

- Command/config: `make train` / `configs/hgb.yml`.
- Data:
  - train wells: 773;
  - visible test wells locally: 3 public examples;
  - training rows: 3,783,989;
  - features: 122;
  - CV rows: 736,178;
  - CV wells: 150.
- Local result:
  - CV RMSE: 16.63554;
  - flat baseline CV RMSE: 19.06126;
  - final train RMSE: 6.61516;
  - full-train flat RMSE: 17.50671;
  - best residual blend weight: 0.75.
- Fold RMSE:
  - fold 1: 8.73992;
  - fold 2: 25.78129;
  - fold 3: 15.07502;
  - fold 4: 13.13872;
  - fold 5: 14.80105.
- Kaggle result: public LB 12.803 via submit ref `52748619`.
- What changed: full HGB with rolling/tail features, typewell features,
  spatial priors, beam/NCC/DTW signals, and residual blending.
- Takeaway: materially better than flat baseline, but fold spread is large.
  Compared with public discussion/notebook score claims around 9-10 RMSE, this
  is a solid framework baseline rather than a leaderboard-grade solution.
- Next: focus on stronger alignment/sequential/geological priors before tuning
  model hyperparameters.

### EXP-20260517-3 - Kaggle Submit Config

- Command/config: `configs/submit.yml`.
- Data: full train data, Kaggle hidden test expected at runtime.
- Local artifact result:
  - rows: 3,783,989;
  - wells: 773;
  - CV: disabled;
  - train metric computation: skipped;
  - flat train RMSE stored for reference: 17.50671.
- Kaggle result: public LB 12.803 via submit ref `52748619`.
- What changed: inherited `hgb.yml`, disabled CV and expensive reporting for
  the 9-hour Kaggle notebook budget.
- Takeaway: this is a production/submission profile, not an experiment-quality
  validation profile.
- Next: submit after choosing `configs/best.yml` and record public/private LB
  scores here.

### SUBMIT-20260517-1 - First End-to-End Submit Attempt

- Command/config: `make submit MESSAGE="initial hgb submit"` /
  `configs/best.yml`.
- Kaggle kernel: `sleep3r/rogii-hgb-submit`, version 1.
- Result: failed quickly on Kaggle.
- Error: runner expected `train/` directly under
  `/kaggle/input/rogii-wellbore-geology-prediction`.
- Fix: updated the Kaggle runner to search the configured input path, nested
  `/kaggle/input` folders, and zipped competition archives before starting the
  training pipeline.
- Takeaway: Kaggle notebook input layout must be discovered at runtime, not
  assumed from local `data/`.

### SUBMIT-20260517-2 - Submit Retry With Kaggle Data Discovery

- Command/config: `make submit MESSAGE="initial hgb submit"` /
  `configs/best.yml`.
- Kaggle kernel: `sleep3r/rogii-hgb-submit`, version 2.
- Observed Kaggle runtime:
  - top-solution spatial context started successfully;
  - training table loaded all 773 train wells;
  - training table build took about 13 minutes on Kaggle CPU;
  - final model training started after 3,783,989 rows and 122 features;
  - Kaggle kernel completed in 19:08;
  - output `submission.csv` had 14,151 prediction rows.
- Kaggle result:
  - public LB score: 12.803;
  - rank when observed: 682;
  - submit ref: `52748619`;
  - scored from the first successful competition submission.
- Takeaway: the data discovery fix worked; Kaggle CPU is much slower than the
  local machine for the row-building phase. The public score is substantially
  better than the noisy 150-well local HGB CV estimate of 16.63554, so the old
  CV split was pessimistic/noisy; still, 12.803 is behind the public notebook
  direction around 9-10 and validates moving to stack + stronger alignment.

### Remote Kaggle Training Split

- Added `make train-kaggle` for running the training notebook on Kaggle without
  submitting to the competition.
- Added `make train-kaggle-dry` for packaging validation.
- Added `make status-kaggle-train` and `make logs-kaggle-train` for monitoring
  the remote training kernel.
- Added `make status-submit` and `make logs-submit` for monitoring the submit
  kernel.
- Default remote training kernel: `sleep3r/rogii-gbm-stack-train`.
- Default remote training config: `configs/stack.yml`.
- Output directory downloaded locally: `artifacts/kaggle_train_output`.
- Difference from `make submit`:
  - `make train-kaggle` is for remote experiment/validation artifacts;
  - `make submit` is for the final code competition submission path.

### Formatting Toolchain

- Fixed `make format` so it no longer calls global `black`/`isort` binaries.
- Added `ruff` as a project dev dependency in `pyproject.toml` and `uv.lock`.
- `make format` now runs project-local `python -m ruff` for import fixes,
  lint fixes, and formatting.
- Added `make check` for project-local lint checks.
- Cleaned two lint issues found by Ruff:
  - removed unused `anchor_idx` in `rogii/features.py`;
  - reused shared numeric alignment helpers in `rogii/top_signals.py` instead
    of redefining `fill_numeric`.

### EXP-20260517-4 - GBM Stack, DWT, and Particle Signals

- Command/config: `make quick-train` / `configs/quick.yml`.
- Data: public example split, 3 train wells and 3 public test wells.
- Local result:
  - rows: 14,151;
  - features: 113;
  - CV folds: 3;
  - CV RMSE: 9.63946;
  - flat baseline RMSE on same rows: 11.40534;
  - final train RMSE: 0.26473;
  - best residual blend weight: 1.0.
- Kaggle result: not submitted.
- What changed:
  - added LightGBM, XGBoost, CatBoost, and PyWavelets dependencies;
  - added `configs/stack.yml` and made it the default `make train` path;
  - changed `configs/submit.yml` and `configs/best.yml` to point at the stack
    submission profile;
  - implemented residual stacking over LightGBM/XGBoost/CatBoost with Ridge or
    hill-climb blending;
  - kept sklearn `HistGradientBoostingRegressor` only as `configs/hgb.yml`
    legacy baseline;
  - added DWT/wavelet-smoothed typewell alignment features;
  - added PF_Z and PF_ANCC-style sequential consensus features over flat,
    beam, DTW, DWT, spatial-formation, and dense-ANCC candidate tracks.
- Takeaway: on the tiny public smoke split, the stack improves CV from the
  previous HGB quick result of 10.11490 to 9.63946. This is a useful direction
  signal, not leaderboard evidence.
- Next:
  - run full grouped CV on all visible wells with `configs/stack.yml`;
  - add proper OOF stack blending instead of fitting the stack blender on
    in-sample base predictions;
  - calibrate PF process/observation noise and add fold-level diagnostics for
    wells that produce extreme errors.

### EXP-20260517-5 - Full GBM Stack Local Validation

- Command/config: `make train` / `configs/stack.yml`.
- Data:
  - train wells: 773;
  - visible test wells locally: 3 public examples;
  - training/CV rows: 3,783,989;
  - features: 131;
  - CV folds: 5 grouped by well;
  - CV wells: 773.
- Local result:
  - CV RMSE: 13.50285;
  - flat baseline CV RMSE: 17.50671;
  - final train RMSE: 4.64289;
  - full-train flat RMSE: 17.50671;
  - best residual blend weight: 0.75;
  - total runtime: 37:59.
- Residual weight grid:
  - 0.00: 17.50671;
  - 0.10: 16.70812;
  - 0.20: 15.97345;
  - 0.35: 15.01163;
  - 0.50: 14.24790;
  - 0.75: 13.50285;
  - 1.00: 13.50413.
- Fold RMSE:
  - fold 1: 14.89952;
  - fold 2: 15.36182;
  - fold 3: 11.50514;
  - fold 4: 13.28090;
  - fold 5: 12.00888.
- Kaggle result: not submitted yet.
- What changed: full all-well validation of the new LightGBM/XGBoost/CatBoost
  stack with DWT and PF-style top-solution features.
- Takeaway: this is a clear local improvement over the previous HGB CV
  16.63554, with much healthier fold variance than the old 150-well HGB split.
  It should be submitted next; if the old CV pessimism carries over, public LB
  could move materially below the first HGB score of 12.803.
- Next: run `make submit MESSAGE="gbm stack dwt pf cv 13.50"` and record the
  public LB score.

### Inference-Only Submit Path

- Added `rogii.inference` for prediction from a saved model artifact.
- Added Makefile targets:
  - `make infer`;
  - `make prepare-kaggle-infer`;
  - `make submit-infer`;
  - `make submit-infer-dry`;
  - `make status-infer`;
  - `make logs-infer`.
- Default inference artifact: `artifacts/stack`.
- Default inference kernel: `sleep3r/rogii-gbm-stack-submit`.
- What changed:
  - Kaggle submit packaging now supports `--mode infer`;
  - inference mode publishes `model.pkl`, `features.json`, and `metrics.json`
    to a private Kaggle Dataset and attaches it as a dataset source;
  - default model dataset: `sleep3r/rogii-stack-artifacts`;
  - inference restores `postprocess.residual_weight` from artifact CV metrics
    when the config still says `auto`;
  - `save_outputs` now writes the resolved/mutated config to `config.yml` and
    keeps the original YAML as `source_config.yml`.
- Local verification:
  - command: `uv run python -m rogii.inference --config configs/best.yml
    --model-dir artifacts/stack --data-dir data --output-dir artifacts/infer_test
    --submission artifacts/infer_test/submission.csv`;
  - runtime: 6.68s on the 3 public test wells;
  - output matched the full train `submission.csv` byte-for-byte;
  - restored residual weight: 0.75;
  - `make submit-infer-dry` succeeded;
  - first embedded-model inference `run.py` was about 5.3 MB and Kaggle
    rejected it with HTTP 400 on `SaveKernel`;
  - after moving the binary artifacts to a Kaggle Dataset, generated inference
    `run.py` is about 56 KB.
- Kaggle API note:
  - creating a brand-new `rogii-gbm-stack-infer` kernel slug failed with
    `Notebook not found`, even for a bootstrap script with no data sources;
  - inference submit now updates the existing `rogii-gbm-stack-submit` kernel
    instead, preserving previous versions and avoiding new-slug creation.
- Takeaway: inference-only submit skips the expensive Kaggle train feature table
  build and final model fitting. It still builds train-derived spatial context
  and hidden-test features, so it is not free, but should be much faster than
  train+infer for reruns of the same local model.

### SUBMIT-20260517-3 - GBM Stack Inference-Only Submit

- Command/config: `make submit-infer MESSAGE="gbm stack inference cv 13.50"` /
  `configs/best.yml`.
- Model dataset: `sleep3r/rogii-stack-artifacts`.
- Kaggle kernel: `sleep3r/rogii-gbm-stack-submit`, version 2.
- Submit ref: `52751801`.
- Kaggle result: public LB 13.033, not an improvement over the previous 12.803.
- Runtime:
  - model artifact load: 8.56s;
  - spatial context: 23.30s;
  - public test prediction: 8.39s;
  - total inference runtime inside Kaggle: 40.34s;
  - output rows: 14,151.
- What changed: used the local full-stack trained artifact instead of
  rebuilding the 3.78M-row train feature table and refitting the stack on
  Kaggle.
- Takeaway: inference-only submission path is operational, but this stack is
  not better on public LB. Local full CV improved from HGB 16.63554 to stack
  13.50285, while public LB moved from HGB 12.803 to stack 13.033. That means
  the current CV still does not rank submissions correctly. The only runtime
  warning was a sklearn pickle version mismatch (`1.8.0` local vs `1.6.1`
  Kaggle), so future artifacts should either be trained in a Kaggle-compatible
  environment or verified against a Kaggle train+infer run.

### Current Direction

- Do not over-index on train RMSE; the important local number is grouped CV by
  wells.
- First public LB anchor: old HGB submit scored 12.803 at observed rank 682.
- Latest full local CV anchor: GBM stack scored 13.50285 on all 773 wells, but
  public LB was worse than HGB at 13.033.
- The biggest current weakness is likely robust validation and alignment path
  quality, not plain model capacity.
- Next high-value experiments:
  - fix validation so it ranks HGB above the current stack, matching public LB;
  - train a Kaggle-compatible stack artifact or run train+infer once to rule out
    pickle-version drift;
  - add OOF Ridge/hill-climb blending for the base models;
  - improve DTW/DWT/NCC/PF typewell alignment and expose the alignment path
    itself as features;
  - add stronger spatial/geological priors by formation and nearby wells;
  - validate candidate solutions with multiple grouped splits.
