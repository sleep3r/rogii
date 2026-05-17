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
  - `configs/hgb.yml` for full local training;
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
- Kaggle result: pending.
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
- Kaggle result: pending.
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
- Kaggle result: submitted to competition, ref `52748619`, status pending.
- Takeaway: the data discovery fix worked; Kaggle CPU is much slower than the
  local machine for the row-building phase.

### Remote Kaggle Training Split

- Added `make train-kaggle` for running the training notebook on Kaggle without
  submitting to the competition.
- Added `make train-kaggle-dry` for packaging validation.
- Added `make status-kaggle-train` and `make logs-kaggle-train` for monitoring
  the remote training kernel.
- Added `make status-submit` and `make logs-submit` for monitoring the submit
  kernel.
- Default remote training kernel: `sleep3r/rogii-hgb-train`.
- Default remote training config: `configs/hgb.yml`.
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

### Current Direction

- Do not over-index on train RMSE; the important local number is grouped CV by
  wells.
- The biggest current weakness is likely alignment/geology, not basic tabular
  capacity.
- Next high-value experiments:
  - improve DTW/DWT/NCC typewell alignment and expose the alignment path itself
    as features;
  - add stronger spatial/geological priors by formation and nearby wells;
  - validate candidate solutions with multiple grouped splits;
  - compare HGB against LightGBM/XGBoost/CatBoost once the feature set is more
    competitive.
