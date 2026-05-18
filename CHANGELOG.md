# Changelog and Experiment Log

This file tracks framework changes and modeling experiments. Add every serious
run here with command, data, CV, LB, runtime, and the next decision.

## Template

```text
### EXP-YYYYMMDD-N - Short name

- Command/config:
- Data:
- Local result:
- Kaggle result:
- Runtime:
- What changed:
- Takeaway:
- Next:
```

## 2026-05-18

### EXP-20260518-1 - Global-Context DWT Stack Anchor

- Command/config: `make train` / `configs/stack.yml` before fold-safe
  `KaggleTopContext` landed.
- Data: 773 train wells, 3,783,989 hidden-zone train rows, 419 features.
- Local result:
  - baseline `last_known_tvt` RMSE: 15.90987;
  - best single model: `cat_lr025` OOF RMSE 10.59987;
  - other CatBoost OOF RMSE: 10.61972 / 10.62590;
  - LightGBM OOF RMSE: 10.91615 / 10.95738 / 10.96038;
  - hill-climb OOF RMSE: 10.54106;
  - OOF + postprocess RMSE: 10.51127;
  - selected postprocess: `{alpha: 1.0, tau: 70.0, w_pf: 0.07}` plus
    Savitzky-Golay `{window: 17, polyorder: 3}`;
  - nonzero blend weights: `cat_lr025=0.3900`, `cat_lr020=0.1307`,
    `cat_lr030=0.2434`, `lgb_lr025=0.2359`.
- Runtime: 7h08m local. Postprocess grid alone took 1h44m for 4,608 candidates.
- Kaggle result: not submitted.
- Important caveat: this is a **global-context / leaky OOF** anchor. Validation
  features were built with `KaggleTopContext` containing validation wells, and
  the old pipeline predicted test from averaged fold models rather than final
  full-context models trained on all rows.
- Takeaway: the DWT-style feature table is in the right family, close to the
  public DWT notebook's reported single-model OOF range, but this number is not
  an honest CV target. Use it only as a pre-fix reference.
- Next: rerun clean schema v5 fold-safe full training after clearing
  `artifacts/feature_cache` and `artifacts/stack`.

### Fold-Safe Validation + Run Registry

- Reworked the main trainer so OOF features are built fold-by-fold with
  `KaggleTopContext` constructed only from train-fold wells.
- Final artifacts now train separate full-context final models on all train
  wells after OOF weights/postprocess are selected.
- `FEATURE_CACHE_SCHEMA_VERSION` bumped to 5; feature cache keys now include the
  `KaggleTopContext` key, so fold-safe and full-context features cannot collide.
- Added run diagnostics and `artifacts/runs.csv` registry with global RMSE,
  per-well RMSE, P90/worst well RMSE, typewell/no-typewell slices, hidden-length
  slices, selected postprocess, runtime, notes, and optional public LB.
- Public config additions:
  - `validation.fold_safe_context: true`;
  - `validation.final_model_strategy: full_context`;
  - `runs.registry_path: artifacts/runs.csv`.
- Validation:
  - `uv run pytest -q`: 13 passed;
  - `make check`: passed;
  - `uv run python -m compileall rogii`: passed;
  - `make quick-train`: 14,151 rows, 384 features, fold-safe OOF+PP RMSE
    10.04130;
  - quick artifact inference parity: same ids, no NaN, max_abs_diff 0.0.
- Full local result: pending; clean caches/artifacts before the next full run.

### Model Bundle Skill

- Expanded the local `kaggle-research-brief` skill into a model-ready bundle
  builder.
- Added `COMPETITION.md` as the canonical local competition description.
- New `make model-bundle` target writes `.kaggle_mining/model_bundle.md` with:
  - competition description;
  - `CHANGELOG.md`;
  - mined public code ideas;
  - mined discussion ideas;
  - repo git state, configs, metrics, package map, and solution code snapshot.
- `make research-db` now refreshes mining inputs and builds the full model
  bundle.
- Result: pending first generated bundle after the current mining DB is present.

### DWT-Repro Data Baseline

- Implemented the public DWT-notebook data approach as the main `stack.yml`
  baseline without depending on `ravaghi/wellbore-geology-prediction-artifacts`.
- Feature table now exposes DWT-style columns:
  - `TVT - last_known_tvt` residual target;
  - hidden-zone `frac`, geometry deltas, GR rolls/lags/diffs;
  - `pf_ancc`, `pf_z`, beam deltas, multi-scale NCC, multi-radius DTW,
    stochastic DTW uncertainty, DWT-lowpass DTW;
  - formation-plane KNN, dense ANCC calibration, segment biases;
  - offset GR residuals `tda*`, `tdbc*`, `tdsc*`, `tdpf*`, `tddtw*`.
- `configs/stack.yml` now mirrors the DWT notebook model set: 3 LightGBM + 3
  CatBoost CPU-safe variants; XGBoost removed from the main stack for a clean
  baseline comparison.
- Postprocess grid narrowed around the public notebook optimum:
  `alpha=0.95..1.00`, `tau=5..120`, `w_pf=0.00..0.15`, PF column `pf_ancc`.
- `FEATURE_CACHE_SCHEMA_VERSION` bumped to 4; old feature caches and old
  `artifacts/stack` are invalid.
- Validation:
  - `uv run pytest -q`: 8 passed;
  - `make check`: passed;
  - `uv run python -m compileall rogii`: passed;
  - `make quick-train`: 14,151 rows, 384 features, OOF+PP RMSE 9.68917;
  - quick artifact inference parity: same ids, no NaN, max_abs_diff 0.0.
- Full local result: pending; next command is `make train-local`.

### No-typewell NaN Fix (schema v3)

- **Root cause found**: `build_kaggle_top_signal_features` returned early for
  no-typewell wells with only the base features from `empty_top_signal_features`.
  Dynamic per-config features (`kg_beam_{tag}_tvt`, `kg_ncc_{w}_tvt`,
  `kg_dtw_r{r}_tvt`, `kg_dwt_r{r}_tvt`, `kg_form_{f}_tvt`, …) were entirely
  absent from the feature dict.  During training `pd.concat` filled those absent
  columns with NaN, so tree models learn a "missing" branch.  At test time
  `submission.py` was overwriting that NaN with `0.0`, sending rows down the
  wrong tree branch — clearly OOD for TVT-scale features (typical values ~1000 m).
- **Fix**: Pre-initialize all dynamic config-driven features with NaN at the
  start of `build_kaggle_top_signal_features` before any early return.  The NaN
  is now always present and consistent with the training distribution.
- `submission.py` now preserves the NaN from `reindex` (no longer writes 0.0);
  warning message updated to "Missing inference features kept as NaN".
- `FEATURE_CACHE_SCHEMA_VERSION` bumped to 3 — old caches invalidated.
- Test renamed to `test_predict_test_keeps_missing_features_as_nan`; asserts NaN.
- Local result: pending (needs `make train-local` after cache flush).
- Expected LB impact: positive; the previous 0.0 fill was an inference-only
  data leak relative to training.

## 2026-05-17

### Framework Simplification

- Removed legacy training modes and old compatibility configs.
- Kept only:
  - `configs/quick.yml` for smoke tests;
  - `configs/stack.yml` for the current baseline and submit artifact source.
- Main trainer is now a single grouped OOF ensemble:
  - LightGBM/CatBoost/XGBoost base models;
  - non-negative hill-climb blend on OOF predictions;
  - PF_ANCC delta postprocess and smoothing tuned on OOF;
  - one artifact format used by local and Kaggle inference.
- `make train-local` trains `configs/stack.yml`.
- `make quick-train` trains `configs/quick.yml`.
- `make submit` is an inference-only Kaggle run from `artifacts/stack`; final
  competition submit is manual from the Kaggle UI.

Takeaway: the framework is back to a simple baseline shape. New ideas should be
added to the main path only when they improve validation or leaderboard.

### Audit Fix Pass

- Fixed latent `TVT`-missing target-mask bug.
- Removed duplicate `idx_since` and `md_since` features; postprocess now uses
  `md_from_last_known`.
- Moved low-resolution DTW dynamic programming into numba.
- Added inference warnings for missing feature columns and preserved NaN so tree
  models use their trained missing-value branches.
- Expanded full-stack residual-weight grid to `[0.7, 0.8, 0.9, 1.0, 1.1]`.
- Added pytest coverage for target masks, feature schema, postprocess, DTW,
  XGBoost early stopping, and inference feature filling.
- Local result: pending.
- Kaggle result: pending.

### Makefile Command Cleanup

- Main commands are now:
  - `make train-local` for local training;
  - `make train-kaggle` is intentionally disabled because full training exceeds
    Kaggle's 9-hour CPU limit;
  - `make submit` for inference-only Kaggle run without competition submit.
- Removed old submit aliases from the primary workflow; Kaggle UI remains the
  place to press the final submit button.
- Training prints the resolved YAML config before building features.

### EXP-20260517-1 - Initial HGB Baseline

- Command/config: historical HGB baseline.
- Data: 773 train wells, 3,783,989 train rows.
- Local result:
  - 150-well CV RMSE: 16.63554;
  - flat baseline CV RMSE: 19.06126;
  - final train RMSE: 6.61516.
- Kaggle result: public LB 12.803.
- Runtime: local training around 6-7 minutes.
- Takeaway: useful first baseline, but sklearn boosting was not the right tool
  and validation was noisy.

### EXP-20260517-2 - First GBM Stack

- Command/config: historical stack config before simplification.
- Data: 773 train wells, 3,783,989 train rows.
- Local result:
  - all-well CV RMSE: 13.50285;
  - flat baseline RMSE: 17.50671;
  - final train RMSE: 4.64289;
  - features: 131.
- Kaggle result: public LB 13.033.
- Runtime: local training around 38 minutes.
- Takeaway: better local CV, worse public LB than HGB. The framework needed
  stronger alignment and cleaner artifact inference before more tuning.

### EXP-20260517-3 - Inference-Only Submit Path

- What changed:
  - local training produces reusable artifacts;
  - Kaggle kernel can run inference only from a private artifact dataset;
  - this avoids retraining all models inside the 9-hour Kaggle notebook budget.
- Takeaway: this is the right submit workflow while experiments are trained
  locally.

### EXP-20260517-4 - 9.251-Style Baseline Smoke

- Command/config: `make quick-train` / `configs/quick.yml`.
- Data: public sample, 3 train wells and 3 public test wells.
- Local result:
  - rows: 14,151;
  - features: 174;
  - baseline: `last_known_tvt`;
  - baseline RMSE: 11.53934;
  - OOF ensemble RMSE: 9.72345;
  - OOF + postprocess RMSE: 9.56255;
  - selected weight: `xgb_quick=1.0`;
  - best PF blend: `{alpha: 1.05, tau: 100.0, w_pf: 0.25}`;
  - best smoothing: `{window: 17, polyorder: 3}`.
- Kaggle result: not submitted.
- Takeaway: alignment features and OOF-tuned postprocess work on the smoke set.
  Needs full `make train-local` after simplification.
