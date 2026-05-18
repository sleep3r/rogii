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
- Added inference warnings for missing feature columns and zero-filled them.
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
