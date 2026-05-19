# ROGII Profiling Notes

This document is the performance baseline for the ROGII framework. Update it
when a change is meant to make training or inference faster.

## Commands

Use `RunLogger` timings for every serious run. Use `cProfile` when we want a
function-level view.

```bash
make profile-quick
make profile-train PROFILE_NAME=stack_YYYYMMDD
make profile-features PROFILE_NAME=features_stack50 FEATURE_PROFILE_WELLS=50
make profile-report PROFILE_NAME=quick
```

Generated files live under `artifacts/profiles/`:

- `*.log`: normal ROGII logs with stage durations;
- `*.prof`: raw `cProfile` output;
- `*.md`: Markdown report with stage timings and top cumulative/self-time
  functions.

For feature-prep-only profiling, use `make profile-features`. By default it
profiles the first fold's train table (`FEATURE_PROFILE_CONTEXT=fold-train`),
uses `FEATURE_PROFILE_WELLS=50`, and disables feature cache so the report
captures cold compute instead of cache reads. Useful variants:

```bash
make profile-features PROFILE_NAME=features_fold1_25 FEATURE_PROFILE_WELLS=25
make profile-features PROFILE_NAME=features_valid_50 FEATURE_PROFILE_CONTEXT=fold-valid FEATURE_PROFILE_WELLS=50
make profile-features PROFILE_NAME=features_cached_50 FEATURE_PROFILE_DISABLE_CACHE=false FEATURE_PROFILE_WELLS=50
```

Feature prep now has two cache layers:

- `context_free`: geometry, GR rolls/lags, typewell, beam/NCC/DTW/DWT, PF, and
  offset residuals. This layer does not include `KaggleTopContext.context_key`,
  so fold-safe contexts can reuse it.
- `context`: spatial formation, dense ANCC, and aggregate signals that depend
  on the current fold/full context. This layer includes `context_key`.

`features.num_workers` enables process-based per-well parallel feature building.
`configs/stack.yml` stays serial by default for local debugging; `configs/stack_gpu.yml`
sets `features.num_workers: 8` for server runs. Stage-level profiling forces the
serial path so per-stage logs remain readable. To measure worker throughput:

```bash
make profile-features CONFIG=configs/stack_gpu.yml FEATURE_PROFILE_WELLS=50 FEATURE_PROFILE_STAGE=false
```

When `feature_profile` runs, it enables per-stage feature logs for
`well.read_horizontal`, `well.typewell`, `top.beam`, `top.ncc`, `top.dtw`,
`top.dwt`, `top.spatial.formations`, `top.spatial.dense`, `top.spatial`,
`top.particle`, `top.offsets`, `well.top_signals_total`, and
`well.materialize_frame`.

For deltas, compare runs with the same config, cache state, machine, and Python
environment. If any of those change, treat the comparison as directional only.

## Current Full-Run Anchor

Run: `run_20260518_103914_global_context_stack_v4`

Config: historical `configs/stack.yml` before fold-safe OOF context.

Result: CV-like OOF+PP RMSE `10.51127`, public LB `9.946`.

Important caveat: this was the pre-fold-safe/global-context run. It is the
current public-LB anchor, not an honest validation anchor.

| stage | duration | share |
| --- | ---: | ---: |
| Build Kaggle top context | 00:04 | 0.0% |
| Build training table | 44:50 | 10.5% |
| Train OOF ensemble | 4:38:50 | 65.2% |
| Tune OOF postprocess | 1:43:57 | 24.3% |
| Predict test | 00:13 | 0.1% |
| Total | 7:07:58 | 100.0% |

OOF model training split:

| model | 5-fold duration | note |
| --- | ---: | --- |
| `cat_lr025` | 1:02:56 | best single model family |
| `cat_lr020` | 48:27 | nonzero blend weight |
| `cat_lr030` | 1:00:26 | nonzero blend weight |
| `lgb_lr025` | 35:39 | nonzero blend weight, diversity |
| `lgb_lr020` | 34:31 | zero final blend weight in this run |
| `lgb_lr030` | 36:31 | zero final blend weight in this run |

Immediate implication: for the old global-context path, the biggest safe win was
postprocess tuning. It consumed almost two hours for about `0.03` OOF RMSE.

## Quick Profile Baseline

Command:

```bash
make profile-quick
```

Artifacts:

- `artifacts/profiles/quick.prof`
- `artifacts/profiles/quick.log`
- `artifacts/profiles/quick.md`

RunLogger total after postprocess basis scoring and dense-spatial vectorization:
`15.95s`.

cProfile total after postprocess basis scoring and dense-spatial vectorization:
`17.644s`, `14.7M` calls.

Top stage timings:

| stage | duration | share |
| --- | ---: | ---: |
| Train fold-safe OOF ensemble | 11.50s | 72.1% |
| Build final training table | 1.45s | 9.1% |
| Predict test | 1.41s | 8.8% |
| Train final full-context models | 1.36s | 8.5% |
| Tune OOF postprocess | 0.02s | 0.1% |

Top cumulative functions from `artifacts/profiles/quick.md`:

| function | cumtime | why it matters |
| --- | ---: | --- |
| `build_well_features` | 7.638s | feature generation envelope |
| `modeling.WrappedRegressor.fit` | 7.244s | model training envelope |
| `build_kaggle_top_signal_features` | 7.179s | DWT/DTW/beam/PF/spatial signal block |
| `spatial.KaggleTopContext.impute_formations` | 4.650s | exact fallback on quick because `spatial_k=2` |
| `spatial.KaggleTopContext.impute_dense_ancc` | 0.264s | dense ANCC vectorized |

Top self-time functions:

| function | selftime | note |
| --- | ---: | --- |
| `lightgbm.basic.update` | 2.641s | native training time |
| `catboost._train` | 1.834s | native training time |
| `_impute_formations_row_loop` | 1.751s | exact fallback for underdetermined formation planes |
| `xgboost.core.update` | 0.824s | native training time |
| `run_pf_ancc_signal` | 0.437s | particle filter signal |

## What To Optimize First

1. **Postprocess tuning**
   - Full-run cost: `1:43:57`.
   - Current OOF gain in anchor run: `10.54106 -> 10.51127`.
   - Implemented exact basis scoring on 2026-05-18: the grid now scores
     candidates from precomputed dot-products instead of rebuilding the full
     prediction vector for every candidate.
   - Synthetic benchmark, 50k rows, 4,608 candidates:
     - brute force: `53.015s`;
     - exact basis scorer: `0.863s`;
     - speedup: `61.5x`;
     - best RMSE delta: `1.8e-9`.
   - Full-run delta is pending; this should remove most of the old 1h44m
     postprocess bottleneck without changing the selected candidate.

2. **Spatial feature generation**
   - Implemented vectorized dense ANCC scoring on 2026-05-18:
     `impute_dense_ancc` quick profile dropped from `1.880s` to `0.264s`.
   - Implemented vectorized formation plane scoring for non-degenerate contexts.
     The quick profile keeps the exact row-loop fallback because `quick.yml`
     uses `spatial_k=2`, which makes plane fitting underdetermined.
   - Full `stack.yml` uses `spatial_k=10`, so the vectorized formation path
     should apply there; exact full-run delta is pending.
   - Implemented exact dense ANCC fetch sizing on 2026-05-19. The old code
     queried `dense_fetch=5000` neighbors per row and then discarded same-well
     points. The new code queries `dense_k + self_well_dense_points + 8`, which
     is enough to preserve the same top non-self candidates while avoiding the
     huge KD-tree result matrix.
   - Stack feature-prep micro-profile, fold 1 train context, 3 wells, cache off:
     - before: `Profile feature table = 9.67s`,
       `impute_dense_ancc = 8.274s`;
     - after: `Profile feature table = 1.38s`,
       `impute_dense_ancc = 0.084s`;
     - table-build speedup: `7.0x`;
     - dense ANCC speedup: `98.5x`.
  - Stack feature-prep profile, fold 1 train context, 25 wells, cache off:
     `Profile feature table = 10.69s`, `117,140` rows, `415` features.
     The matching server symptom before this fix was `25/618` wells in `03:18`,
     so rerun server logs should show whether the remote bottleneck was the
     same dense-neighbor fetch path.
   - After split-cache and 8 worker processes, the same 25-well profile with
     `CONFIG=configs/stack_gpu.yml FEATURE_PROFILE_STAGE=false` built `117,140`
     rows / `415` features in `4.59s` (`25,529 rows/sec`). This is a `2.3x`
     table-build speedup over the post-dense-optimization serial profile, and
     about `43x` faster than the original server symptom.
   - New spatial stage logs on the same after-profile:
     `top.spatial.formations = 0.014-0.017s/well`,
     `top.spatial.dense = 0.024-0.033s/well`,
     `top.particle = 0.220-0.308s/well`.
   - Next idea: split context-independent alignment features from fold-specific
     spatial features so fold-safe OOF does not recompute everything.

3. **Fold-safe feature cache strategy**
   - Fold-safe OOF intentionally rebuilds features under different context keys.
   - The expensive design question: cache/reuse context-independent features once,
     then append fold-specific spatial/context features.
   - `features.progress_interval` controls feature-table heartbeat logs. Full
     `stack.yml` logs every 25 wells so cold server runs show progress/ETA
     before the old 100-well checkpoint.

4. **Model training**
   - CatBoost dominates the old full OOF model time, but it also gives the best
     single-model OOF.
   - Do not trim this first unless the comparison run is explicitly an ablation.

5. **Inference**
   - Current visible test inference is small; hidden test has about 200 wells.
   - Profile hidden-like inference only after the training path is stable.

## Delta Log Template

Use this when changing performance-sensitive code:

```text
### YYYY-MM-DD - change name

- Config:
- Cache state:
- Command:
- Before:
- After:
- Speed delta:
- Metric delta:
- Keep/revert:
- Notes:
```

## Rules

- Do not optimize away quality signals before one clean fold-safe full run.
- Prefer reducing repeated work over reducing model count.
- Treat public LB as the final check, but use profile deltas to decide where to
  spend engineering time.
- Keep raw profiles in `artifacts/profiles/`; summarize only stable conclusions
  in this document.
