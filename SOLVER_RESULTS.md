# Direct Solver Experiment Results

A self-contained record of the "direct test-time TVT path solver" line of
work and what we learned from it. This file is the source of truth that
``PROJECT_BRIEF.md`` summarizes; it deliberately keeps the raw numbers so we
do not have to re-run anything to reason about them.

## Context

This experiment was motivated by an initial (and as we later learned,
incorrect) assumption that the public LB is computed on `3` wells. Under that
assumption it made sense to build per-well bespoke solvers that try to "solve
each trajectory" rather than train a global model. The framework was then
extended into a multi-family solver pack with the idea that any one of those
families could be submitted standalone.

After we discovered that the real hidden test is **~200 wells** (see
``COMPETITION.md``), the bespoke-per-well rationale collapsed. The direct
solver is still useful as a feature-engineering source for the GBM stack
(schema16), but the standalone-submission framing is dead.

## Solver families implemented

All families share one energy function used to rank candidate paths:

```
energy(path) = w_gr  * GR-vs-typewell mismatch        (w_gr=1.00)
             + w_geo * |path - geo_path| / sigma      (w_geo=0.50)
             + w_slope * (slope - tail_slope)^2       (w_slope=0.20)
             + w_anchor * |path - anchor| / sigma     (w_anchor=0.00, off)
             + w_endpoint * |path_end - anchor_end|   (w_endpoint=0.00, off)
```

After the 2026-05-20 simplification both anchor-pull terms are zero. The
energy is anchor-free; the submission anchor is only used for the prediction
guard, not the search.

### Families and what each one solves

| Family             | Predicts                          | DoF/well | Where it lives                                       |
| ------------------ | --------------------------------- | -------- | ---------------------------------------------------- |
| `linear_tailfit`   | ridge regression on `MD, X, Y, Z` | ~5       | `rogii/direct_solver.py::fit_linear_candidate`       |
| `geo_tailfit`      | best single formation linear fit  | 3        | `rogii/direct_solver.py::fit_geo_candidate`          |
| `geo_consensus`    | inverse-RMSE weighted median over all 6 formations | 3 * 6 | same                                                 |
| `stage1_raw`       | global `tvt = last + a*dmd + b*dz` grid search     | 2     | `rogii/path_solver_extras.py::stage1_global_linear`  |
| `stage12_raw`      | Stage1 + 10 bounded knot offsets  | 2 + 10   | `rogii/path_solver_extras.py::stage2_local_refine`   |
| `cem_raw`          | Cross-entropy search of (offset, slope, curvature) around `geo_path` | 3 | `rogii/path_solver_extras.py::cem_path_search`       |
| `cem_top_median`   | row-wise median of CEM top-5 paths| —        | same                                                 |
| `crosswell_md_raw` | median of k=8 nearest train wells' TVT(MD-anchor) | 0   | `rogii/cross_well_prior.py::cross_well_typewell_path`|
| `crosswell_z_raw`  | same but aligned by Z             | 0        | same                                                 |
| `crosswell_median` | row-wise median of MD and Z variants | 0     | same                                                 |
| `cem_over_crosswell_*` | CEM corrections on top of cross-well base | 3 | `rogii/direct_solver.py::solve_well`               |

Removed during cleanup:
- `tie-point / landmark solver` — GR landmark monotonic assignment to typewell.
  Rarely produced enough good landmarks on real wells; deleted.
- `gr_safe / geo_safe / gr_bold / geo_bold` and their gated blends — every
  one of these pulled the answer toward the submission anchor (`10.084`),
  capping the search at the anchor. Deleted along with the `VARIANTS` tuple.
- Rust `rogii_solver_core` crate — affine grid kernel applied on top of the
  anchor; same anchor-pinning problem. Deleted with its `ctypes` wrapper.

## Train-eval results (773 wells)

Command:

```bash
make direct-solver-train-eval \
  DIRECT_SOLVER_OUTPUT=artifacts/direct_solver_tier1 \
  DIRECT_SOLVER_PROGRESS_INTERVAL=10 \
  DIRECT_SOLVER_PSEUDO_PUBLIC_TRIALS=200 \
  DIRECT_SOLVER_PSEUDO_PUBLIC_ANCHOR=stage12_raw \
  DIRECT_SOLVER_PSEUDO_PUBLIC_MATCHED=true \
  DIRECT_SOLVER_PSEUDO_PUBLIC_CANDIDATE_K=80 \
  DIRECT_SOLVER_CROSS_WELL=true \
  DIRECT_SOLVER_CROSS_WELL_K=8 \
  DIRECT_SOLVER_WORKERS=8 \
  DIRECT_SOLVER_ANCHOR=
```

- Wells: `773` train wells, hidden-row evaluation against ground truth `TVT`.
- Rows total: `3,783,989`.
- Backend: pure Python with `multiprocessing.Pool(8)`.
- Wall-clock: roughly `2 h`.

Artifacts: `artifacts/direct_solver_tier1/train_eval/`.

### Weighted RMSE ranking (the headline number)

`weighted_rmse` is the row-count-weighted RMSE across all hidden rows. It is
the closest proxy we have to public LB RMSE before submitting.

| variant                       | weighted_rmse | mean_well_rmse | median_well_rmse | p90_well_rmse |
| ----------------------------- | ------------- | -------------- | ---------------- | ------------- |
| `geo_consensus`               | `0.97`        | `0.21`         | `0.06`           | `0.34`        |
| `geo_tailfit`                 | `0.98`        | `0.21`         | `0.06`           | `0.33`        |
| `cem_top_median`              | `4.00`        | `2.22`         | `0.80`           | `5.58`        |
| `cem_raw`                     | `4.18`        | `2.36`         | `0.76`           | `5.83`        |
| `stage12_raw`                 | `16.00`       | `12.11`        | `9.22`           | `23.80`       |
| `stage1_raw`                  | `16.08`       | `12.23`        | `9.29`           | `23.91`       |
| `crosswell_md_raw`            | `16.09`       | `13.30`        | `11.13`          | `23.28`       |
| `cem_over_crosswell_raw`      | `25.40`       | `17.00`        | `10.69`          | `38.72`       |
| `cem_over_crosswell_top_median`| `25.46`      | `16.97`        | `10.70`          | `38.56`       |
| `crosswell_median`            | `31.19`       | `22.01`        | `15.07`          | `48.73`       |
| `crosswell_z_raw`             | `57.71`       | `36.31`        | `18.49`          | `93.74`       |
| `linear_tailfit`              | `702.77`      | `384.58`       | `180.00`         | `1041.47`     |

### Matched-triple pseudo-public ranking

`200` triples of 3 train wells each, drawn from the nearest `80` train wells
to each of the visible test wells (signature-matched). This was useful as a
sanity check on relative ranks; it is not a reliable LB simulator because the
real public LB is over ~200 wells, not 3.

| variant                       | median_triple_rmse | win_rate_vs_stage12 | trials |
| ----------------------------- | ------------------ | ------------------- | ------ |
| `geo_tailfit`                 | `0.12`             | `1.000`             | 200    |
| `geo_consensus`               | `0.13`             | `1.000`             | 200    |
| `cem_top_median`              | `1.98`             | `0.975`             | 200    |
| `cem_raw`                     | `2.21`             | `0.975`             | 200    |
| `stage12_raw` (anchor)        | `10.57`            | `0.000`             | 200    |
| `stage1_raw`                  | `10.61`            | `0.425`             | 200    |
| `crosswell_md_raw`            | `12.32`            | `0.395`             | 200    |
| `cem_over_crosswell_top_median`| `15.40`           | `0.285`             | 200    |
| `cem_over_crosswell_raw`      | `15.82`            | `0.290`             | 200    |
| `crosswell_median`            | `21.04`            | `0.150`             | 200    |
| `crosswell_z_raw`             | `37.35`            | `0.085`             | 200    |
| `linear_tailfit`              | `418.62`           | `0.000`             | 200    |

## What this tells us, plainly

### 1. `geo_consensus 0.97 ft` is in-sample

`fit_geo_candidate` does ridge regression on the known tail (`TVT_input`) of
the same well it is evaluated on. The hidden rows are out-of-fit only in the
sense that `TVT_input` is NaN there; the well-specific coefficients
`(a_S, b_S, c_S)` per formation are estimated from the head of the same well.

So `geo_consensus 0.97 ft` is "how well do per-well linear models in (Z - S)
extrapolate inside the same well". It is not an LB simulation. The real LB
will inflate this number through:

- **Train-only formation columns missing on test wells.** This alone
  collapses geo_consensus on test to whatever falls out of the
  `fit_linear_candidate` fallback when no formation columns exist.
- **Spatial geological drift** between known tail and hidden interval.
- **Typewell-anchored uncertainty** when typewell GR shape disagrees with
  horizontal GR.

The honest expectation for `geo_consensus` submitted as-is, after a fix that
imputes formations on test wells, is **somewhere around `~10 ft` LB**, i.e.
similar to the current `schema10` baseline at `10.084`. Not the breakthrough
we hoped for.

### 2. `cem_raw 4.18 ft` is similarly in-sample-flattering

CEM searches an offset/slope/curvature correction on top of `geo_path`. Since
`geo_path` is near-truth on train, even modest corrections look great. On
test, `geo_path` will be a much weaker base (or absent) so the correction is
applied to a more uncertain anchor and the result inflates.

### 3. Stage1/Stage2 at `~16 ft` is honest but worse than baseline

These families do not use formation surfaces directly. Their `~16 ft` train
weighted RMSE is the floor for "what can we do with only MD/Z and a tail
slope". This is `~6 ft` worse than the GBM baseline (`10.084` LB), which
already uses typewell and a richer feature stack.

### 4. Cross-well prior is noise, not signal, on the metric

`crosswell_*` variants land at `16-58 ft` weighted RMSE with a `+9.6 ft`
systematic positive shift versus the `stage12_raw` anchor in matched
triples. Median over 8 nearest train wells' TVT trajectories at the same
relative MD or Z does not match the local well well enough to be useful.

We initially hypothesized "cross-well is bad on train because every train
well has a strong own typewell, but will be good on test where typewells are
weak". That hypothesis is rejected by two observations:

- The shift is **systematic and one-sided**, not high-variance: cross-well
  is biased, not noisy.
- The same shift appears even on signature-matched neighbors. Better
  neighbor selection does not fix the bias.

`cross_well_prior` therefore should not become a GBM feature in schema16+;
the bias would teach the model the wrong relationship in train and that
relationship would not improve on test either.

### 5. `cem_over_crosswell` is the worst of both worlds

CEM applies a bounded correction (`|offset| <= 24 ft`, `|slope_offset| <= 24
ft`, `|curvature| <= 16 ft`) on top of `crosswell_median`. When the base is
biased by ~10 ft, CEM cannot pull back to the correct path within those
bounds. The combination ranks below `crosswell_md_raw` alone.

## Why no single direct solver variant should be a submission

Stacking up the conclusions:

- `geo_consensus / geo_tailfit / cem_*` are in-sample-flattering. Their
  numbers do not transfer to ~200-well hidden test where formations are
  unavailable as columns and have to be imputed.
- `stage1 / stage12` are honest but worse than the GBM baseline.
- `crosswell_*` and `cem_over_crosswell_*` are biased or noisy.

The right framing is: **direct solver outputs are features for the GBM
stack**, not standalone submissions. Schema16 (`rogii/path_features.py`)
does exactly that, modulo the train-only-columns issue that still needs to
be solved by imputing formations before computing path features at test
time (or by a teacher-student approach that skips surfaces entirely; see
``PROJECT_BRIEF.md`` for the alternatives).

## Code locations

- `rogii/direct_solver.py` — solver wiring, `solve_well`,
  `fit_linear_candidate`, `fit_geo_candidate`, `fit_gr_calibration`,
  `score_candidate_path`, `_ENERGY_VARIANT`.
- `rogii/path_solver_extras.py` — `cem_path_search`, `stage12_path`,
  matched-triple harness, signature builder.
- `rogii/cross_well_prior.py` — train-paths loader, signature
  standardization, `cross_well_typewell_path`. Kept for reference; not
  recommended for inclusion in future schemas.
- `rogii/path_features.py` — schema16 feature wrapper exporting per-row
  TVT predictions + per-well scalars to the GBM training stack.
- `rogii/prediction_guard.py` — strict/bold-mode shift guard against an
  anchor submission. Submit-time safety, not part of the energy.
- `configs/direct_solver_policy.yml` — submit-discipline policy, bold
  family list, anti-overfit constraints.
- `tests/test_direct_solver.py`,
  `tests/test_path_solver_extras.py`,
  `tests/test_cross_well_prior.py`,
  `tests/test_path_features.py` — unit coverage for the solver families
  and the schema16 feature wrapper.

## Recommendations from this experiment for any redesign

1. **Drop cross-well prior entirely.** Both submit and feature use are
   rejected by data.
2. **Drop tie-point and rust path families.** Already removed.
3. **Keep `geo_consensus` (or its successor) as a known-signal indicator.**
   But solve the train-only-columns problem before relying on it for test.
4. **Keep `cem_raw` and `stage12_raw` paths as features**, with the energy
   parameters in `configs/stack.yml::features.direct_path` exposed for
   ablation.
5. **The right next step** is not another direct solver family but a
   teacher-student that learns to predict `geo_consensus_delta` from
   test-available features only. See ``PROJECT_BRIEF.md`` for details.
