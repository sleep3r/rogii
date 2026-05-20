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

## 2026-05-20

### Direct Solver: Anchor-Free Simplification

The LB landscape (top is `8.239`, ours is `10.084`) made it clear that the
24-variant safe/bold/consensus/Rust zoo we just shipped is the wrong shape.
Every safety mechanism — `gated_blend`, `anchor_weight` in the energy
function, `consensus_safe/bold/bold_family` blends, the self-anchoring
fallback when no submission anchor is provided, the Rust correction kernel
applied on top of `anchor_path` — was pulling each candidate back toward a
`10.084` anchor. The next radical move could not coexist with that gravity.

This change strips the defenses that pulled the search toward the anchor
and leaves a small, focused solver pack.

- Removed:
  - `gated_blend` helper. Solver paths are not partially blended with the
    submission anchor anymore. Raw candidates are emitted as-is and clipped
    by the per-step physics limiter only.
  - `VARIANTS` tuple (`gr_safe`, `geo_safe`, `gr_bold`, `geo_bold`) and the
    big `for variant in VARIANTS:` loop in `solve_well`.
  - `affine_search_grid_values`, `build_grid_paths`: deterministic affine
    grid that produced `lin_s*_e*` candidates. CEM with 320 candidates per
    iteration over `(offset, slope_offset, curvature)` is strictly more
    expressive than this fixed grid.
  - `RUST_CORRECTION_VARIANTS` and the Rust kernel integration
    (`rust_affine_candidate`, `rust_correction_candidates`). The Rust kernel
    `correction_grid_search` corrected on top of `anchor_path` — i.e. it
    was an anchor-pinned defense by construction. Removed the crate entirely
    (`rust/`), the ctypes wrapper (`rogii/solver_core.py`), the
    `solver-core-build` / `solver-core-test` Make targets and the
    `--backend rust` CLI flag. Keep a placeholder in mind for re-introducing
    a Rust energy-scoring batch later once we know the energy is correct.
  - Tie-point / landmark solver (`detect_landmarks`,
    `_monotonic_landmark_assignment`, `_interpolate_path`,
    `fit_tiepoint_path`) and all its tests. It rarely produced enough
    monotonic GR-typewell matches to move the path on real wells and added
    220 lines of code.
  - `consensus_safe`, `consensus_bold`, `consensus_bold_family` outputs.
    These were all anchor-medianed and pulled solutions back to the same
    anchor.
  - `*_anchor_blend40/60` proliferation for every variant. Only kept the
    two blends that are useful as an explicit safety net:
    `cem_top_median_anchor_blend{40,60}` and
    `stage12_raw_anchor_blend{40,60}`. These are emitted only when a
    submission anchor is actually provided.
  - The `(linear + geo) / 2` self-anchor fallback that fired whenever no
    submission anchor was supplied. There is no fabricated anchor anymore;
    if you do not pass `--anchor-submission`, no anchor diagnostics or
    blends are produced.

- Changed:
  - `solve_well` is anchor-free by design. The energy function uses
    `anchor_weight = 0` and `endpoint_weight = 0`; only GR/typewell match,
    geological distance and slope priors drive the search.
  - CEM applies its `(offset + slope_offset * centered + curvature * shape)`
    correction on top of `geo_path` (the geological tailfit), not on top of
    the submission anchor. The principled prior is the geology, not a
    known-suboptimal submission.
  - `solve_well` now produces six raw variants instead of twenty-four:
    `linear_tailfit`, `geo_tailfit`, `stage1_raw`, `stage12_raw`,
    `cem_raw`, `cem_top_median`.
  - Default pseudo-public anchor variant flipped from `consensus_safe` to
    `stage12_raw`. Default `DIRECT_SOLVER_PSEUDO_PUBLIC_MATCHED=true`:
    matched-triple harness is now the default, not an opt-in.
  - `configs/direct_solver_policy.yml` rewritten to reflect the new
    energy_weights, base_path, raw variant list and bold-only submit
    discipline.

- Validation:
  - `uv run python -m compileall rogii`: passed;
  - `uv run ruff check rogii tests`: passed;
  - `uv run pytest -q`: `57 passed` (was `60` before; the 3 tiepoint tests
    are gone, the rust-backend test is gone, replaced with two new tests
    that check the simplified variant surface and the anchor-blend gating);
  - end-to-end smoke (3 train wells, matched-triple harness with 4
    trials):
    - `python -m rogii.direct_solver --data-dir data --train-eval
      --max-wells 3 --progress-interval 1 --sample-seed 11
      --pseudo-public-trials 4 --pseudo-public-anchor-variant stage12_raw
      --pseudo-public-matched --pseudo-public-candidate-k 5`;
    - `01:16` runtime, `6` submission files, no NaN warnings;
    - train hidden-row RMSE on this slice (smaller is better):
      `geo_tailfit 0.24`, `cem_raw 2.57`, `cem_top_median 4.32`,
      `stage1_raw 8.00`, `stage12_raw 8.38`, `linear_tailfit 388.38`.
    - Note: `geo_tailfit` looks too good here because train wells expose
      the geology that we used to fit it. Real public LB will reward `cem_raw`
      and `stage12_raw` more, which is the whole point of the simplification.

- Code statistics:
  - `direct_solver.py`: `1457` → `1085` lines (-25%);
  - `path_solver_extras.py`: `787` → `567` lines (-28%);
  - removed the `rust/` crate (-616 lines of Rust);
  - removed `rogii/solver_core.py` (-295 lines of ctypes glue);
  - net delta: roughly `-1500` lines of code with no loss of relevant
    functionality.

- Why this is the right move now:
  - The leaderboard gap is ~`1.8 RMSE`. Defenses that pull our output
    toward a `10.084` anchor cap us at roughly `10.084` regardless of how
    good the solver is.
  - The matched-triple harness is the relevant evaluation, not gated
    train-eval rankings.
  - With fewer variants and an honest energy, each submit is a real
    statement, not a partial nudge.
  - Reintroducing complexity is cheap once we know it earns LB. Removing
    accumulated complexity later is much harder.

### Direct Solver Plan Completion: Tie-Point, CEM, Stage1/Stage2, Matched Triples, Bold Guard

- Context:
  - Day-1 chunk shipped the bold affine/correction solver (plan item 1) and
    the random pseudo-public harness (plan item 0). Plan items 2/3/5, the
    matched-triple sampling, and the relaxed prediction guard were still open;
  - MTP CNN (plan item 4) intentionally remains out of scope: a 6-12 h training
    prototype does not fit the same submit-discipline window as the other
    direct-solver families.
- What changed:
  - added `rogii/path_solver_extras.py` with:
    - `detect_landmarks` + `_monotonic_landmark_assignment` (plan item 2,
      tie-point / landmark solver);
    - `fit_tiepoint_path` interpolating monotonic GR landmarks against typewell
      TVT;
    - `cem_path_search` (plan item 3, cross-entropy iterative search over
      `(offset, slope_offset, curvature)` with elite refit, top-k median path,
      clipped `|param| <= 24/24/16 ft`);
    - `stage1_global_linear` + `stage2_local_refine` + `stage12_path`
      (plan item 5, global `tvt = last_tvt + a*dmd + b*dz` grid then
      knot-by-knot bounded refinement, max `+/-12 ft`, smoothness penalty
      `0.05 * sum(diff(offsets)^2)`);
    - `compute_well_signature`, `signatures_to_frame`,
      `collect_well_signatures`, `matched_triples` for signature-matched
      train-triples in the pseudo-public harness.
  - wired all new families through `direct_solver.solve_well`, sharing
    `EnergyContext` + `score_candidate_path` so they are ranked on the same
    loss as the affine grid;
  - new output variants:
    `tiepoint_raw`, `tiepoint_blend40`, `tiepoint_blend60`,
    `cem_raw`, `cem_top_median`, `cem_blend40`, `cem_blend60`,
    `stage1_raw`, `stage12_raw`, `stage12_blend40`, `stage12_blend60`,
    `consensus_bold_family` (median of CEM top-k, tie-point, stage12, anchor);
  - relaxed prediction guard:
    - new `--mode {strict,bold}` flag (default `strict` keeps EXP-20260520-8
      semantics);
    - bold mode raises hard-fail thresholds to `p95=35 ft` and
      `median_abs=20 ft` (still blocks HMM-level shifts at >25/12 ft +
      the existing tail-continuity / one-sided checks) and adds an explicit
      warn band at `median_abs>5 ft` / `p95>12 ft`;
    - new `--warn-well-median-abs-shift` / `--warn-well-p95-shift` flags
      surface the warn band as a status (not a fail) for honest 5-12 ft
      per-well shifts.
  - new Makefile targets and vars:
    - `PREDICTION_GUARD_MODE` (defaults `strict`, used by `make prediction-guard`);
    - `make direct-solver-guard-bold DIRECT_SOLVER_VARIANT=...` runs the bold
      preset and writes a separate `_bold.{md,json}` report;
    - `DIRECT_SOLVER_PSEUDO_PUBLIC_MATCHED=true` plus
      `DIRECT_SOLVER_PSEUDO_PUBLIC_CANDIDATE_K=80` route train-eval to use
      signature-matched test-like triples instead of uniform-random ones.
  - new CLI args on `python -m rogii.direct_solver`:
    `--pseudo-public-matched`, `--pseudo-public-candidate-k`;
  - bundle now ships `rogii/path_solver_extras.py`;
  - updated `configs/direct_solver_policy.yml` with the `bold_families`
    block and `bold_guard_required_for_bold_families` rule so future runs
    cannot silently submit a bold family without the bold guard report.
- Anti-overfit discipline (carried over from the prior pack and reaffirmed):
  - submit budget stays at `<=2` public submits for the safe + bold pair;
  - bold families (`tiepoint_raw`, `cem_raw`, `cem_top_median`,
    `stage12_raw`, `consensus_bold_family`) must always be checked with the
    bold guard; the bold guard hard-fails at HMM-level shifts but lets honest
    5-12 ft per-well shifts pass as a `warn`;
  - matched-triple harness is required before submitting any bold family.
- Validation:
  - `uv run python -m compileall rogii`: passed;
  - `uv run ruff check rogii tests`: passed;
  - `uv run pytest -q`: `60 passed` (was `48` before this pack);
  - `make solver-core-test`: passed (1 Rust unit test, unchanged);
  - end-to-end smoke:
    `python -m rogii.direct_solver --data-dir data --train-eval
    --max-wells 5 --progress-interval 1 --sample-seed 11
    --pseudo-public-trials 6 --pseudo-public-triple-size 3
    --pseudo-public-anchor-variant consensus_safe
    --pseudo-public-matched --pseudo-public-candidate-k 8`
    completed in `02:55`, generated `24` submission variants (was `12`),
    wrote signature CSVs and matched-triple pseudo-public summary;
  - bold-mode guard smoke:
    `python -m rogii.prediction_guard --mode bold ...`: PASS on a
    self-compared submission, no crash, separate report path.
- Tests added (`tests/test_path_solver_extras.py`, `10 passed`):
  - `test_detect_landmarks_finds_extrema`;
  - `test_monotonic_landmark_assignment_orders_pairs`;
  - `test_fit_tiepoint_path_uses_typewell`;
  - `test_cem_path_search_returns_finite_paths`;
  - `test_stage1_global_linear_picks_finite_pair`;
  - `test_stage2_local_refine_respects_max_offset`;
  - `test_stage12_path_chains_stage1_and_stage2`;
  - `test_compute_well_signature_returns_expected_keys`;
  - `test_matched_triples_picks_neighbors`;
  - `test_collect_well_signatures_train`.
- Plan coverage after this pack:
  - item 0 (pseudo-public harness): DONE, now with optional matched triples;
  - item 1 (bold affine/correction): DONE (Rust correction kernel);
  - item 2 (tie-point / landmark): DONE in Python;
  - item 3 (CEM iterative refit): DONE in Python (Rust scoring stays via the
    existing `path_energy` kernel for the affine grid path; CEM is fast
    enough in Python because the candidate path build is vectorized);
  - item 4 (MTP CNN candidate generator): NOT STARTED, deliberately deferred
    until a safer two-day window is available;
  - item 5 (Stage1 / Stage2 reproduction): DONE.
- Submit discipline reminder:
  - first safe submit candidate stays
    `submission_direct_consensus_safe.csv` from the test run;
  - second submit, when matched-triple harness ranks a bold family above
    the consensus anchor and bold-guard passes, should be one of
    `submission_direct_cem_top_median.csv`,
    `submission_direct_stage12_raw.csv`,
    `submission_direct_tiepoint_raw.csv`,
    or `submission_direct_consensus_bold_family.csv`. Never tune weights
    after seeing public LB.

### Direct Solver Rust Core

- Context:
  - direct inversion / local-search solvers are the next plausible route for
    large LB movement;
  - Python is fine for CSV/reporting, but CEM/stage2 candidate scoring should
    run in a tight compiled loop.
- What changed:
  - added Rust crate `rust/rogii_solver_core`;
  - added `rogii.solver_core` ctypes wrapper;
  - added explicit `--backend rust` to `rogii.direct_solver`;
  - added Make targets:
    - `make solver-core-build`;
    - `make solver-core-test`;
  - no silent fallback: `--backend rust` requires the Rust dylib/so to exist.
- Current Rust kernel:
  - affine TVT path grid search over slope and endpoint correction;
  - scores GR/typewell match, geo path distance, anchor distance, slope
    penalty, endpoint penalty;
  - returns best path, score, slope, endpoint correction.
- Added Rust correction kernel:
  - searches `anchor + offset + slope_shape + curvature_shape`;
  - emits candidates:
    - `rust_corr_bal_raw`, `rust_corr_bal_blend40`, `rust_corr_bal_blend60`;
    - `rust_corr_gr_raw`, `rust_corr_gr_blend40`, `rust_corr_gr_blend60`;
    - `rust_corr_geo_raw`, `rust_corr_geo_blend40`, `rust_corr_geo_blend60`.
- Validation:
  - `make solver-core-test`: passed;
  - `make solver-core-build`: passed;
  - `uv run pytest -q tests/test_direct_solver.py`: `2 passed`;
  - `uv run pytest -q`: `48 passed`;
  - `make check`: passed;
  - `uv run python -m compileall rogii`: passed.
- Smoke:
  - command:
    `make direct-solver-train-eval DIRECT_SOLVER_OUTPUT=artifacts/direct_solver_rust_smoke
    DIRECT_SOLVER_MAX_WELLS=1 DIRECT_SOLVER_PROGRESS_INTERVAL=1
    DIRECT_SOLVER_BACKEND=rust DIRECT_SOLVER_ANCHOR=`;
  - runtime: `07.43s`;
  - rows: `3,836`;
  - generated `12` direct-solver variants.
- Correction smoke:
  - command:
    `make direct-solver-train-eval DIRECT_SOLVER_OUTPUT=artifacts/direct_solver_rust_corr_smoke
    DIRECT_SOLVER_MAX_WELLS=1 DIRECT_SOLVER_PROGRESS_INTERVAL=1
    DIRECT_SOLVER_BACKEND=rust DIRECT_SOLVER_ANCHOR=`;
  - runtime: `08.74s`;
  - rows: `3,836`;
  - generated `21` direct-solver variants.
- Test candidate generation:
  - command:
    `make direct-solver-test DIRECT_SOLVER_OUTPUT=artifacts/direct_solver_rust_corr
    DIRECT_SOLVER_PROGRESS_INTERVAL=1 DIRECT_SOLVER_BACKEND=rust`;
  - runtime: `33.56s`;
  - rows: `14,151`;
  - generated `21` candidate submissions.
- Prediction guard for Rust correction candidates vs schema10 anchor:
  - all correction candidates passed the current guard;
  - `rust_corr_bal_raw`: median abs shift `1.780 ft`, P95 `4.863 ft`,
    max `7.500 ft`;
  - `rust_corr_geo_raw`: median abs shift `2.016 ft`, P95 `5.013 ft`,
    max `6.000 ft`;
  - blend40 candidates move much less:
    `rust_corr_bal_blend40` median abs `0.712 ft`, P95 `1.945 ft`,
    max `3.000 ft`;
    `rust_corr_geo_blend40` median abs `0.806 ft`, P95 `2.005 ft`,
    max `2.400 ft`.
- Random 10-well train hidden-row pseudo-public sample:
  - command:
    `make direct-solver-train-eval DIRECT_SOLVER_OUTPUT=artifacts/direct_solver_rust_corr_sample10
    DIRECT_SOLVER_MAX_WELLS=10 DIRECT_SOLVER_SAMPLE_SEED=42
    DIRECT_SOLVER_PROGRESS_INTERVAL=1 DIRECT_SOLVER_BACKEND=rust
    DIRECT_SOLVER_ANCHOR= DIRECT_SOLVER_PSEUDO_PUBLIC_TRIALS=200`;
  - runtime: `02:09`;
  - rows: `49,062`;
  - caveat: no real schema10 train OOF anchor was provided, so safe/blend
    candidates use the internal fallback anchor and should not be read as
    schema10 deltas;
  - `geo_tailfit` was extremely strong on this sampled train-hidden setup
    (weighted RMSE `1.62765`, median triple RMSE `0.41079`), while Rust
    correction variants stayed much worse on this proxy.
- Takeaway:
  - the Rust ABI path works end-to-end;
  - Rust correction candidates now provide bounded 2-7 ft public-test shifts
    without HMM-level path explosions;
  - the train-hidden sample says formation tailfit can be very strong, but this
    may be distribution-specific. A real OOF-anchor pseudo-public harness is
    still needed before trusting raw geo candidates;
  - direct-solver roadmap status after this pass:
    - implemented: uniform 3-well pseudo-public harness and bold
      affine/correction solver;
    - not started: tie-point/landmark solver, CEM iterative refit, MTP CNN
      candidate generator, Stage1/Stage2 local-search reproduction, and relaxed
      prediction-guard policy;
  - train-eval without a real OOF/schema anchor uses an internal fallback anchor,
    so gated train-eval ranks are diagnostic only. Prefer `*_raw` and
    pseudo-public summaries in that mode;
  - next Rust work should be CEM/stage2 local search over low-dimensional path
    families, not more GBM features.

### Direct TVT Path Solver Pack

- Context:
  - after HMM schema13 failed hard on public LB, new path experts must be
    bounded against a known safe anchor and checked with prediction guard before
    any submit;
  - this pack does not train GBM and does not add stack features. It generates
    direct test-time TVT path candidates from fixed solver variants.
- What changed:
  - added `rogii.direct_solver`;
  - added `configs/direct_solver_policy.yml`;
  - added Makefile targets:
    - `make direct-solver-train-eval`;
    - `make direct-solver-test`;
    - `make direct-solver-guard DIRECT_SOLVER_VARIANT=...`;
    - `make direct-solver-use DIRECT_SOLVER_VARIANT=...`;
  - added progress logging:
    - start line with mode/wells/rows/output;
    - per-well start line;
    - progress line with elapsed, ETA, rows/sec;
    - final duration line;
  - added `DIRECT_SOLVER_MAX_WELLS` for fast partial train hidden-row sanity.
- Validation:
  - `uv run pytest -q`: `48 passed`;
  - `make check`: passed;
  - `uv run python -m compileall rogii`: passed.
- Partial train hidden-row sanity:
  - command:
    `make direct-solver-train-eval DIRECT_SOLVER_MAX_WELLS=10
    DIRECT_SOLVER_PROGRESS_INTERVAL=1`;
  - runtime: `01:46`;
  - rows: `46,484`;
  - estimated full 773-well train-eval runtime from this sample:
    roughly `2-2.5h`;
  - caveat: safe variants cannot be ranked honestly in train-eval when the
    anchor submission contains only test ids. In that mode the solver falls
    back to internal paths for the anchor proxy;
  - raw geo sanity looked strong on the first 10 train wells
    (`geo_tailfit`/`geo_*_raw` weighted RMSE `0.23090`), while linear tailfit
    was poor (`681.44225`).
- Test candidate generation:
  - command:
    `make direct-solver-test DIRECT_SOLVER_PROGRESS_INTERVAL=1`;
  - runtime: `30.83s`;
  - generated `12` submission candidates under `artifacts/direct_solver/test`.
- Prediction guard vs clean schema10 anchor
  `ed4d9dc6c7cb479881f087fee1217253`:
  - `consensus_safe`: `PASS`, median abs shift `0.000 ft`,
    P95 `0.202 ft`, max `0.526 ft`;
  - `gr_safe`: `PASS`, median abs shift `0.000 ft`,
    P95 `1.671 ft`, max `2.261 ft`;
  - `geo_safe`: `PASS`, median abs shift `0.000 ft`,
    P95 `0.288 ft`, max `0.700 ft`;
  - `gr_bold`: `PASS`, median abs shift `0.000 ft`,
    P95 `3.217 ft`, max `4.354 ft`;
  - `geo_bold`: `PASS`, median abs shift `0.291 ft`,
    P95 `11.582 ft`, max `16.300 ft`;
  - `geo_bold_raw`: `FAIL`, median abs shift `2.281 ft`,
    P95 `39.946 ft`, max `56.218 ft`;
  - `geo_tailfit`: `FAIL`, median abs shift `47.402 ft`,
    P95 `1160.587 ft`, max `1674.705 ft`.
- Takeaway:
  - logs are now good enough to run the long train-eval deliberately;
  - safe variants are extremely conservative on the public test rows and mostly
    behave like tiny anchor nudges;
  - raw geological paths can move the solution violently and should remain
    diagnostics only unless a stronger guard/report justifies them;
  - if a direct-solver variant is submitted, the only low-risk first choices are
    `submission_direct_consensus_safe.csv` or `submission_direct_gr_safe.csv`,
    but expected LB movement is probably small because shifts are tiny.

### EXP-20260520-8 - Schema15 DTW/DWT Confidence Inference Candidate

- Command/config:
  - trained ClearML task: `fd5fdcf043234e7fad4ba615a24ba956`;
  - local artifact fetch:
    `make fetch-clearml-model CML_ID=fd5fdcf043234e7fad4ba615a24ba956`;
  - inference kernel:
    `make submit CML_ID=fd5fdcf043234e7fad4ba615a24ba956
    MESSAGE="schema15 dtw dwt confidence"`;
  - Kaggle model dataset slug:
    `sleep3r/rogii-baseline-artifacts-fd5fdcf04323`;
  - Kaggle kernel version: `13`.
- Artifact:
  - schema version: `15`;
  - features: `477`;
  - HMM columns: `0`;
  - robust columns: `0`;
  - final model strategy: `full_context`;
  - fold-safe OOF+postprocess RMSE: `10.70543`;
  - best postprocess:
    `alpha=1.05`, `tau=90`, `w_pf_ancc=0.07`, `w_pf_z=0.02`,
    smoothing `(17, 3)`.
- Inference:
  - Kaggle kernel completed successfully in `45.35s`;
  - submission rows: `14,151`;
  - no NaN predictions;
  - prediction range: `11591.078` to `12239.345`.
- Prediction guard vs clean schema10 anchor
  `ed4d9dc6c7cb479881f087fee1217253`:
  - result: `PASS`;
  - global median absolute shift `0.948 ft`;
  - global P95 absolute shift `2.781 ft`;
  - max absolute shift `5.347 ft`;
  - max endpoint absolute shift `1.012 ft`;
  - report:
    `artifacts/prediction_guard_schema15_fd5.md`.
- Kaggle result:
  - not competition-submitted by the CLI (`--skip-competition-submit`);
  - manual UI submit can use
    `artifacts/kaggle_submit_output/submission.csv`.
- Takeaway:
  - this is a safe prediction-shift candidate, not an HMM-style failure;
  - OOF is slightly worse than the clean schema10 isolated anchor
    (`10.70543` vs `10.68983`), so expected LB improvement is uncertain;
  - use this as the last trained schema15 diagnostic candidate, not as proof of
    progress until a public LB is recorded.

### Schema15 DTW/DWT Confidence Pack + Prediction Shift Guard

- Context:
  - after isolated submits, the reproducible baseline is worse than the old
    pre-isolation `9.94/9.95` rows;
  - HMM schema13 proved that local/fold-safe diagnostics can accept a feature
    family that catastrophically shifts public-test predictions;
  - next step is conservative: add confidence/gating information for existing
    DTW/DWT paths, not a new aggressive TVT path.
- What changed:
  - bumped `FEATURE_CACHE_SCHEMA_VERSION` to `15`;
  - added DTW confidence/agreement features:
    - per-radius cost, slope mean/std, local stretch, endpoint gap,
      and deviation from the DTW ensemble;
    - `kg_dtw_radii_agreement = 1 / (1 + kg_dtw_std)`;
    - `kg_dtw_cost_mean`, `kg_dtw_cost_std`;
    - `kg_dtw_best_radius_id`, normalized `kg_dtw_best_radius_margin`;
  - added DWT-lowpass confidence/agreement features:
    - per-radius cost, slope mean/std, local stretch, endpoint gap,
      and deviation from the DWT ensemble;
    - `kg_dwt_radii_agreement = 1 / (1 + kg_dwt_std)`;
    - `kg_dwt_best_radius_id`, normalized `kg_dwt_best_radius_margin`;
    - `kg_dwt_raw_gap = abs(kg_dwt_vs_dtw)`,
      `kg_dwt_vs_dtw_slope_gap`;
  - added `rogii.prediction_guard` and `make prediction-guard` to compare a
    candidate submission against an anchor before spending a Kaggle submit.
  - added explicit endpoint-shift guard on the last hidden prediction;
  - intentionally did not add `kg_dwt_level_agreement`: the current production
    DWT block uses one configured wavelet level and varies only DTW radius on
    the low-pass signal, so a "level agreement" column would be synthetic.
- Guard defaults:
  - anchor: `artifacts/cml_audit/ed4d9dc6c7cb479881f087fee1217253/submission.csv`;
  - hard-fails when a test well exceeds:
    - P95 absolute shift `>25 ft`;
    - median absolute shift `>12 ft`;
    - endpoint absolute shift `>25 ft`;
    - slope/curvature ratio `>3x`;
  - warns on mostly one-sided well shifts.
- Validation:
  - `uv run python -m compileall rogii`: passed;
  - `uv run ruff check rogii tests`: passed;
  - `uv run pytest -q`: `46 passed`;
  - `make quick-train`: passed with schema `15`, features `418`,
    quick OOF+postprocess RMSE `10.14868`;
  - robust schema11 vs clean schema10 guard:
    - command:
      `make prediction-guard
      PREDICTION_GUARD_CANDIDATE=artifacts/cml_audit/cc9e7996dd9e4a82b3e681891b085cd2/submission.csv
      PREDICTION_GUARD_ANCHOR=artifacts/cml_audit/ed4d9dc6c7cb479881f087fee1217253/submission.csv`;
    - result: `PASS`;
    - global median absolute shift `0.578 ft`, P95 `1.731 ft`;
    - max endpoint absolute shift `0.487 ft`;
    - this is deliberately `PASS`, not `WARN`: the shift is below 2 ft P95 on
      all visible test rows, so the earlier plan's desired schema11 warning was
      too conservative for the calibrated guard thresholds.
  - HMM schema13 vs clean schema10 guard:
    - local inference from CML `adfcc7b5806b433e94fa4ff394b161bf`;
    - result: `FAIL`;
    - global median absolute shift `24.065 ft`, P95 `30.001 ft`;
    - max endpoint absolute shift `25.340 ft`;
    - all 3 test wells had one-sided negative shifts.
- Takeaway:
  - the guard would have blocked the HMM `21.064` submit;
  - DTW/DWT confidence pack is low-risk relative to HMM because it adds trust
    indicators for existing paths rather than a new absolute path.

### EXP-20260520-7 - Isolated Schema9 Full-Context Resubmit

- Command/config:
  - `make submit CML_ID=2b77f48a1a294304bf859ea798666d65
    MESSAGE="schema9 full_context isolated check"`;
  - CML task: `2b77f48a1a294304bf859ea798666d65`;
  - source config: `configs/stack_gpu.yml`;
  - artifact dataset slug:
    `sleep3r/rogii-baseline-artifacts-2b77f48a1a29`.
- Artifact:
  - schema version: `9`;
  - features: `414`;
  - HMM columns: `0`;
  - robust columns: `0`;
  - fold-safe OOF+postprocess RMSE: `10.66742`;
  - final model strategy: `full_context`.
- Kaggle result:
  - public LB: `10.229`;
  - Kaggle ref: `52844767`.
- Takeaway:
  - this also does not reproduce the old `9.945` / `9.952` public scores;
  - after CML-specific dataset isolation, neither schema9 `2b77...` nor
    schema10 `ed4d...` is a reliable `9.95` anchor;
  - the old `9.945` / `9.952` submissions were likely produced by a different
    artifact/code combination from the pre-isolation shared Kaggle dataset era,
    or by source-code drift between training and current inference packaging.
- Decision:
  - stop treating pre-isolation leaderboard rows as clean CML-to-LB evidence;
  - future model artifacts must be submitted with a CML-specific Kaggle dataset
    slug and should include enough source/config metadata to reproduce feature
    generation exactly.

### EXP-20260520-6 - Isolated Schema10 Resubmit Corrects The Anchor

- Command/config:
  - `make submit CML_ID=ed4d9dc6c7cb479881f087fee1217253
    MESSAGE="clean schema10 baseline resubmit"`;
  - CML task: `ed4d9dc6c7cb479881f087fee1217253`;
  - source config: `configs/stack_gpu.yml`;
  - artifact dataset slug after the isolation fix:
    `sleep3r/rogii-baseline-artifacts-ed4d9dc6c7cb`.
- Artifact:
  - schema version: `10`;
  - features: `419`;
  - HMM columns: `0`;
  - robust columns: `0`;
  - fold-safe OOF+postprocess RMSE: `10.68983`;
  - final model strategy: `full_context`.
- Kaggle result:
  - public LB: `10.084`.
- Correction:
  - the earlier EXP-20260520-1 attribution of public LB `9.952` to this CML
    task is no longer trustworthy;
  - before the Kaggle artifact dataset isolation fix, several submits reused
    the same `sleep3r/rogii-baseline-artifacts` slug, so the Kaggle kernel could
    mount a different artifact version than the local CML id implied;
  - with the isolated `ed4d...` dataset slug, schema10 `ed4d...` is a `10.084`
    artifact, not a `9.952` artifact.
- Takeaway:
  - do not use `ed4d9dc6c7cb479881f087fee1217253` as the clean 9.95 anchor;
  - the remaining old `9.945` / `9.952` submissions must be revalidated with
    CML-specific dataset slugs before assigning them to `2b77...` or
    `429438...`;
  - from now on, only post-isolation submissions should be treated as reliable
    artifact-to-LB pairs.

### EXP-20260520-5 - HMM Schema13 Public LB Failure

- Command/config:
  - trained/submitted ClearML artifact
    `adfcc7b5806b433e94fa4ff394b161bf`;
  - submit command:
    `make submit CML_ID=adfcc7b5806b433e94fa4ff394b161bf
    MESSAGE="hmm schema13 remount check"`.
- Kaggle plumbing check:
  - after the Kaggle artifact isolation fix, the inference kernel mounted the
    correct CML-specific dataset:
    `/kaggle/input/rogii-baseline-artifacts-adfcc7b5806b`;
  - loaded artifact feature count: `437`;
  - source CV RMSE: `10.05919`;
  - `train_wells=773`, `test_wells=3`;
  - submission rows: `14,151`;
  - no missing-feature warnings in the corrected run.
- Kaggle result:
  - public LB: `21.064`.
- Takeaway:
  - this is not the previous stale-dataset/missing-column failure; the corrected
    kernel used the intended schema13 HMM artifact;
  - HMM schema13 strongly overfit the local/quick diagnostics and generalized
    catastrophically to the public test wells;
  - do **not** use `configs/stack_gpu_hmm.yml` or CML
    `adfcc7b5806b433e94fa4ff394b161bf` for leaderboard submissions.
- Decision:
  - keep HMM only as a quarantined ablation/research branch;
  - submit defaults stay on the clean schema10 family (`hmm_enabled: false`,
    `robust_expert_enabled: false`);
  - future HMM work needs a separate diagnostic explaining why public-test
    predictions shift before another full submit.

### EXP-20260520-3 - HMM Path Expert Ablation Pack

- Context:
  - added the external Viterbi/HMM path expert pack from
    `/Users/alexander/Desktop/hmm_path_expert`;
  - this is intentionally an ablation path, not the default submit config.
- What changed:
  - added `rogii/hmm_path.py`;
  - added `tests/test_hmm_path.py`;
  - added `configs/quick_hmm.yml` and `configs/stack_gpu_hmm.yml`;
  - wired HMM feature generation behind
    `features.kaggle_top.hmm_enabled: true`;
  - bumped `FEATURE_CACHE_SCHEMA_VERSION` to `12`;
  - `make quick-train CONFIG=configs/quick_hmm.yml` now honors the passed
    config instead of always using `configs/quick.yml`;
  - model bundle inputs now include the HMM configs and `rogii/hmm_path.py`.
- New HMM columns:
  - `kg_hmm_tvt`, `kg_hmm_delta`, `kg_hmm_minus_flat`;
  - `kg_hmm_path_cost`, `kg_hmm_emit_cost`, `kg_hmm_transition_cost`;
  - `kg_hmm_confidence_gap`, `kg_hmm_slope`, `kg_hmm_curvature`;
  - `kg_hmm_vs_pf`, `kg_hmm_vs_dtw`;
  - `kg_hmm_state_index`, `kg_hmm_candidate_prior`, `kg_hmm_geo_prior`.
- Validation:
  - `uv run python -m compileall rogii`: passed;
  - `uv run ruff check rogii tests`: passed;
  - `uv run pytest -q tests/test_hmm_path.py tests/test_audit_fixes.py`:
    `43 passed`;
  - `make expert-report CONFIG=configs/quick_hmm.yml
    EXPERT_REPORT_MAX_WELLS=100`: passed;
  - `make quick-train CONFIG=configs/quick_hmm.yml`: passed.
- Quick diagnostics:
  - expert report best standalone candidate:
    `kg_pf_ancc_minus_last__as_tvt`, RMSE `10.32416`;
  - HMM standalone candidates were weak on quick public rows:
    - `kg_hmm_minus_flat__as_tvt`: RMSE `14.61095`;
    - `kg_hmm_delta__as_tvt`: RMSE `14.61114`;
    - `kg_hmm_tvt`: RMSE `14.61117`;
  - quick HMM train produced:
    - features: `398`;
    - raw OOF ensemble RMSE: `10.26238`;
    - OOF+postprocess RMSE: `10.05803`;
    - best postprocess: `alpha=1.05`, `tau=100`,
      `w_pf_ancc=0.20`, `w_pf_z=0.00`, Savitzky-Golay `(17, 3)`.
- Takeaway:
  - HMM integration works technically, but it did **not** pass the standalone
    expert criterion on quick diagnostics;
  - the better quick-train number is too small/noisy to justify a 7-hour full
    CML run by itself;
  - next step, if we keep pursuing HMM, is to tune/debug the HMM expert on
    larger expert reports before launching full training.

### EXP-20260520-4 - Tuned HMM Around PF_ANCC Anchor

- Context:
  - raw HMM from EXP-20260520-3 was worse than `PF_ANCC`;
  - quick diagnostics showed the HMM path helped some wells but failed when
    its PF anchor was polluted by `PF_Z`.
- What changed:
  - HMM state grid is now local around candidate paths instead of spanning the
    whole typewell TVT range by default;
  - HMM uses `PF_ANCC` as the primary PF anchor, with fallback to the old PF
    median only when ANCC candidates are missing;
  - added HMM/PF confidence-gated outputs:
    - `kg_hmm_pf_abs_gap`;
    - `kg_hmm_pf_gated_tvt`;
    - `kg_hmm_pf_gated_delta`;
    - `kg_hmm_pf_gated_minus_flat`;
  - tuned `quick_hmm.yml` and `stack_gpu_hmm.yml` toward a PF-anchored path:
    - lower GR/typewell weight;
    - higher PF_ANCC weight;
    - local candidate state span;
    - `hmm_pf_gate_threshold: 6.0`;
  - bumped `FEATURE_CACHE_SCHEMA_VERSION` to `13`.
- Validation:
  - `uv run ruff check rogii tests`: passed;
  - `uv run python -m compileall rogii`: passed;
  - `uv run pytest -q tests/test_hmm_path.py`: `2 passed`;
  - `uv run pytest -q`: `43 passed`;
  - `make expert-report CONFIG=configs/quick_hmm.yml
    EXPERT_REPORT_MAX_WELLS=100`: passed;
  - `make quick-train CONFIG=configs/quick_hmm.yml`: passed.
- Quick diagnostics:
  - best standalone expert is now `kg_hmm_tvt`, RMSE `9.71811`;
  - `kg_hmm_pf_gated_tvt` RMSE `9.82155`;
  - previous `PF_ANCC` anchor RMSE `10.32416`;
  - quick HMM train:
    - features: `402`;
    - raw OOF ensemble RMSE: `9.71731`;
    - OOF+postprocess RMSE: `9.59224`;
    - best postprocess: `alpha=1.05`, `tau=100`,
      `w_pf_ancc=0.20`, `w_pf_z=0.00`, Savitzky-Golay `(17, 3)`.
- Takeaway:
  - HMM is now a plausible ablation candidate rather than a broken standalone
    path;
  - quick public rows are still too small for certainty, but this clears the
    bar for a controlled `stack_gpu_hmm.yml` run if we want to spend the hour.
- Server run attempt:
  - command:
    `INSTANCE=el CONFIG=configs/stack_gpu_hmm.yml SERVER_NOTES=hmm_schema13_pf_ancc_anchor make train-server`;
  - Docker image build succeeded;
  - Docker push failed before Portainer launch:
    `lookup harbor.wildberries.ru ... no such host`;
  - no ClearML/server training result from this attempt.

### Kaggle Artifact Dataset Isolation Fix

- Problem:
  - `make submit CML_ID=adfcc7b5806b433e94fa4ff394b161bf` was expected to
    submit the schema13 HMM artifact;
  - Kaggle logs instead showed `features=432`, `cv_rmse=10.63946`, and missing
    `kg_signal_robust_*` warnings;
  - local ClearML artifact `adfcc7...` was correct:
    `features=437`, schema `13`, `hmm=18`, `robust=0`.
- Root cause:
  - all submits reused the same Kaggle dataset slug
    `sleep3r/rogii-baseline-artifacts`;
  - the kernel mounted an older dataset version/artifact while local files had
    already been replaced;
  - old schema11 artifacts also lacked the new explicit
    `robust_expert_enabled` config flag, so current inference code could skip
    columns that existed in `features.json`.
- Fix:
  - `MODEL_DATASET` now defaults to a CML-specific slug:
    `sleep3r/rogii-baseline-artifacts-<first12-cml-id>`;
  - Kaggle dataset title is kept under Kaggle's 50-character limit;
  - `rogii.inference` now auto-enables feature gates from artifact
    `features.json`:
    - robust pack if robust columns are present;
    - HMM pack if `kg_hmm_*` columns are present;
  - `kaggle_submit` logs artifact feature count/schema/robust/HMM counts before
    publishing the model dataset.
- Validation:
  - `uv run ruff check rogii tests`: passed;
  - `uv run python -m compileall rogii`: passed;
  - `uv run pytest -q`: `43 passed`;
  - `make submit-dry CML_ID=adfcc7b5806b433e94fa4ff394b161bf`: passed and
    used dataset source
    `sleep3r/rogii-baseline-artifacts-adfcc7b5806b`;
  - `make submit CML_ID=adfcc7b5806b433e94fa4ff394b161bf
    MESSAGE="hmm schema13 remount check"` completed Kaggle kernel version `10`.
- Kaggle inference check:
  - mounted model dir:
    `/kaggle/input/rogii-baseline-artifacts-adfcc7b5806b`;
  - loaded `features=437`;
  - source CV RMSE `10.05919`;
  - `train_wells=773`, `test_wells=3`;
  - submission rows `14,151`;
  - no missing-feature warnings in the corrected run.

### EXP-20260520-2 - Conservative schema10 default plus PF_Z postprocess candidate

- Context:
  - after EXP-20260520-1, robust schema-v11 features were proven harmful on
    public LB despite better OOF;
  - the next safe step was to prevent accidental full-runs with schema-v11 as
    the default and add a cheap `pf_z` postprocess candidate without touching
    the heavy feature/model stack.
- What changed:
  - `FEATURE_CACHE_SCHEMA_VERSION` set back to `10`;
  - `features.kaggle_top.robust_expert_enabled: false` added to default,
    `stack.yml`, `stack_gpu.yml`, and `quick.yml`;
  - robust PF/beam columns are now emitted only when
    `robust_expert_enabled: true`;
  - `model.blend.allow_negative_weights` set back to `false` in default,
    `stack.yml`, and `quick.yml`;
  - notebook postprocess now supports multiple PF columns through:
    - `pf_columns: [pf_ancc, pf_z]`;
    - `w_pf_ancc_*`;
    - `w_pf_z_*`;
    - `w_pf_total_max`;
  - both grid and Optuna postprocess scoring use the same multi-PF basis scorer.
- Config defaults:
  - `stack.yml` and `stack_gpu.yml` search:
    - `w_pf_ancc_range: [0, 0.20, 0.01]`;
    - `w_pf_z_range: [0, 0.10, 0.01]`;
    - `w_pf_total_max: 0.30`;
  - `quick.yml` uses compact grids for the same parameters.
- Validation:
  - `uv run python -m compileall rogii`: passed;
  - `uv run ruff check rogii tests`: passed;
  - `uv run pytest -q`: `41 passed`;
  - `make quick-train`: passed in `09.18s`;
  - quick result:
    - features: `384` (robust columns absent);
    - raw OOF ensemble RMSE: `10.43919`;
    - OOF+postprocess RMSE: `10.13017`;
    - best postprocess: `alpha=1.05`, `tau=100`,
      `w_pf_ancc=0.20`, `w_pf_z=0.00`, Savitzky-Golay `(17, 3)`.
- Takeaway:
  - this restores the submit default to the clean schema-v10 family;
  - `pf_z` is now available in postprocess, but quick smoke did not select it;
  - no new full-run is justified from this alone until the public-notebook gap
    analysis points at a stronger change.

### EXP-20260520-1 - Clean Schema v10 vs Robust Schema v11 CML Audit

- Context:
  - user reported Kaggle public LB `9.952` for the clean main run without the
    PF/beam robust feature pack;
  - user reported Kaggle public LB `10.084` for the latest run with the robust
    schema-v11 feature pack;
  - before changing code, downloaded ClearML artifacts/logs for the relevant
    CML tasks under `artifacts/cml_audit/`.
- Clean/reference task:
  - CML task: `ed4d9dc6c7cb479881f087fee1217253`;
  - name: `rogii-stack_gpu-run_20260519_192601`;
  - status: `completed`;
  - notes: `clean_head_no_robust`;
  - config: `configs/stack_gpu.yml`;
  - feature schema: `10`;
  - feature count: `419`;
  - robust columns present: `false`;
  - validation mode: fold-safe context, final model strategy `full_context`;
  - model family: 3 CatBoost GPU + 3 XGBoost CUDA variants;
  - raw OOF ensemble RMSE: `10.75071`;
  - OOF+postprocess RMSE: `10.68983`;
  - final train/OFF prediction RMSE: `10.69085`;
  - baseline RMSE: `15.90987`;
  - best postprocess: `alpha=1.05`, `tau=80`, `w_pf=0.08`,
    Savitzky-Golay `(17, 3)`;
  - blend weights:
    - `cat_lr025=0.11822`;
    - `cat_lr020=0.22443`;
    - `cat_lr030=0.30963`;
    - `xgb_lr025=0.19841`;
    - `xgb_lr020=0.23900`;
    - `xgb_lr030=-0.08969`;
  - diagnostics:
    - mean well RMSE `8.23719`;
    - median well RMSE `6.66675`;
    - P90 well RMSE `14.49605`;
    - P95 well RMSE `19.90347`;
    - worst well RMSE `51.41239`;
    - long hidden RMSE `11.42930`;
    - short hidden RMSE `9.46799`;
  - Kaggle public LB:
    - originally attributed as `9.952` from the Kaggle UI;
    - superseded by EXP-20260520-6: after CML-specific dataset isolation, this
      exact artifact scores `10.084`;
    - treat the earlier `9.952` attribution as unreliable.
- Robust schema-v11 task:
  - CML task: `cc9e7996dd9e4a82b3e681891b085cd2`;
  - name: `rogii-stack_gpu-run_20260519_194125`;
  - status: `completed`;
  - notes: `spacebridge_train`;
  - config: `configs/stack_gpu.yml`;
  - feature schema: `11`;
  - feature count: `432`;
  - robust columns present: `true`;
  - validation mode: fold-safe context, final model strategy `full_context`;
  - model family: 3 CatBoost GPU + 3 XGBoost CUDA variants;
  - raw OOF ensemble RMSE: `10.69313`;
  - OOF+postprocess RMSE: `10.63946`;
  - final train/OFF prediction RMSE: `10.64113`;
  - baseline RMSE: `15.90987`;
  - best postprocess: `alpha=1.05`, `tau=90`, `w_pf=0.07`,
    Savitzky-Golay `(17, 3)`;
  - blend weights:
    - `cat_lr025=0.19589`;
    - `cat_lr020=0.73850`;
    - `cat_lr030=-0.17571`;
    - `xgb_lr025=-0.46637`;
    - `xgb_lr020=0.73876`;
    - `xgb_lr030=-0.03108`;
  - diagnostics:
    - mean well RMSE `8.29187`;
    - median well RMSE `6.66673`;
    - P90 well RMSE `15.05909`;
    - P95 well RMSE `19.72334`;
    - worst well RMSE `50.55713`;
    - long hidden RMSE `11.33270`;
    - short hidden RMSE `9.50263`;
  - Kaggle public LB: `10.084` (reported from Kaggle UI).
- Stopped/intermediate robust attempt:
  - CML task: `98afa683779b4bffaafc60b3ae7aeeea`;
  - name: `rogii-stack_gpu-run_20260519_192632`;
  - status: `stopped`;
  - reached fold 1 feature prep and started `cat_lr025`;
  - no final artifacts.
- Takeaway:
  - schema v11 robust PF/beam features improved local fold-safe OOF
    (`10.63946` vs `10.68983`), but the public-LB comparison against schema10
    was polluted by the pre-isolation shared Kaggle dataset slug;
  - after EXP-20260520-6, schema10 `ed4d...` and schema11 `cc9e...` both have
    observed public LB `10.084` under the old/new submit history, so this pair
    is not evidence of a robust-pack public improvement;
  - robust standalone expert looked good on 100-well diagnostics, but the full
    model did not produce a better reliable LB anchor;
  - do not treat schema v11 robust pack as a default improvement.
- Decision:
  - no code rollback was performed in this audit step;
  - before the next full train, either disable/remove the robust pack from the
    main submit config or gate it behind a controlled ablation;
  - avoid further 7h experiments until the feature/candidate gap to the best
    open notebook is audited more directly.

## 2026-05-19

### PF/Beam Robust Expert Pack

- Added schema v11 feature pack focused on the expert-report finding that
  `PF_ANCC` is the strongest standalone path while blind `kg_signal_mean_*`
  is polluted by weak DTW/DWT/NCC/spatial candidates.
- New features:
  - `kg_signal_robust_tvt`;
  - `kg_signal_robust_minus_last`;
  - `kg_signal_robust_minus_flat`;
  - `kg_signal_robust_std`;
  - `kg_signal_robust_range`;
  - `kg_signal_robust_vs_pf`;
  - `kg_signal_robust_vs_beam`;
  - `pf_ancc_conf`, `beam_conf`;
  - `pf_beam_gap`, `pf_beam_abs_gap`, `pf_dtw_gap`, `pf_dwt_gap`.
- The old `kg_signal_mean_*` columns are intentionally kept unchanged for
  compatibility and ablation safety.
- Implemented the same formula in full feature building and split-cache
  context rebuilding, so fold-safe OOF and inference stay aligned.
- Validation:
  - `uv run ruff check .`: passed;
  - `uv run python -m compileall rogii`: passed;
  - `uv run pytest -q`: 38 passed;
  - `make check`: passed;
  - `make quick-train`: passed, quick OOF ensemble RMSE `10.35219`,
    OOF+postprocess RMSE `10.02730`, features `397`;
  - `make expert-report CONFIG=configs/quick.yml ...`: robust candidate RMSE
    `10.78672` vs `kg_signal_mean_*` RMSE `90.37376`;
  - `make expert-report CONFIG=configs/stack_gpu.yml EXPERT_REPORT_MAX_WELLS=100 ...`:
    `kg_signal_robust_minus_flat__as_tvt` became the best standalone expert
    with RMSE `12.86215`, ahead of `PF_ANCC` RMSE `13.81448` and old
    `kg_signal_mean_*` RMSE `70.01170`.
- Full train/LB result: pending after the current running baseline finishes.

### Candidate Expert Report

- Added `rogii.expert_report` and `make expert-report`.
- Purpose: rank absolute TVT candidate paths before training another GBM stack,
  so the next full experiment can target weak experts instead of chasing small
  LB noise.
- Default report uses fold-safe contexts:
  - for each fold, context is built from train-fold wells;
  - only validation-fold rows are scored;
  - this avoids optimistic spatial/dense/formation diagnostics.
- Outputs:
  - `artifacts/expert_report.md`;
  - `artifacts/expert_report.csv`;
  - `artifacts/expert_report.json`.
- Supports `EXPERT_REPORT_CONTEXT=full` for faster full-context diagnostics and
  `EXPERT_REPORT_MAX_WELLS=<n>` for quick checks.
- Smoke:
  - `make expert-report CONFIG=configs/quick.yml ...`: passed;
  - quick best candidate was `kg_pf_ancc_minus_last__as_tvt` with RMSE
    `10.32416`.

### Ravaghi-Style Blend/Postprocess Quick Wins

- Enabled negative hill-climb weights in the main blend config:
  - `model.blend.allow_negative_weights: true`;
  - coordinate moves can now go in both positive and negative directions, with
    weights normalized by their sum rather than by absolute values.
- Replaced the broad postprocess grid in `stack.yml` and `stack_gpu.yml` with
  deterministic Optuna TPE:
  - `postprocess.search_method: optuna`;
  - `postprocess.optuna_trials: 500`;
  - fixed seed `42`.
- Fixed `residual_weight` at `1.0` and removed `residual_weight_grid` from the
  main configs to reduce OOF overfit degrees of freedom.
- Fixed smoothing to Ravaghi-style `savgol(window=17, polyorder=3)` instead of
  searching no-op vs smoothing.
- `quick.yml` keeps grid postprocess for smoke speed, but now shares negative
  blend weights, fixed residual weight, and fixed smoothing.
- `stack_gpu.yml` makes these settings explicit rather than relying on
  inheritance.
- Dependency update: added `optuna>=4.0`.
- Validation:
  - `uv run pytest -q`: 38 passed;
  - `uv run python -m compileall rogii`: passed;
  - `uv run ruff check .`: passed;
  - `make quick-train`: passed, OOF ensemble RMSE `10.43739`,
    OOF+postprocess RMSE `10.07086`.
- Full result: pending next server/local train.

### Fold-Average Submit A/B

- Submitted the ClearML task `4294380230f645feb4a2189cdd7beeb8`
  (`rogii-stack_gpu-run_20260519_155906`) through the inference-only Kaggle
  path.
- First attempt produced Kaggle kernel version 5 but failed because the
  ClearML artifact `config.yml` had `data.clearml.enabled: true`; the Kaggle
  notebook already has competition data mounted and does not install ClearML.
- Fixed `rogii.inference`: an explicit `--data-dir` now disables
  `data.clearml.enabled` before dataset preparation, so Kaggle inference never
  tries to import ClearML when competition data is supplied directly.
- Retried as Kaggle kernel version 6:
  - output downloaded and validated locally;
  - `submission.csv` has 14,151 rows;
  - public LB: `9.952`.
- A/B result:
  - version 4 full-context baseline: `9.945`;
  - version 6 fold-average artifact: `9.952`.
- Takeaway: fold averaging is not worth making the default. The delta is small
  but worse, and the train/inference context mismatch remains conceptually
  ugly. `configs/stack_gpu.yml` is back to
  `validation.final_model_strategy: full_context`.

### Public DWT Feature Schema Parity

- Compared the local feature builder against the visible Roman/Ravaghi/DWT
  notebook code without downloading the large `ravaghi/.../train.csv` artifact.
- Confirmed the heavy public blocks are already in our pipeline:
  PF_ANCC/PF_Z, beam paths, multi-scale NCC, multi-radius DTW, DWT-lowpass DTW,
  formation-plane KNN, dense ANCC, segment biases, and
  `tda*`/`tdbc*`/`tdsc*`/`tdpf*`/`tddtw*` offset residual features.
- Restored public-compatible geometry aliases used by those notebooks:
  `md_since`, `dx`, `dy`, `dz`, and `dxy`.
  These duplicate the canonical `*_from_last_known` columns, but they make the
  schema closer to the public DWT baseline and can matter for sampled tree
  models.
- Matched the public aggregate signal construction more closely:
  `sig_mean_d`, `sig_std`, and `kg_signal_*` now aggregate individual beam
  paths, NCC window paths, NCC ensemble, ANCC formation, dense ANCC, PF_ANCC,
  DTW, and DWT-lowpass signals instead of the coarser
  `beam_mean`/`form_mean`/hybrid mix. The split-cache context builder uses the
  same aggregate recipe as the full builder.
- Kept `idx_since` and `tvt_input_isna` out of the schema.
- Bumped `FEATURE_CACHE_SCHEMA_VERSION` to 10. Old schema-v9 feature caches and
  artifacts are invalid for the next train.
- Added a unit test that the aliases match the canonical columns exactly.

### ClearML Artifact Submit Path

- Added ClearML-task based submit artifacts:
  - `CML_ID` defaults to `2b77f48a1a294304bf859ea798666d65`;
  - `INFER_MODEL_DIR` defaults to `artifacts/clearml/$(CML_ID)`;
  - `make submit` now passes `--clearml-task-id $(CML_ID)` to
    `rogii.kaggle_submit`.
- Added `rogii.kaggle_submit fetch-clearml` and `make fetch-clearml-model`.
  The downloader pulls `output/model.pkl`, `output/features.json`,
  `output/metrics.json`, `output/config.yml`, and `output/source_config.yml`
  from the ClearML task into the model dir.
- `make submit` now performs:
  ClearML task artifact download -> local model dir -> private Kaggle model
  dataset -> inference kernel.
- Validated with task `2b77f48a1a294304bf859ea798666d65`:
  - downloaded `model.pkl` (`238M`), `features.json`, `metrics.json`,
    `config.yml`, and `source_config.yml`;
  - `make submit-dry MESSAGE="cml 2b77 dry"` prepared the Kaggle inference
    kernel successfully.

### Validation Anchor Hardening Pass

- Updated `configs/stack_gpu.yml` to be a hybrid server config:
  - CatBoost stays on GPU;
  - LightGBM is removed from the server stack;
  - XGBoost CUDA variants `xgb_lr025`, `xgb_lr020`, and `xgb_lr030` replace the
    LightGBM variants.
- `configs/stack_gpu.yml` now uses `validation.final_model_strategy:
  fold_average` to keep the server run practical after OOF; the main
  `stack.yml` stays `full_context` for the stricter local/anchor path.
- Rationale: the Portainer run showed CatBoost GPU folds finishing in under a
  minute, while LightGBM spent 40+ minutes on the first fold both through
  OpenCL/GPU and CPU. The server config now avoids LightGBM entirely until we
  have a separate reason to debug/tune it.
- Added XGBoost CUDA detection and a one-tree smoke fit to `rogii.gpu_preflight`
  so a broken XGBoost/CUDA runtime fails before feature preparation starts.
- Switched the default final model strategy back to `full_context`:
  - `DEFAULT_CONFIG.validation.final_model_strategy`;
  - `configs/stack.yml`;
  - `configs/quick.yml`;
  - `configs/stack_gpu.yml` now makes the same server-side choice explicit
    instead of relying only on inheritance.
- Rationale: fold-average inference is fast, but fold models are trained on
  fold-safe context features and then applied to full-context inference
  features. Until an A/B proves that mismatch is harmless, the main submit path
  should pay the extra final fit and produce a cleaner LB anchor.
- Reduced the main `stack.yml` notebook postprocess search from the very wide
  exploratory grid to a conservative band around the previous useful optimum:
  - `alpha_range: [0.95, 1.05, 0.01]`;
  - `tau_range: [40, 120, 5]`;
  - `w_pf_range: [0, 0.15, 0.01]`.
  `configs/stack_gpu.yml` repeats these values explicitly for remote runs.
- Made `postprocess.notebook_blend.pf_column` explicit and required whenever
  notebook blend is enabled. This removes the hidden default to
  `kg_pf_ancc_tvt` and avoids silently switching PF columns between configs.
- Removed the constant `tvt_input_isna` training feature. With
  `target_rows: hidden_only`, train and inference rows are always hidden rows,
  so the column carried no signal.
- Bumped `FEATURE_CACHE_SCHEMA_VERSION` to 9. Old schema-v8 feature caches and
  artifacts are invalid for the next clean train.
- Added tests for:
  - missing explicit PF postprocess column raising a clear error;
  - missing PF feature column no-oping the notebook blend safely;
  - `tvt_input_isna` staying out of the feature schema.
- Validation:
  - `uv run pytest -q`: 33 passed;
  - `make check`: passed;
  - `uv run python -m compileall rogii`: passed;
  - `make quick-train`: passed, final strategy `full_context`, 14,151 rows,
    379 features, OOF+PP RMSE 10.05721.
- Next:
  - clear stale `artifacts/feature_cache` and `artifacts/stack`;
  - run one honest full-context, fold-safe server train;
  - submit it as the new clean LB anchor before changing feature families again;
  - run a separate fold-average A/B later if we want the speedup back.

### Split-Cache + Parallel Feature Preparation

- Bumped `FEATURE_CACHE_SCHEMA_VERSION` to 8.
- Split feature caching into two layers:
  - `context_free` caches geometry/GR/typewell/alignment/PF/offset features once
    per well and ignores `KaggleTopContext.context_key`;
  - `context` caches fold/full-context spatial formation, dense ANCC, and
    context-aware aggregate signal columns with `context_key` included.
- Added `features.num_workers`:
  - default/config `stack.yml` and `quick.yml`: `1` for deterministic local
    debugging;
  - `configs/stack_gpu.yml`: `8` for server/GPU runs.
- Added process-based per-well feature preparation. Completed worker outputs are
  sorted by original path order before concatenation, so row order stays
  deterministic.
- Feature table logs now report workers, cache layers, context key, rows/sec,
  and final rows/sec.
- Added tests for split cache keys, serial/parallel feature parity, and worker
  error reporting with well name.
- Validation:
  - `uv run pytest -q`: 31 passed;
  - `make check`: passed;
  - `uv run python -m compileall rogii`: passed;
  - `make quick-train`: passed, 14,151 rows, 380 features, OOF+PP RMSE 10.05721;
  - `make profile-features PROFILE_NAME=features_stack_gpu25_split_workers CONFIG=configs/stack_gpu.yml FEATURE_PROFILE_WELLS=25 FEATURE_PROFILE_STAGE=false`:
    25 wells, 117,140 rows, 415 features in 4.59s.
- Rust remains deferred until the next profiling pass shows a stable kernel-level
  bottleneck after split-cache and workers.

### GPU Container Build Hardening

- Restored LightGBM GPU params in `configs/stack_gpu.yml`.
- Added Docker OpenCL runtime pieces:
  - `clinfo`;
  - `ocl-icd-libopencl1`;
  - `/etc/OpenCL/vendors/nvidia.icd` pointing at `libnvidia-opencl.so.1`;
  - `NVIDIA_VISIBLE_DEVICES=all`;
  - `NVIDIA_DRIVER_CAPABILITIES=compute,utility`.
- Added `rogii.gpu_preflight` and wired it into Docker `CMD` before training.
  It detects GPU models from the resolved config, runs `nvidia-smi -L`, prints
  `clinfo -l`, and performs a one-tree LightGBM GPU smoke fit. A broken
  OpenCL/LightGBM runtime now fails immediately instead of after feature prep.

### Spacebridge + ClearML Server Training Pass

- Added top-level experiment destination config:
  - `project_name: ROGII/Wellbore`;
  - `output_uri: s3://s3-basket-cold.wb.ru/ds-experiments`;
  - ClearML Task, ClearML Dataset, and spacebridge defaults inherit these
    values unless explicitly overridden.
- Added ClearML Dataset support:
  - `make upload-clearml-data` uploads the local unpacked `data/` tree to
    ClearML Dataset, excluding zip archives by default;
  - upload now requires an `s3://` output URI and uses version `20260519_s3`
    by default after the initial fileserver attempt proved too slow;
  - `make clearml-data-local-path` resolves/downloads the dataset locally;
  - `data.clearml` config block plus CLI/env overrides let train/inference use
    ClearML data instead of a local `data/` directory.
- Uploaded dataset:
  - project/name/version:
    `ROGII/Wellbore` / `rogii-wellbore-geology-prediction` / `20260519_s3`;
  - dataset id: `48555c7ef0d44dc8a1eb1406cd7d4fcb`;
  - selected files: 2,343;
  - compressed upload size: 1.07 GiB, 3 chunks;
  - S3 upload runtime: 1m43s, versus the interrupted fileserver upload that was
    still crawling after several minutes.
- Added a spacebridge-compatible Docker path:
  - `Dockerfile` builds a CUDA/uv image and runs `uv run python -m rogii $CMD_ARGS`;
  - `portainer.yml.example` documents the local Portainer/ClearML secret setup;
  - `.dockerignore` keeps generated artifacts and mining output out of images
    and now excludes `data/`; remote training fetches the ClearML Dataset.
- Added Make targets:
  - `make train-server INSTANCE=...`;
  - alias `make train-spacebridge INSTANCE=...`;
  - configurable `SPACEBRIDGE_CMD_ARGS`, `CLEARML_*`, and `SERVER_NOTES`;
  - `make check-server-env` validates Docker/Buildx plus local `portainer.yml`
    without requiring `IMAGE_NAME`; spacebridge owns Harbor image naming and
    auto-tagging from `REGISTRY_USERNAME`;
  - by default, spacebridge passes `data_clearml_enabled=true` and the ClearML
    Dataset project/name/version.
- Added optional ClearML tracking:
  - `tracking.clearml` config block in defaults, `stack.yml`, and `quick.yml`;
  - CLI/env overrides such as `--clearml_enabled=true` and
    `ROGII_CLEARML_ENABLED=true`;
  - resolved config, scalar metrics, submission, and every file under the run
    output directory are logged when enabled;
  - `model.pkl` upload is enabled for server runs by default via
    `CLEARML_LOG_MODEL=true`;
  - because the task `output_uri` points to S3, model/artifact uploads land in
    `s3://s3-basket-cold.wb.ru/ds-experiments`.
- Fixed ClearML Dataset download compatibility with server ClearML versions
  where `Dataset.get_local_copy()` does not accept `local_cache_path`; those
  runs now fall back to ClearML's default cache instead of failing before
  training.
- Removed the local `artifacts/runs.csv` registry from the training path; ClearML
  is now the single experiment registry, while per-run details stay in
  `metrics.json` and ClearML scalars/artifacts.
- Added feature-table heartbeat logging with ETA via `features.progress_interval`
  so cold fold-safe server runs do not look stuck while the first 100 wells are
  still building.
- Added `make profile-features` / `rogii.feature_profile` for feature-prep-only
  profiling with per-stage logs for CSV/typewell, beam/NCC/DTW/DWT, spatial,
  particle filters, residual offsets, and frame materialization.
- Profiled `configs/stack.yml` feature prep and found the bottleneck in dense
  ANCC KNN: `impute_dense_ancc` queried `dense_fetch=5000` neighbors per row.
  It now queries only `dense_k + same-well dense points + 8`, enough to preserve
  exact self-well exclusion while cutting the micro-profile from `9.67s` to
  `1.38s` for 3 wells (`impute_dense_ancc`: `8.274s -> 0.084s`).
  A 25-well cold feature-prep profile now builds `117,140` rows and `415`
  features in `10.69s`.
- Added spatial sub-stage logs:
  `top.spatial.formations` and `top.spatial.dense`.
- Added fold-level validation metrics to long OOF runs:
  - baseline RMSE per validation fold before model training;
  - per-model fold RMSE with delta vs baseline and fitted iteration count;
  - per-fold summary with best single model and equal-weight ensemble RMSE;
  - post-hill-climb weighted ensemble RMSE per fold;
  - the same fold details are stored under `metrics.json -> model.folds` and
    `metrics.json -> model.base_models[].folds`, so ClearML records them too.
- Added explicit server GPU model config:
  - `configs/stack_gpu.yml` inherits `stack.yml` and sets CatBoost
    `task_type=GPU, devices=0`;
  - LightGBM gets `device_type=gpu, gpu_device_id=0, gpu_use_dp=false,
    max_bin=63`;
  - server runs use the normal `CONFIG=...` Make variable, so GPU training is
    launched explicitly as `make train-server INSTANCE=... CONFIG=configs/stack_gpu.yml`;
  - model fits log `Model backend config`, so GPU params are visible before
    the expensive fold starts;
  - the Docker runtime installs the OpenCL ICD loader required by LightGBM's
    GPU backend.
- Removed the obsolete `features.kaggle_top.mode` switch and its proxy/fallback
  branches. The top-solution signal block now has one path: numba-backed beam,
  DTW/DWT, PF_Z, and PF_ANCC.
- Added `rogii/train.py` as a thin entrypoint for tools expecting a train module.
- Validation:
  - `uv lock`: passed with `spacebridge` from nexus;
  - `uv run spacebridge train --help`: passed after adding explicit `pydantic`;
  - `uv run pytest -q`: 26 passed;
  - `uv run python -m ruff check .`: passed;
  - `uv run python -m compileall rogii`: passed;
  - `make quick-train`: passed with ClearML disabled by config, top-level
    `project_name`/`output_uri` resolved, and the S3 artifact-reporting step as
    a no-op.

### Pre-Full-Run Hardening Pass

- `FEATURE_CACHE_SCHEMA_VERSION` bumped to 7; old schema-v6 caches/artifacts are
  invalid for the next clean full run.
- Removed duplicate last-known offset aliases from the feature table:
  - dropped `dx`, `dy`, `dz`, and `dxy`;
  - kept canonical `x_from_last_known`, `y_from_last_known`,
    `z_from_last_known`, and `xy_dist_from_last_known`.
- Unified notebook postprocess PF column to `pf_ancc` in defaults, `quick.yml`,
  and `stack.yml`.
- Expanded `configs/stack.yml` postprocess residual grid to
  `[0.9, 1.0, 1.05]`; this is now cheap after the exact basis scorer.
- Hardened sample-submission inference:
  - malformed sample ids now raise `ValueError`;
  - row ids outside a well prediction array now raise `IndexError` instead of
    silently indexing the wrong row.
- Strengthened tests:
  - DWT repro smoke now runs with `dwt_enabled: true`;
  - canonical PF column is checked across default/quick/stack configs;
  - out-of-range sample ids are covered.
- Validation:
  - `uv run pytest -q`: 19 passed;
  - `uv run python -m ruff check .`: passed;
  - `make quick-train`: 14,151 rows, 380 features, fold-safe OOF+PP RMSE
    10.06087, final summary RMSE 10.06095.
- Next: rebuild the model bundle, then start a clean full `make train-local`
  after clearing stale caches/artifacts.

### Capacity + Fold-Average Inference Pass

- Increased main `configs/stack.yml` model capacity toward the public DWT
  notebook setup:
  - CatBoost variants now use `iterations: 8000`;
  - LightGBM variants now use `n_estimators: 8000`;
  - early stopping patience is now `300`.
- Switched the main inference strategy to fold averaging:
  - `validation.final_model_strategy: fold_average`;
  - fold-safe OOF models are retained in the artifact;
  - inference averages each base model across its fold models before applying
    hill-climb blend weights.
- Kept final full-train `KaggleTopContext` for test feature generation, but
  skipped the final full-context model fit under fold-average strategy.
- Widened `stack.yml` postprocess search using compact ranges:
  - `alpha_range: [0.50, 1.10, 0.01]`;
  - `tau_range: [0, 500, 5]`;
  - `w_pf_range: [0, 0.50, 0.01]`;
  - with 3 residual weights and 2 smoothing candidates this is 1,885,266
    candidates, now feasible because scoring uses the exact basis cache.
- Added `*_range` support for notebook-blend grids so large searches do not
  require enormous YAML lists.
- Validation:
  - `uv run pytest -q`: 21 passed;
  - `uv run python -m ruff check .`: passed;
  - `make quick-train`: fold-average smoke passed, 9 fold models retained,
    OOF+PP RMSE 10.06087.
- Expected impact:
  - full training will be slower per fold because model ceilings are higher;
  - final full-context fit is skipped, partly offsetting runtime;
  - artifact size will grow because it stores 30 fold models for the full stack.

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
- Kaggle result: public LB 9.946, submission ref `52780788`, submitted from
  inference-only Kaggle kernel `sleep3r/rogii-global-stack-infer` version 2.
- Important caveat: this is a **global-context / leaky OOF** anchor. Validation
  features were built with `KaggleTopContext` containing validation wells, and
  the old pipeline predicted test from averaged fold models rather than final
  full-context models trained on all rows.
- Takeaway: the DWT-style feature table is in the right family, close to the
  public DWT notebook's reported single-model OOF range. Public LB improved from
  the old HGB anchor 12.803 to 9.946, but the local OOF is still not an honest
  CV target. Use this run as the current public-LB anchor, not as validation
  truth.
- Next: rerun clean schema v7 fold-safe full training after clearing
  `artifacts/feature_cache` and `artifacts/stack`.

### Fold-Safe Validation

- Reworked the main trainer so OOF features are built fold-by-fold with
  `KaggleTopContext` constructed only from train-fold wells.
- Final artifacts now train separate full-context final models on all train
  wells after OOF weights/postprocess are selected.
- `FEATURE_CACHE_SCHEMA_VERSION` bumped to 6; feature cache keys now include the
  `KaggleTopContext` key, so fold-safe and full-context features cannot collide.
- Added run diagnostics with global RMSE, per-well RMSE, P90/worst well RMSE,
  typewell/no-typewell slices, and hidden-length slices.
- Public config additions:
  - `validation.fold_safe_context: true`;
  - `validation.final_model_strategy: full_context`.
- Validation:
  - `uv run pytest -q`: 13 passed;
  - `make check`: passed;
  - `uv run python -m compileall rogii`: passed;
  - `make quick-train`: 14,151 rows, 384 features, fold-safe OOF+PP RMSE
    10.04130;
  - quick artifact inference parity: same ids, no NaN, max_abs_diff 0.0.
- Full local result: pending; clean caches/artifacts before the next full run.

### Profiling Baseline

- Added `make profile`, `make profile-quick`, `make profile-train`, and
  `make profile-report`.
- Added `rogii.profile_report` to convert `cProfile` output plus ROGII logs into
  Markdown reports under `artifacts/profiles/`.
- Added `PROFILING.md` as the performance ledger:
  - full-run anchor: 7h08m, with OOF training 65.2% and postprocess tuning 24.3%;
  - quick profile: 26.36s RunLogger total, 28.316s cProfile total;
  - current hot paths: model training, `build_kaggle_top_signal_features`,
    `impute_formations`, and `impute_dense_ancc`.
- `make model-bundle` now includes `PROFILING.md`, so performance context is
  sent to external review models too.
- Next: use this document as the baseline for speed deltas before changing model
  count or feature families.

### Postprocess Tuning Speedup

- Replaced brute-force postprocess scoring with an exact basis scorer.
- The selected formula is unchanged:
  `last_known_tvt + alpha * decay(md_from_last_known, tau) *
  ((1 - w_pf) * model_delta + w_pf * pf_delta)`, with optional Savitzky-Golay
  smoothing by well.
- Instead of materializing a full prediction vector for every grid candidate,
  the tuner now builds a linear basis per `(tau, smoothing)` pair and scores
  candidates from cached dot-products.
- Synthetic benchmark, 50k rows, 4,608 candidates:
  - brute force: 53.015s;
  - basis scorer: 0.863s;
  - speedup: 61.5x;
  - best RMSE delta: 1.8e-9.
- Added unit coverage comparing the fast scorer against brute-force
  `apply_postprocess`.
- Expected full-stack impact: the historical 1h44m postprocess phase should drop
  sharply; exact full-run delta is pending.

### Spatial Imputation Speedup

- Vectorized `KaggleTopContext.impute_dense_ancc`:
  - one KD-tree query per chunk;
  - vectorized neighbor selection, weighted mean, weighted std, and nearest
    distance;
  - preserves the old row-loop outputs up to floating noise (`~1e-12` on public
    sample checks).
- Added a vectorized formation-plane path for non-degenerate contexts:
  - batched weighted normal equations;
  - exact row-loop fallback when fewer than 3 valid neighbors are available.
- Important guardrail: `quick.yml` uses `spatial_k=2`, so formation planes are
  underdetermined there and intentionally stay on the exact old fallback. Full
  `stack.yml` uses `spatial_k=10`, so the fast formation path should apply in the
  serious run.
- Added unit coverage comparing vectorized spatial outputs against the previous
  row-loop reference.
- `FEATURE_CACHE_SCHEMA_VERSION` bumped to 6 after the spatial kernel change so
  stale row-loop feature caches do not mask the new implementation.
- Quick profile after postprocess + spatial changes:
  - RunLogger total: 15.95s;
  - cProfile total: 17.644s;
  - `impute_dense_ancc`: 1.880s -> 0.264s;
  - quick CV stayed at 10.04136 after fallback guard.
- Expected full-stack impact: dense ANCC is now much cheaper, and formation
  imputation should speed up materially on `stack.yml`; exact full-run delta is
  pending.

### Model Bundle Skill

- Expanded the local `kaggle-research-brief` skill into a model-ready bundle
  builder.
- Added `COMPETITION.md` as the canonical local competition description.
- Added generated best-public-solution context:
  - `make best-public-solution` writes
    `.kaggle_mining/best_public_solution.md`;
  - selection rule: lowest score-looking value in mined public notebook
    title/slug, with votes as tie-breaker;
  - the document includes source metadata, extracted idea rows, a code inventory,
    a paraphrased implementation shape, and high-claim alternative notebooks.
- New `make model-bundle` target writes `.kaggle_mining/model_bundle.md` with:
  - competition description;
  - `CHANGELOG.md`;
  - mined public code ideas;
  - mined discussion ideas;
  - best open public solution context;
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
