# Changelog and Experiment Log

This file tracks MTPNet framework changes and serious research runs. It is not a
submission log; row-level OOF or test-submit status must be explicit before a
run is treated as a candidate.

## Template

```text
### EXP-YYYYMMDD-N - Short name

- Command/config:
- Data:
- Result:
- What changed:
- Takeaway:
- Next:
```

## 2026-05-23

### PROJECT-PATHFORMER - PathFormer full-well TVT transformer scaffold

- What changed:
  - created standalone `pathformer/` package alongside `mtpnet/`;
  - added `pathformer/__init__.py`, `config.py` (`PathFormerConfig`,
    `N_FEATURES=28`, `load_config()`), `dataset.py` (`WellSample`,
    28-feature engineering, prior dropout augmentation, `WellDataset`,
    `make_tail_balanced_indices()`), `model.py` (`SinusoidalPE`,
    `PathFormer` transformer encoder, `pathformer_loss()` Huber +
    smoothness, `count_parameters()`), `train.py` (tail-balanced sampling,
    AdamW + cosine LR, per-epoch eval, checkpointing), `evaluate.py`
    (per-well RMSE, p50/p90/p95/worst, `gain_vs_b2`, per-tail-class
    breakdown);
  - added `configs/pathformer_v0.yml` (full run: all wells, 20 epochs,
    d_model=192, 4 layers, 8 heads, device=auto) and
    `configs/pathformer_smoke.yml` (20 wells, 1 epoch, d_model=64);
  - added `pf-smoke`, `pf-train`, `pf-eval` Makefile targets with
    `PF_CONFIG` / `PF_EPOCHS` overrides.
- Architecture:
  - one sample = one full well sequence (not a sliding window);
  - bidirectional transformer encoder predicts TVT delta from
    `last_known_tvt` for all hidden steps;
  - 28-feature input: position×4, GR×3, TVT context×2, formation
    boundaries×7, priors×9 (base/B2/A), global×3;
  - target = `TVT − last_known_tvt`; loss computed on hidden steps only;
  - prior dropout aug: `drop_b2=0.30`, `drop_base=0.20`, `drop_a=0.25`,
    `drop_all=0.15` prevents model becoming a B2-smoother;
  - tail-balanced sampler: non-OK wells 2× oversample per epoch.
- Smoke test result:
  - 20 wells, 1 epoch, MPS, ~16 s load + 4 s train;
  - loss decreased, eval printed RMSE `18.39 ft` (random weights, not
    meaningful), checkpoint saved and reloaded clean.
- Motivation:
  - tail audit showed 35 `G_all_candidates_fail` wells where every MTP
    candidate fails; candidate space is the bottleneck, not the selector;
  - PathFormer is designed to generate new path families rather than select
    among existing ones.
- Decision:
  - scaffold GO; smoke test passed;
  - next step: full 20-epoch training run; GO criterion `gain_vs_b2 >= +0.15 ft`.

### AUDIT-TAIL-V1 - Well tail classification and candidate space audit

- Command/config:
  - `make tail-audit` → `artifacts/tail_audit_v1/`;
  - all 155 validation wells, 8 candidates from tracker output;
  - primary candidate: `mtp_track_weighted`.
- Data:
  - wells: `155`;
  - total hidden rows: `758,962`;
  - candidates evaluated: `8`.
- B2 baseline on audit wells:
  - mean well RMSE: `7.905 ft`;
  - p50: `6.823 ft`;
  - p90: `13.816 ft`;
  - p95: `15.437 ft`;
  - worst: `44.064 ft`.
- Tail class distribution:
  - `C_GR_missing_or_noisy`: `42` wells;
  - `G_all_candidates_fail`: `35` wells;
  - `OK_or_mixed`: `41` wells;
  - `A_base_b2_level_shift`: `18` wells;
  - `D_alignment_ambiguity`: `16` wells;
  - `B_long_well_drift`: `3` wells.
- Key diagnostics:
  - `all_candidates_bad_wells`: `35` — all 8 candidates worse than any
    meaningful threshold; no selector can fix these;
  - `b2_bad_and_oracle_good_wells`: `12` — selector can help here;
  - `mtp_improves_wells`: `53`; `mtp_worsens_wells`: `0`.
- Artifacts:
  - `artifacts/tail_audit_v1/well_tail_audit.csv` — per-well class labels;
  - `artifacts/tail_audit_v1/tail_candidate_summary.csv` — per-candidate
    deployable/oracle RMSE by tail class;
  - `artifacts/tail_audit_v1/tail_audit_metrics.json` — summary statistics.
- Takeaway:
  - the `G_all_candidates_fail` class (35 wells, ~23% of the set) is the
    dominant source of LB gap;
  - selector improvement can only address the 12 `b2_bad_and_oracle_good`
    wells; no amount of ranker or tracker tuning closes the G-class gap;
  - LB gap of ~0.80 ft (B2=9.948 → top-10≈9.12) requires a path-generation
    model that produces valid trajectories for wells where no prior is good.
- Decision:
  - MTPNet ranker/tracker family closed as NO-GO for closing the LB gap;
  - pivot to PathFormer: full-well sequence model that generates new path
    families from scratch, not one that selects among existing candidates.

### EXP-MTPNET-V4-CHEAP-RETRY - GeoMTP v4 sim2real cheap retry config

- Command/config:
  - `make train CONFIG=configs/mtp_v4_sim2real_cheap_retry.yml`;
  - `make track RUN_DIR=artifacts/mtp_v4_sim2real_cheap_retry`;
  - init from `mtp_v4_sim2real` checkpoint;
  - corr head: `enabled=True`, `source=input_shallow`, `loss=vertical_kl`,
    `alpha_synth=0.5`, `alpha_real=0.5` (increased from `0.25`);
  - aug: `drop_anchor=0.30`, `drop_b2=0.50`, `drop_all=0.50` (increased
    from `0.25`).
- Window-level result on `valid_base_center_all_hidden`:
  - top1 RMSE: `7.806 ft`;
  - weighted RMSE: `7.653 ft`;
  - synthetic top1 RMSE: `2.988 ft` (improved from `3.252 ft`).
- Track result (corr logits, beta grid `0.25/0.5/1.0`):
  - B2 mean well RMSE: `7.905 ft`;
  - best: `mtp_track_anchored_corr_b025_weighted_a0.1_clip30`,
    mean well RMSE `7.889 ft`, gain `+0.017 ft`;
  - p95 well RMSE: `15.180 ft`.
- Takeaway:
  - synthetic fit improved under higher `alpha_real` and `drop_all`;
  - real-data window top1 also improved slightly vs base v4;
  - corr tracker still trails v3 best gain (`+0.066 ft`) by a large margin;
  - corr head consistently hurts tracker vs raw NN logits.
- Decision:
  - NO-GO; gain `+0.017 ft` is below all thresholds;
  - corr head does not improve path selection on real data in either v4
    variant; closing the MTPNet v4 line.

### EXP-MTPNET-V4-SIM2REAL - GeoMTP v4 sim2real correlation head

- Command/config:
  - `make train CONFIG=configs/mtp_v4_sim2real.yml`;
  - `make track RUN_DIR=artifacts/mtp_v4_sim2real`;
  - init from `mtp_v3_gr_forced` checkpoint;
  - added `CorrelationHeadConfig` and `SyntheticConfig`;
  - corr head: `enabled=True`, `source=input_shallow`,
    `loss=vertical_kl`, `alpha_synth=0.50`, `alpha_real=0.25`,
    `tracker_beta_grid=[0.25, 0.5, 1.0]`,
    `score_normalization=centered`;
  - aug: `drop_anchor=0.30`, `drop_b2=0.50`, `drop_all=0.25`;
  - training extended with synthetic samples (`train_windows=30,707` vs
    `21,495`).
- What changed:
  - added `SyntheticTemplate`, `generate_synthetic_sample()`,
    `generate_synthetic_samples()`, `templates_from_samples()` and GR
    stretch/flip utilities in `mtpnet/synthetic.py`;
  - added `CorrelationHeadConfig` and per-source corr logit path in model;
  - tracker now accepts `logit_source=corr` and runs beta grid sweep,
    writing per-beta `track_metrics_corr_b{025,05,10}.json`;
  - hardened OOF runner with `_selection_score()`, `_path_selection_score()`,
    and `_load_init_checkpoint_checked()`.
- Window-level result on `valid_base_center_all_hidden`:
  - top1 RMSE: `7.884 ft`;
  - weighted RMSE: `7.852 ft`;
  - synthetic top1 RMSE: `3.252 ft`.
- Track result (corr logits, best beta):
  - B2 mean well RMSE: `7.905 ft`;
  - best: `mtp_track_anchored_corr_b10_weighted_a0.1_clip30`,
    mean well RMSE `7.886 ft`, gain `+0.019 ft`;
  - p95 well RMSE: `15.187 ft`.
- Takeaway:
  - corr head improved synthetic fit but degraded real-data window metrics
    vs v3 (top1 `7.884` vs `7.562`);
  - corr tracker gain (`+0.019`) is worse than v3 raw NN tracker (`+0.066`);
  - sim2real signal from corr head does not transfer to path quality on real
    wells.
- Decision:
  - NO-GO; `corr_tracker_gain_vs_nn <= 0` across all beta values;
  - best single-split gain vs B2 across all MTPNet variants: `+0.066 ft`;
  - maximum OOF gain: `+0.037 ft`; all below the `+0.15 ft` GO threshold.

### EXP-MTPNET-V3-GR-FORCED - GeoMTP v3 GR-forced path generation

- Command/config:
  - `make train CONFIG=configs/mtp_v3_gr_forced.yml`;
  - `make track RUN_DIR=artifacts/mtp_v3_gr_forced`;
  - `make oof CONFIG=configs/mtp_v3_gr_forced.yml N_FOLDS=2
    OUTPUT=artifacts/mtp_v3_gr_forced_oof2`;
  - GR-forced path head: model penalized for deviating from GR correlation
    during training;
  - aug: `drop_anchor=0.30`, `drop_b2=0.50`, `drop_all=0.25` (stronger
    than v2).
- Window-level result on `valid_base_center_all_hidden`:
  - top1 RMSE: `7.562 ft` (improved vs v2 `7.826 ft`);
  - weighted RMSE: `7.815 ft`.
- Track result (single split, NN logits):
  - B2 mean well RMSE: `7.905 ft`;
  - best: `mtp_track_anchored_weighted_a0.2_clip30`,
    mean well RMSE `7.839 ft`, gain `+0.066 ft`;
  - p95 well RMSE: `15.080 ft`.
- 2-fold OOF result:
  - OOF B2 RMSE: `10.422 ft`;
  - best OOF candidate: `mtp_track_anchored_weighted_a0.2_clip30`,
    RMSE `10.391 ft`, gain `+0.031 ft`;
  - fold 0: B2 `10.769 ft`, best `10.732 ft`, gain `+0.036 ft`;
  - fold 1: B2 `10.062 ft`, best `10.036 ft`, gain `+0.026 ft`.
- Takeaway:
  - GR-forced training improved window top1 by `0.27 ft` vs v2;
  - single-split gain is the highest of the MTPNet family (`+0.066 ft`);
  - 2-fold OOF gain drops to `+0.031 ft`, confirming single-split
    optimism; robust gain still well below `+0.15 ft` threshold;
  - weighted RMSE regressed (`7.815` vs `7.619`), suggesting mode
    calibration was disturbed by the GR-forcing penalty.
- Decision:
  - NO-GO; OOF gain `+0.031 ft` is below all thresholds;
  - best deployable line from MTPNet is v3 with `+0.066 ft` single-split
    and `+0.031 ft` OOF;
  - root cause is the candidate space (35 `G_all_candidates_fail` wells),
    not selector fidelity.

### INFRA-MTPNET-OOF-RUNNER - Fold-safe MTP OOF training and evaluation

- Command/config:
  - `make oof CONFIG=<config> N_FOLDS=2 OUTPUT=<dir>`;
  - used for `mtp_v2_train_time_selection_oof2` and
    `mtp_v3_gr_forced_oof2`.
- What changed:
  - added `OOFFold` dataclass and `make_oof_folds()` (well-grouped
    stratified splits with configurable seed);
  - added `run_oof()`: trains one checkpoint per fold on the fold's train
    wells, runs the tracker on fold's held-out wells, aggregates across
    folds;
  - added per-fold and aggregate metrics: RMSE, gain vs B2, coverage,
    `worst_well_rmse_max`, `p95_abs_shift_vs_b2_max`;
  - writes `oof_metrics.json`, `oof_report.md`, `oof_candidates.csv`,
    `oof_track_row_predictions.parquet`, and per-fold subdirs;
  - added `oof` CLI and `mtp-oof` Makefile target.
- Decision:
  - infrastructure GO;
  - used to validate that v2 selection (`+0.037 ft`) and v3
    (`+0.031 ft`) OOF gains are consistent across folds and not
    single-split artifacts.

### EXP-MTPNET-V2-SELECTION-LOSS - Train-time top3 / continuation selection loss

- Command/config:
  - `make train CONFIG=configs/mtp_v2_train_time_selection.yml`;
  - `make track RUN_DIR=artifacts/mtp_v2_train_time_selection`;
  - `make oof CONFIG=configs/mtp_v2_train_time_selection.yml N_FOLDS=2
    OUTPUT=artifacts/mtp_v2_train_time_selection_oof2`;
  - aug: `drop_anchor=0.15`, `drop_b2=0.30`, `drop_all=0.10`;
  - logit source: `nn`.
- What changed:
  - added `_top3_margin_loss()`: pushes top-3 ranked modes closer to
    target than lower modes via margin ranking;
  - added `_continuation_probability_loss()`: rewards modes that continue
    smoothly from known history;
  - both losses blended into the main MTP objective.
- Window-level result on `valid_base_center_all_hidden`:
  - top1 RMSE: `7.859 ft`;
  - weighted RMSE: `7.693 ft`.
- Track result (single split, NN logits):
  - B2 mean well RMSE: `7.905 ft`;
  - best: `mtp_track_anchored_weighted_a0.3_clip30`,
    mean well RMSE `7.842 ft`, gain `+0.063 ft`;
  - p95 well RMSE: `14.957 ft`.
- 2-fold OOF result:
  - OOF B2 RMSE: `10.422 ft`;
  - best OOF candidate: `mtp_track_anchored_weighted_a0.2_clip30`,
    RMSE `10.385 ft`, gain `+0.037 ft`;
  - fold 0: B2 `10.769 ft`, best `10.756 ft`, gain `+0.013 ft`;
  - fold 1: B2 `10.062 ft`, best `9.991 ft`, gain `+0.071 ft`.
- Takeaway:
  - selection losses improve single-split tracker gain vs anchor dropout
    alone (`+0.063` vs `+0.047 ft`);
  - fold 1 shows `+0.071 ft` but fold 0 shows only `+0.013 ft` — high
    fold variance indicates model is not robustly better;
  - OOF aggregate `+0.037 ft` is the cleanest estimate and is still well
    below the `+0.15 ft` GO threshold.
- Decision:
  - NO-GO; gain `+0.037 ft` (OOF) is below all thresholds.

### EXP-MTPNET-V2-ANCHOR-DROPOUT - Anchor SDF prior dropout augmentation

- Command/config:
  - `make train CONFIG=configs/mtp_v2_anchor_dropout.yml`;
  - `make track RUN_DIR=artifacts/mtp_v2_anchor_dropout`;
  - aug: `drop_anchor_sdf_prob=0.15`, `drop_b2_sdf_prob=0.30`,
    `drop_all_priors_prob=0.10`;
  - logit source: `ranker_oof` (cross-fit ranker logits reused from v1).
- What changed:
  - added `AugmentationConfig` with `drop_anchor_sdf_prob`,
    `drop_b2_sdf_prob`, `drop_a_density_prob`, `drop_all_priors_prob`,
    `anchor_jitter_ft`, `anchor_swap_prob`;
  - during training, each channel group is independently zeroed with
    its configured probability, preventing the model from relying on any
    single prior;
  - config loader updated to parse `[augmentation]` section.
- Window-level result on `valid_base_center_all_hidden`:
  - top1 RMSE: `7.826 ft`;
  - weighted RMSE: `7.619 ft`.
- Track result (ranker OOF logits, anchored blend):
  - B2 mean well RMSE: `7.905 ft`;
  - best: `mtp_track_anchored_weighted_a0.2_clip20`,
    mean well RMSE `7.859 ft`, gain `+0.047 ft`;
  - p95 well RMSE: `15.155 ft`.
- Takeaway:
  - augmentation marginally improves robustness but the tracker gain is
    still small;
  - window RMSE is nearly identical to v1 (`7.826` vs `7.859 ft`);
  - prior dropout alone does not fix mode quality on hard wells.
- Decision:
  - NO-GO; gain `+0.047 ft` is below the `+0.15 ft` threshold;
  - augmentation framework kept as infrastructure for subsequent configs.

### EXP-MTPRANKER-PAIRWISE-V0 - Query-level CatBoostRanker

- Command/config:
  - `make mtp-ranker-crossfit RUN_DIR=artifacts/mtp_v1_prior_conditioned N_FOLDS=5 OUTPUT=artifacts/mtp_ranker_crossfit_pairwise_v0 RANKER_VARIANT=pairwise`;
  - `make track RUN_DIR=artifacts/mtp_v1_prior_conditioned LOGIT_SOURCE=ranker_oof RANKER_LOGITS=artifacts/mtp_ranker_crossfit_pairwise_v0/oof_ranker_logits.parquet RANKER_BETA={0.25,0.5,1.0}`;
  - query group: `window_id`;
  - objective: CatBoostRanker `YetiRank`;
  - model params: depth `4`, learning rate `0.05`, iterations `1000`,
    `l2_leaf_reg=20`, early stopping `100`.
- What changed:
  - added pair/listwise CatBoost ranker variant for window-level mode ordering;
  - labels are relevance within each window, with lower mode error receiving
    higher relevance;
  - kept the same conservative feature whitelist and the same
    `nn_logit + beta * z(ranker_score)` tracker interface;
  - `rank-crossfit` now accepts `--ranker-variant pairwise`;
  - cross-fit report now includes feature list, model params, per-fold window
    counts, NN/simple-GR/ranker window metrics, and Spearman score-error
    diagnostic.
- Cross-fit ranker OOF metrics:
  - OOF top1 mode RMSE: `7.37252 ft`;
  - best-mode top1 rate: `0.46124`;
  - best-mode top3 rate: `0.91487`;
  - Spearman score-error: `-0.92393`.
- Window-level beta metrics:
  - NN top1/weighted: `7.85912 / 7.44794 ft`;
  - simple GR beta1 top1/weighted: `10.95025 / 8.54118 ft`;
  - pairwise beta `0.25` top1/weighted: `7.79958 / 7.43962 ft`;
  - pairwise beta `0.5` top1/weighted: `7.74097 / 7.43395 ft`;
  - pairwise beta `1.0` top1/weighted: `7.63276 / 7.42751 ft`.
- OOF tracker beta grid:
  - beta `0.25`: best guarded RMSE `9.93541 ft`, gain `+0.01245 ft`;
  - beta `0.5`: best guarded RMSE `9.93329 ft`, gain `+0.01457 ft`;
  - beta `1.0`: best guarded RMSE `9.93514 ft`, gain `+0.01272 ft`;
  - B2 baseline RMSE: `9.94786 ft`.
- Takeaway:
  - pairwise ranking improves window-level top1 more than conservative
    regression;
  - the row-level tracker still does not convert that improvement into a useful
    B2 gain;
  - best pairwise tracker is worse than prior OOF ranker-only `9.92943 ft` and
    conservative beta1 `9.93099 ft`.
- Decision:
  - pairwise ranker is NO-GO as deployable selector;
  - current supervised ranker family does not transfer enough to row-level path
    selection;
  - next path should be training-time improvement or a different selector
    objective, not wider beta/search.
- Verification:
  - pairwise focused tests: `3 passed`;
  - real pairwise cross-fit and beta-grid tracker jobs completed.

### EXP-MTPRANKER-CONSERVATIVE-V0 - Conservative beta-blend selector

- Command/config:
  - `make mtp-ranker-crossfit RUN_DIR=artifacts/mtp_v1_prior_conditioned N_FOLDS=5 OUTPUT=artifacts/mtp_ranker_crossfit_conservative_v0`;
  - `make track RUN_DIR=artifacts/mtp_v1_prior_conditioned LOGIT_SOURCE=ranker_oof RANKER_LOGITS=artifacts/mtp_ranker_crossfit_conservative_v0/oof_ranker_logits.parquet RANKER_BETA={0.25,0.5,1.0}`;
  - CNN was not retrained.
- What changed:
  - replaced the ranker training whitelist with a conservative regression
    feature set;
  - removed high-flex context/categorical features from the CatBoost pool;
  - added explicit feature aliases for NN posterior, GR/NCC, B2/base/A
    distances, path geometry, and compact context;
  - ranker application now supports
    `combined_logit = nn_logit + beta * z(-predicted_error_ft)`;
  - cross-fit OOF parquet now includes combined logits/probabilities for
    beta `0.25`, `0.5`, and `1.0`;
  - tracker accepts `--ranker-beta` / `RANKER_BETA`.
- Cross-fit ranker OOF metrics:
  - mode error MAE: `6.55487 ft`;
  - OOF top1 mode RMSE: `7.50400 ft`;
  - best-mode top1 rate: `0.44770`;
  - best-mode top3 rate: `0.92192`.
- OOF tracker beta grid:
  - beta `0.25`: best guarded RMSE `9.93706 ft`, gain `+0.01080 ft`;
  - beta `0.5`: best guarded RMSE `9.93364 ft`, gain `+0.01422 ft`;
  - beta `1.0`: best guarded RMSE `9.93099 ft`, gain `+0.01687 ft`;
  - B2 baseline RMSE: `9.94786 ft`.
- Takeaway:
  - conservative features slightly improve window-level OOF ranker top1;
  - conservative beta-blend does not improve row-level tracker versus the prior
    OOF ranker-only result `9.92943 ft`;
  - selector overfit risk is lower, but the deployable gain remains too small.
- Decision:
  - keep as a cleaner ranker baseline;
  - performance remains NO-GO for submit by the `+0.10 ft` criterion;
  - next selector work needs a better objective, not a wider beta grid.
- Verification:
  - focused conservative-ranker tests: `3 passed`;
  - ranker/track tests after implementation: `19 passed`;
  - real cross-fit and beta-grid tracker jobs completed.

### EXP-MTPRANKER-CROSSFIT-V0 - OOF ranker logits for tracker

- Command/config:
  - `make mtp-ranker-crossfit RUN_DIR=artifacts/mtp_v1_prior_conditioned N_FOLDS=5 OUTPUT=artifacts/mtp_ranker_crossfit_v0`;
  - `make track RUN_DIR=artifacts/mtp_v1_prior_conditioned LOGIT_SOURCE=ranker_oof RANKER_LOGITS=artifacts/mtp_ranker_crossfit_v0/oof_ranker_logits.parquet`;
  - CNN was not retrained; only existing validation-window MTP modes were used.
- What changed:
  - added 5-fold group cross-fit for the CatBoost mode ranker over the current
    MTP validation wells;
  - every validation well now receives ranker logits from a fold that did not
    train on that well;
  - added `rank-crossfit` CLI, `rank-crossfit`/`mtp-ranker-crossfit` Make
    targets, and external `ranker_oof` tracker logits;
  - cross-fit writes `oof_ranker_logits.parquet`,
    `crossfit_ranker_mode_features.parquet`, `crossfit_ranker_metrics.json`,
    `crossfit_ranker_report.md`, and fold checkpoints.
- Cross-fit ranker OOF metrics:
  - mode error MAE: `6.60133 ft`;
  - OOF top1 mode RMSE: `7.57280 ft`;
  - best-mode top1 rate: `0.44288`;
  - best-mode top3 rate: `0.92415`.
- OOF-ranker tracker result on full validation hidden rows:
  - B2 baseline RMSE: `9.94786 ft`;
  - best guarded candidate `b2_plus_mtp_track_weighted_a0.2_clip20`:
    `9.92943 ft`;
  - gain vs B2: `+0.01843 ft`;
  - covered RMSE: `9.85010 ft`;
  - P95 shift vs B2: `1.08984 ft`;
  - worst well RMSE: `44.22050 ft`;
  - coverage: `98.45%`.
- Takeaway:
  - cross-fit removes the ranker leakage risk identified by the split audit;
  - clean OOF ranker logits keep the tracker mildly positive but far below the
    previous in-sample `+0.12082 ft` gain;
  - bottleneck remains learned mode selection/calibration, not tracker
    infrastructure.
- Decision:
  - tracker plus OOF ranker is a clean diagnostic GO;
  - performance is NO-GO for submit by the `+0.10 ft` gain threshold;
  - next work should improve ranker objective/features or build full fold-safe
    MTP/ranker OOF before packaging.
- Verification:
  - focused cross-fit tests: `3 passed`;
  - full suite before report update: `75 passed`;
  - real cross-fit and tracker jobs completed and wrote their artifacts.

### AUDIT-MTPTRACK-V0 - Ranker split hygiene audit

- Command/config:
  - `make track-audit RUN_DIR=artifacts/mtp_v1_prior_conditioned`;
  - same tracker defaults as `EXP-MTPTRACK-V0`;
  - compares NN logits vs ranker logits on all validation wells, ranker-train
    wells, and ranker-valid wells.
- All validation wells:
  - B2 RMSE: `9.94786 ft`;
  - best NN-logit tracker: `9.92946 ft`, gain `+0.01839`;
  - best ranker-logit tracker: `9.82704 ft`, gain `+0.12082`.
- Ranker-train wells:
  - wells: `101`;
  - rows: `506,703`;
  - B2 RMSE: `10.37686 ft`;
  - best NN-logit tracker: `10.36271 ft`, gain `+0.01415`;
  - best ranker-logit tracker: `10.07314 ft`, gain `+0.30371`;
  - P95 shift vs B2 for best ranker tracker: `5.79297 ft`.
- Ranker-valid wells:
  - wells: `54`;
  - rows: `252,259`;
  - B2 RMSE: `9.02473 ft`;
  - best NN-logit tracker: `8.99640 ft`, gain `+0.02833`;
  - best ranker-logit tracker: `9.01199 ft`, gain `+0.01274`;
  - P95 shift vs B2 for best ranker tracker: `1.15430 ft`;
  - worst well RMSE for best ranker tracker: `22.34533 ft`.
- Takeaway:
  - tracker itself has a small clean positive signal with NN logits;
  - the large all-valid `+0.12082 ft` gain is dominated by ranker in-sample
    wells;
  - current CatBoost ranker is not cleanly validated for submit use.
- Decision:
  - tracker infrastructure remains GO;
  - performance status is downgraded from submit-candidate to leakage-risk
    diagnostic;
  - next required step is fold-safe OOF ranker/tracker or stronger ranker
    validation before any MTPTrack submit packaging.
- Verification:
  - split audit wrote `track_split_audit.json` and `track_split_audit.md`.

### EXP-MTPTRACK-V0 - Sequential multi-realization particle tracker

- Command/config:
  - `make track RUN_DIR=artifacts/mtp_v1_prior_conditioned`;
  - logit source: `ranker`;
  - tau: `5.0 ft`;
  - `n_realizations=32`, `keep_top=32`;
  - merge tolerance: `3.0 ft`;
  - overlap penalty: `0.10`;
  - max modes per window: `8`.
- What changed:
  - added `mtpnet.track` as the first sequential multi-realization tracker;
  - tracker carries top particle realizations well-by-well through ordered MTP
    windows;
  - each window expands existing particles by K trajectory modes, scores by mode
    probability and overlap consistency, merges near-duplicate trajectories, and
    prunes to `keep_top`;
  - added CLI/Make target `track`;
  - writes `track_particles.parquet`, `track_row_predictions.parquet`,
    `track_candidates.csv`, `track_metrics.json`, and `track_report.md`.
- Result on full validation hidden rows:
  - hidden rows: `758,962`;
  - covered rows: `747,218`;
  - coverage: `98.45%`;
  - B2 baseline RMSE: `9.94786 ft`;
  - base schema10 RMSE: `10.47385 ft`;
  - raw `mtp_track_top1`: `9.83750 ft`;
  - raw `mtp_track_weighted`: `9.83993 ft`;
  - best guarded candidate `b2_plus_mtp_track_weighted_a0.3_clip20`:
    `9.82704 ft`;
  - gain vs B2: `+0.12082 ft`;
  - covered RMSE: `9.74524 ft`;
  - P95 shift vs B2: `1.69434 ft`;
  - worst well RMSE: `43.70971 ft`.
- Takeaway:
  - sequential carry-forward tracking now exists and beats B2 by more than the
    `+0.10 ft` threshold on this validation setup;
  - later split audit shows this performance number is ranker-leakage-risk and
    must not be treated as clean submit-ready OOF;
  - raw tracker is already stronger than previous overlap/ranker stitched
    candidates;
  - strong GO threshold `<=9.80 ft` is close but not reached.
- Decision:
  - tracker infrastructure GO;
  - performance GO for `+0.10 ft` blend criterion;
  - next work should tune tracker/ranker jointly, improve coverage beyond
    `98.45%`, and then validate with fold-safe OOF before submit packaging.
- Verification:
  - focused tracker tests: `5 passed`;
  - full suite after implementation: `70 passed`;
  - real tracker job completed and wrote all tracker artifacts.

### EXP-MTPRANKER-V0 - CatBoost mode selector over MTP hypotheses

- Command/config:
  - `make rank RUN_DIR=artifacts/mtp_v1_prior_conditioned`;
  - ranker valid split: `35%` of existing MTP validation wells, grouped by
    `well_id`, seed `42`.
- Data:
  - mode feature rows: `43,136`;
  - windows: `5,392`;
  - modes per window: `8`;
  - ranker train wells: `101`;
  - ranker valid wells: `54`.
- What changed:
  - added CatBoost-based MTP mode ranker as a second-stage selector;
  - added explicit anti-leak `FEATURE_COLUMNS`;
  - added mode-level features from neural logits, path geometry, base/B2/A
    priors, GR/typewell scores, and window context;
  - added ranker CLI/Make target and artifacts:
    `ranker_mode_features.parquet`, `ranker_predictions.parquet`,
    `ranker_candidates.csv`, `ranker_metrics.json`, `ranker_report.md`,
    `checkpoints/mtp_ranker_catboost.cbm`;
  - fixed ranker `window_id` stability so filtered validation windows align
    with feature rows before applying learned logits.
- Window-level ranker-valid result:
  - NN logits top1 RMSE: `7.7140 ft`;
  - NN weighted RMSE: `7.0651 ft`;
  - CatBoost ranker top1 RMSE: `7.4533 ft`;
  - CatBoost ranker weighted RMSE at tau `5`: `6.9245 ft`;
  - CatBoost ranker top3 oracle by logit: `4.2396 ft`;
  - CatBoost ranker best-mode top3 rate: `0.9364`.
- Row-level ranker-valid result:
  - B2 baseline RMSE: `9.0247 ft`;
  - base schema10 RMSE: `9.7790 ft`;
  - best raw ranker candidate `mtp_ranker_top3_t5`: `8.9632 ft`;
  - best guarded B2+ranker candidate:
    `b2_plus_mtp_ranker_weighted_t2.5_a0.3_clip20`: `8.9752 ft`;
  - deployable gain vs B2: `+0.0615 ft` raw top3, `+0.0495 ft` guarded;
  - P95 shift vs B2 for best raw: `3.8770 ft`;
  - P95 shift vs B2 for best guarded: `1.1670 ft`.
- Takeaway:
  - learned selection is materially better than NN logits and simple GR rerank;
  - the planned `+0.10 ft` gain threshold over B2 is not reached yet;
  - ranker is a useful selector direction, but not enough to justify 5-fold OOF
    as-is.
- Decision:
  - diagnostic GO;
  - performance GO is still conditional;
  - next options are better B/NCC features, pairwise/listwise rank objective, or
    OOF ranker only if the selector gain improves past `0.10 ft`.
- Verification:
  - focused ranker tests: `9 passed`;
  - real ranker job completed and wrote all ranker artifacts.

### EXP-MTPNET-V1.2 - Oracle and GR/typewell rerank diagnostics

- Command/config:
  - `make stitch RUN_DIR=artifacts/mtp_v1_prior_conditioned`.
- Data:
  - validation wells: `155`;
  - full hidden rows: `758,962`;
  - covered rows: `747,218`;
  - coverage: `98.45%`;
  - uncovered rows fallback to `b2_guarded_submit`.
- Baselines:
  - `base_schema10_pp` full RMSE: `10.47385`;
  - `b2_guarded_submit` full RMSE: `9.94786`;
  - `base_schema10_pp` covered RMSE: `10.36784`;
  - `b2_guarded_submit` covered RMSE: `9.86897`;
  - uncovered B2 fallback RMSE: `14.08660`.
- Best deployable stitched result:
  - candidate `b2_plus_mtp_gr_rerank_weighted_b2_a0.1_clip30`;
  - full RMSE `9.93236`;
  - covered RMSE `9.85310`;
  - gain vs full B2 `+0.01550`;
  - P95 shift vs B2 `1.17480`;
  - worst well RMSE `43.90626`.
- Oracle diagnostics:
  - `mtp_row_oracle` full RMSE `3.25548`, covered RMSE `2.76514`;
  - `mtp_window_oracle_overlap` full RMSE `4.67178`;
  - `mtp_top3_logit_row_oracle` full RMSE `5.95475`;
  - `b2_plus_mtp_oracle_a0.3_clip20` full RMSE `7.66418`.
- GR/typewell rerank:
  - beta `0.25` window top1 `8.07704`, weighted `7.51246`,
    top3-logit oracle `4.92849`, best-mode top3 rate `0.88984`;
  - stronger betas degrade top1/weighted, so the simple rerank is useful only
    as a weak correction.
- What changed:
  - added row-level oracle upper bounds over all MTP modes and top-3 logit modes;
  - added full-hidden scoring with B2 fallback on uncovered rows;
  - added a simple GR/typewell score and beta grid for MTP mode reranking;
  - updated `stitch_report.md` to separate deployable candidates from oracle
    diagnostics.
- Takeaway:
  - MTP mode space is very strong, so the bottleneck is selection/reranking, not
    candidate generation;
  - the simple GR/typewell rerank gives a small real full-hidden gain over B2,
    but not enough for a confident submit candidate.
- Decision:
  - `MTP v1.2` is a diagnostic GO;
  - not a standalone submit candidate;
  - continue with a learned mode reranker or B/NCC feature stack over MTP modes.
- Artifacts:
  - `artifacts/mtp_v1_prior_conditioned/stitch_report.md`;
  - `artifacts/mtp_v1_prior_conditioned/stitch_metrics.json`;
  - `artifacts/mtp_v1_prior_conditioned/stitch_candidates.csv`.
- Verification:
  - `uv run --extra dev pytest -q` -> `57 passed`;
  - code bundle rebuilt and excludes private research inputs.

### EXP-MTPNET-V1.1 - Row-level stitching

- Command/config:
  - `make stitch RUN_DIR=artifacts/mtp_v1_prior_conditioned`.
- Data:
  - covered hidden rows: `747,218 / 758,962`;
  - coverage: `98.45%`.
- Baselines on covered rows:
  - `base_schema10_pp` RMSE: `10.36784`;
  - `b2_guarded_submit` RMSE: `9.86897`.
- Best deployable covered-row result:
  - `b2_plus_mtp_weighted_overlap_a0.1_clip20`;
  - covered RMSE `9.86424`;
  - gain vs covered B2 `+0.00472`;
  - P95 shift vs B2 `0.50684`.
- Raw MTP stitched candidates:
  - `mtp_weighted_overlap` RMSE `10.1117`;
  - `mtp_top1_overlap` RMSE `10.4461`;
  - `mtp_dp_decode_l1_0.1` RMSE `10.4639`.
- What changed:
  - added `mtpnet.stitch`;
  - added CLI/Make target `stitch`;
  - added overlap aggregation, top3-logit aggregation, confidence fallback,
    simple DP decode, and guarded blends with B2/base.
- Takeaway:
  - stitching does not collapse and coverage is high;
  - raw MTP is not better than B2;
  - B2+MTP gives only a micro-gain, so performance bottleneck is independent
    signal and mode selection.
- Decision:
  - infrastructure GO;
  - not a performance GO;
  - proceed to oracle and rerank diagnostics.
- Verification:
  - `uv run --extra dev pytest -q` -> `52 passed`;
  - `make stitch RUN_DIR=artifacts/mtp_v1_prior_conditioned` -> OK.

### EXP-MTPNET-V1 - Prior-conditioned MTP decoder

- Command/config:
  - `make train CONFIG=configs/mtp_v1_prior_conditioned.yml`;
  - `make eval RUN_DIR=artifacts/mtp_v1_prior_conditioned`.
- Data:
  - train wells: `617`;
  - validation wells: `155`;
  - train windows: `21,495`;
  - valid all-hidden windows: `5,392`.
- Input:
  - wider crop `vertical_radius_ft=240`, `vertical_bins=96`;
  - channels include GR/history core plus `base_sdf`, `b2_sdf`, `a_p50_sdf`,
    `a_density`, `a_p10_p90_band`, `base_offset_value`, `b2_delta_value`;
  - priors loaded from old artifacts:
    `schema10_oof.parquet`, `guarded_predictions.parquet`,
    `oof_candidates.parquet`.
- Loss/model:
  - bounded sigmoid path output;
  - mode bias initialization;
  - soft probability calibration via KL over mode errors;
  - small entropy/diversity regularization.
- Window-level result on `valid_base_center_all_hidden`:
  - top1 RMSE: `7.85912 ft`;
  - weighted RMSE: `7.44795 ft`;
  - oracle top3-by-logit RMSE: `5.11870 ft`;
  - oracle topK RMSE: `4.00520 ft`;
  - best mode top3 rate: `0.87481`;
  - best mode rank mean: `1.95660`;
  - logit-error Spearman: `-0.93620`;
  - target in crop rate: `1.0`;
  - target at edge fraction: `0.0`.
- Sanity:
  - `no_GR` top1 RMSE `7.78778 ft`;
  - `shuffled_GR` top1 RMSE `7.76488 ft`;
  - `no_history` top1 RMSE `7.87554 ft`;
  - `no_base_b2_a` top1 RMSE `9.26777 ft`;
  - `base_b2_a_only` top1 RMSE `7.80084 ft`;
  - static mode oracle topK RMSE `19.19179 ft`.
- Takeaway:
  - MTPNet learned useful local modes and logits;
  - most signal comes from base/B2/A priors rather than GR/history;
  - static baseline is decisively beaten, so the network is not only spreading
    fixed offsets.
- Decision:
  - strong window-level GO;
  - not validated as full hidden OOF until stitching.
- Verification:
  - `uv run --extra dev pytest -q` -> `47 passed`;
  - real-prior sanity on `k_wells=3` passed with 14 channels and `96x24`
    windows;
  - `make smoke` and `make eval RUN_DIR=artifacts/mtp_smoke` passed.

### EXP-MTPNET-V0.2 - Mixed all-hidden windows

- Command/config:
  - `make train CONFIG=configs/mtp_v0_2_mixed.yml`;
  - `make eval RUN_DIR=artifacts/mtp_v0_2_mixed`.
- What changed:
  - added mixed training sample buckets:
    `teacher_forcing_hidden`, `known_tail_start`, `base_center_hidden`;
  - added validation splits:
    `valid_first_chunk_known_tail` and `valid_base_center_all_hidden`;
  - primary validation became all-hidden base-centered windows.
- Result on `valid_base_center_all_hidden`:
  - top1 RMSE: `43.84506 ft`;
  - weighted RMSE: `44.13186 ft`;
  - oracle topK RMSE: `13.64227 ft`;
  - entropy mean: `1.60569`;
  - all 8 modes used.
- Takeaway:
  - v0.2 created real ambiguity and non-collapsed modes;
  - logits/selection were weak;
  - topK coverage did not depend enough on GR/history.
- Decision:
  - infrastructure GO;
  - model predictor/selector NO-GO;
  - move to prior-conditioned v1.
- Verification:
  - `make test` -> `36 passed`;
  - `make smoke`, `make train CONFIG=configs/mtp_v0_2_mixed.yml`,
    and `make eval RUN_DIR=artifacts/mtp_v0_2_mixed` passed.

### EXP-MTPNET-V0.1 - Diversity and bounded output

- What changed:
  - added bounded sigmoid path output;
  - added raw path OOB diagnostics;
  - added mode bias initialization;
  - added CE warmup;
  - added entropy and diversity regularization;
  - kept winner dropout out for this pass.
- Goal:
  - break the v0 single-mode collapse.
- Takeaway:
  - diversity machinery created useful multi-mode behavior;
  - by itself it was insufficient for row-level performance without stronger
    priors.

### EXP-MTPNET-V0 - Test-like geometry and evaluation contract

- What changed:
  - validation no longer uses true hidden history;
  - validation center uses known-tail/base-style anchor rather than true hidden
    TVT;
  - typewell crop uses a regular TVT grid with `vertical_radius_ft`;
  - `crop_tvt` is saved and ft-level metrics are computed;
  - final metrics reload `best.pt` before evaluation;
  - head `BatchNorm1d` was replaced by `LayerNorm`;
  - added geometry report, sanity checks, and parquet output contract.
- Metrics added:
  - `top1_rmse_ft`, `weighted_mean_rmse_ft`,
    `oracle_topk_rmse_ft`, `oracle_top3_rmse_ft`;
  - `target_in_crop_rate`, `target_at_crop_edge_frac`;
  - `classification_accuracy_best_mode`, entropy, mode histogram;
  - prediction OOB diagnostics.
- Takeaway:
  - geometry/eval contract became trustworthy for first-chunk research;
  - pure heatmap first-chunk model worked as smoke but collapsed to one mode.
- Verification:
  - `20 passed` after geometry/eval fixes in the local report;
  - later full suites grew as features were added.

### PROJECT-MTPNET - Standalone research prototype scaffold

- What changed:
  - created standalone `MTPNet` project separate from old `rogii`;
  - added config, IO, heatmap/window builder, CNN-MTP model, MTP loss, train/eval
    CLI, tests, Make targets, and code bundle generator;
  - added `data.k_wells` with `-1` meaning all wells;
  - copied Kaggle-style data into local `MTPNet/data` when needed;
  - ignored `artifacts/` and `data/`;
  - code bundle target excludes private research inputs.
- Initial design:
  - local heatmap windows;
  - Conv/BN/GELU encoder;
  - FC head;
  - K trajectory modes plus logits;
  - MTP best-mode regression plus mode classification and smoothness.
- Decision:
  - GO as research scaffold;
  - OOF/submit candidacy requires row-level stitching and guarded evaluation.
