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
