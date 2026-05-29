# OOF Ablation Diagnostics Design

## Goal

Add a fold-safe OOF diagnostics runner for RAC-Former that compares deterministic baselines, full model predictions, and inference-time head ablations on the same hidden training rows.

## Scope

The first implementation is a post-training diagnostic over existing `fold_*/best.pt` checkpoints. It does not retrain alternative configs. This keeps the tool fast enough to run after each experiment and separates real OOF validation from the local visible test debug submission.

## Variants

For every hidden validation row, the runner records:

- `last_known_tvt`: anchor TVT repeated over the hidden interval.
- `base_no_c0`: `anchor_tvt - (Z_row - Z_anchor)`.
- `base_with_c0`: current dataset baseline from `base_tvt_hidden`.
- `model_full`: current RAC-Former materialization.
- `model_no_direct`: materialization with `direct_resid_step` zeroed.
- `model_no_s_pred`: materialization with `s_pred` zeroed.

## Outputs

The row-level CSV contains `fold`, `well_id`, `id`, `row_idx`, `tvt_true`, all prediction variants, and diagnostic fields. The summary JSON contains pooled RMSE per variant, per-well RMSE rows, and bucketed RMSE tables for `hidden_length`, `GR_valid_frac`, `abs_c0`, `event_count`, and `base_only_rmse`.

## Diagnostics

Per-well diagnostic fields are computed only from OOF validation wells:

- `hidden_length`: number of hidden rows in the well.
- `GR_valid_frac`: mean hidden-step GR valid fraction from feature channel 26.
- `abs_c0`: absolute drift estimate.
- `event_count`: count of train-only hidden `dC_forward` rows whose absolute value exceeds `EVENT_THRESHOLD`.
- `base_only_rmse`: per-well RMSE of `base_with_c0`.

Buckets use quantile-style bins where possible, falling back to stable single-bin summaries for tiny smoke datasets.

## Integration

Expose the runner through `infer.py --oof_diagnostics` and add `make oof-diagnostics`. Keep reusable metric logic in a focused `oof_diagnostics.py` module so it can be unit-tested without loading checkpoints.

## Testing

Use TDD around pure functions for variant materialization, row assembly, pooled/per-well RMSE, and bucket summaries. Add a CLI wiring smoke check through import/argument behavior rather than running full training.
