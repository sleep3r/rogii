# TraceBack v0: Local Segment Dictionary / Bag-of-Signals

## Summary

Build a schema-safe diagnostic and candidate-generation experiment inspired by
the late updates in Kaggle discussion `699853`.

The current evidence says global GR/typewell heatmaps are weak:

- raw/global GR matching often does not beat shuffled GR;
- CNN/MTP/SoftSegment can generate plausible paths, but deployable selection is
  weak;
- location/plane/state priors suppress false ridges, but GR itself has not yet
  passed shuffled-GR sanity;
- the useful typewell GR reference may be short and event-like, not a long
  continuous ridge.

TraceBack v0 tests a narrower hypothesis:

> GR may still contain local event anchors: peaks, troughs, short signatures,
> and neighbourhood shapes. These anchors may be enough to generate better
> candidate paths or broad TVT bands, even if global heatmap correlation fails.

This is not a submit model by itself. It is a fold-safe signal audit and
candidate-bank expansion experiment.

## Goals

1. Test whether local GR events beat shuffled GR under a strict true-path audit.
2. Generate new candidate paths/bands from high-confidence event matches.
3. Measure whether the expanded candidate bank reduces `all_candidates_bad`
   wells and improves tail-class oracle.
4. Only if the candidate oracle improves, pass TraceBack candidates into
   `policy_solver` as optional candidates/features.

## Non-Goals

- Do not train another global CNN/Transformer in this experiment.
- Do not use horizontal formation columns or `Geology` as inference inputs.
- Do not optimize a final selector before proving candidate-space gain.
- Do not treat target-derived event matches as deployable predictions.

## Schema-Safe Inputs

Allowed inference inputs:

```text
well_id
id / row_idx / step
MD, X, Y, Z
GR
TVT_input, known mask
typewell TVT/GR
deployable prior paths if already available in test/OOF-safe form
```

Forbidden inference inputs:

```text
TVT hidden target
Geology
ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA
tail_class
oracle labels
target_rmse / best_candidate / regret
```

Training/evaluation may use hidden `TVT` only to compute labels, true-match
metrics, and diagnostic oracle scores after candidate generation.

## Core Idea

Instead of scoring every horizontal step against every typewell bin, extract
local GR events and match those events against a dictionary of known/local/typewell
signatures.

The matching unit is:

```text
hidden event at compressed step i
  -> GR patch around i
  -> match to dictionary patches
  -> candidate TVT anchor(s)
```

Then convert event anchors into full hidden-run candidates using smooth bridge,
drift, and band construction.

## Data Flow

```text
compressed train/test-safe well rows
        |
        v
event extraction
        |
        v
dictionary construction
  - same-well known TVT_input snippets
  - typewell snippets
  - fold-safe train-well snippets
        |
        v
hidden-event matching
  normal GR / shuffled GR / zero GR / shuffled typewell
        |
        v
event anchors and broad bands
        |
        v
traceback candidate paths
        |
        v
candidate-bank oracle + tail audit
```

## Event Extraction

Operate on compressed rows, default `rows_per_step = 32`.

For each well:

1. Interpolate/fill GR exactly as other test-safe pipelines do.
2. Keep finite mask and local finite fraction.
3. Compute robust normalized GR:

```text
gr_z = (GR - median(GR_known_or_all_finite)) / robust_mad
dgr_z = first difference / robust scale
lowpass_gr_z = rolling / Gaussian smoothed
```

4. Extract event candidates:

```text
peaks
troughs
dGR sign changes
high absolute dGR steps
high prominence extrema
local maxima/minima after smoothing
```

Each event row stores:

```text
well_id
step
row_idx_center
MD, X, Y, Z
hidden/known flag
event_type
prominence
gr_value_z
dgr_left/right
finite_frac
patch_radius
GR patch
dGR patch
```

Default patch radii:

```text
[3, 5, 9, 15] compressed steps
```

## Dictionary Construction

Build three dictionary sources.

### A. Same-Well Known Dictionary

Use only non-null `TVT_input` regions from the same well.

For every known event/snippet:

```text
source = same_well_known
source_well_id = current well
source_step
source_TVT = TVT_input[source_step]
source_Z / MD / X / Y
GR patch / dGR patch
event metadata
```

This is deployable because test wells also contain known `TVT_input` outside
hidden intervals.

### B. Typewell Dictionary

Sample typewell GR on a regular TVT grid.

For every typewell bin and local event/snippet:

```text
source = typewell
source_TVT = typewell_tvt_bin
typewell_GR patch
typewell_dGR patch
event metadata
```

This tests whether the vertical typewell contains reusable event shapes.

### C. Fold-Safe Train-Well Dictionary

For OOF validation, each validation fold may use only train-fold wells.

For test inference, this dictionary can use all train wells.

Dictionary entries use known or labelled train paths only for the source TVT,
but validation wells must never appear in their own dictionary.

## Event Match Scoring

For each hidden event and dictionary entry with compatible patch radius:

```text
shape_corr      = robust correlation(GR_patch_hidden, GR_patch_dict)
dgr_corr        = robust correlation(dGR_patch_hidden, dGR_patch_dict)
mad_score       = -robust_mad(GR_patch_hidden - GR_patch_dict)
prom_score      = -abs(prominence_hidden - prominence_dict)
event_type_hit  = 1 if peak/trough/sign matches else 0
finite_weight   = min(finite_frac_hidden, finite_frac_dict)
```

Optional weak priors:

```text
location_score  = -abs(candidate_TVT - broad_bridge_prior) / sigma
z_context_score = compatibility of Z / MD / hidden progress
```

Initial combined score:

```text
score =
    1.00 * shape_corr
  + 0.50 * dgr_corr
  + 0.30 * mad_score
  + 0.20 * event_type_hit
  + 0.15 * prom_score
  + location_weight * location_score
```

The first audit must include:

```text
location_weight = 0.0
location_weight = 0.25
location_weight = 0.50
```

This separates pure GR signal from location-prior rescue.

## Sanity Variants

Every metric must be computed for:

```text
normal_GR
shuffled_hidden_GR
zero_hidden_GR
shuffled_typewell_GR
location_only
```

Rules:

- If `normal_GR ~= shuffled_hidden_GR`, TraceBack does not prove GR signal.
- If `location_only ~= normal_GR`, the result is a location/state prior, not GR
  matching.
- If pure GR fails but location+GR improves oracle candidates, keep it as a
  candidate generator but label it correctly.

## Event-Level Evaluation

Use true hidden `TVT` only for evaluation.

For each hidden event:

```text
true_TVT = TVT at event step
candidate_TVTs = top-K matched dictionary TVTs
```

Metrics:

```text
event_top1_rmse_ft
event_top3_oracle_rmse_ft
event_top10_oracle_rmse_ft
event_true_top1_rate_at_10ft
event_true_top3_rate_at_10ft
event_true_top10_rate_at_10ft
true_score_percentile_mean
normal_vs_shuffled_top10_gap
normal_vs_location_only_gap
```

Also report by:

```text
event_type
patch_radius
finite_frac bucket
hidden progress bucket
tail_class diagnostic group
source dictionary type
```

## Band Generation

Convert top event matches into broad TVT bands.

For each hidden run:

1. Collect high-confidence event anchors.
2. Convert each anchor to `(step, candidate_TVT, confidence)`.
3. Build bands by interpolation and uncertainty expansion:

```text
band_center(step) = weighted linear / monotone interpolation over anchors
band_width(step)  = base_width + uncertainty_from_match_entropy
```

Default candidate bands:

```text
traceback_band_w40
traceback_band_w80
traceback_band_w120
traceback_band_adaptive
```

Band metrics:

```text
coverage@40ft
coverage@80ft
coverage@120ft
mean_band_width
band_center_rmse
normal_vs_shuffled_band_coverage_gap
```

## Candidate Path Generation

Generate full hidden-row paths from event anchors.

Candidate families:

```text
traceback_anchor_linear
traceback_anchor_drift
traceback_anchor_smooth_spline
traceback_anchor_piecewise
traceback_minmax_bridge
traceback_topk_path_0..N
```

Construction rules:

- Respect known `TVT_input` boundaries if available.
- Penalize or discard paths with extreme jumps/slope changes.
- Allow broad shifts: this candidate family is meant to create paths missing
  from current bank.
- Candidate generation itself must not use hidden `TVT`.

## Candidate-Bank Oracle Evaluation

Append TraceBack candidates to the current candidate bank and compute strict
diagnostic oracle.

Metrics:

```text
base bank oracle row_rmse
base + traceback bank oracle row_rmse
all_candidates_bad_wells before/after
G_all_candidates_fail mean RMSE before/after
A_base_b2_level_shift mean RMSE before/after
D_alignment_ambiguity mean RMSE before/after
candidate family win counts
whole-well oracle
chunk oracle
```

Do not train a selector until this shows candidate-space improvement.

## Artifacts

Write to:

```text
artifacts/traceback_v0/
```

Files:

```text
traceback_events.parquet
traceback_dictionary.parquet
traceback_event_matches.parquet
traceback_bands.parquet
traceback_candidates.parquet
traceback_candidate_oracle.csv
traceback_metrics.json
traceback_report.md
figures/event_match_examples.png
figures/normal_vs_shuffled_contact_sheet.png
figures/band_coverage_examples.png
figures/candidate_bank_oracle_delta.png
```

## CLI / Make Targets

Suggested module:

```text
mtpnet/traceback.py
```

Suggested commands:

```bash
python -m mtpnet.traceback \
  --data-dir data/train \
  --output-dir artifacts/traceback_v0 \
  --rows-per-step 32 \
  --patch-radii 3,5,9,15 \
  --n-folds 5 \
  --seed 42
```

Make target:

```bash
make traceback
```

Useful dev target:

```bash
make traceback TRACEBACK_K_WELLS=80 TRACEBACK_PATCH_RADII=3,5,9
```

## Tests

Unit tests:

1. Feature schema guard rejects forbidden columns.
2. Event extractor finds synthetic peak/trough events.
3. Same-well dictionary uses only known `TVT_input` rows.
4. Fold-safe dictionary excludes validation wells.
5. Normal and shuffled variants produce same shapes and comparable row counts.
6. Event match scorer ranks an exact synthetic patch above a shuffled patch.
7. Band generation covers synthetic anchors and expands uncertainty correctly.
8. Candidate generation does not read hidden `TVT`.
9. Candidate oracle is marked diagnostic-only.
10. Smoke run writes metrics/report/parquet/figures.

Verification:

```bash
uv run --extra dev pytest -q tests/test_traceback.py
uv run --extra dev pytest -q
make traceback TRACEBACK_K_WELLS=80
```

## GO / NO-GO Criteria

### Weak GO

```text
normal_GR event top10 beats shuffled by >= 3 percentage points
or true_score_percentile improves by >= 0.03
and band coverage@120ft >= 75%
```

### GO

```text
normal_GR event top10 beats shuffled by >= 5 percentage points
band coverage@120ft >= 80%
candidate-bank oracle reduces all_candidates_bad_wells from 35 to <= 25
G_all_candidates_fail oracle mean improves by >= 1 ft
```

### Strong GO

```text
all_candidates_bad_wells <= 15
G_all_candidates_fail oracle mean improves by >= 2 ft
policy_solver with traceback candidates captures >= 0.10 ft OOF gain
no p95/worst blow-up after guarded policy
```

### NO-GO

```text
normal_GR ~= shuffled_GR
location_only ~= normal_GR
TraceBack candidates do not improve bank oracle
event anchors fail mostly in C_GR_missing_or_noisy and G_all_candidates_fail
```

If NO-GO, close GR/event matching as a main direction and continue with
geometry/state/candidate-bank approaches only.

## Implementation Order

1. Implement event extraction and synthetic tests.
2. Implement same-well known dictionary and pure event matching.
3. Add typewell dictionary.
4. Add fold-safe train-well dictionary.
5. Add sanity variants and event metrics.
6. Add band generation.
7. Add candidate generation and bank oracle.
8. Add report and figures.
9. Only after GO: integrate TraceBack candidates into `policy_solver`.

## Open Design Notes

- Keep location priors as a controlled variable, not a hidden crutch.
- Keep outputs candidate-oriented; do not force a final path too early.
- Report both human-readable plots and strict numeric shuffled sanity.
- Prefer small, inspectable matching functions over another large neural model
  until the event-level signal is proven.
