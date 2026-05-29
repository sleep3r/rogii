# Template Envelope v0 Design

## Goal

Build a separate promising-track experiment for the late Kaggle `TW_GR(a * tvt + offset)` idea: generate test-safe TVT corridor candidates by matching many scaled/offset typewell GR templates, then export the envelope midpoint as candidate paths for the existing candidate-bank/chunk-DP stack.

## Scope

This is not a neural model and not another exact-path GR ridge matcher. It is a candidate/corridor generator with strict sanity diagnostics:

- normal hidden GR
- shuffled hidden GR
- zero hidden GR

The first version should answer whether the envelope midpoint/candidates create deployable signal or at least candidate-space oracle headroom.

## Architecture

Create an isolated package:

```text
promising/template_envelope_v0/
    README.md
    config.py
    data.py
    templates.py
    scoring.py
    envelope.py
    cli.py
```

Artifacts go to:

```text
artifacts/template_envelope_v0/
    metrics.json
    report.md
    envelope_candidates.parquet
    step_scores.parquet
```

`envelope_candidates.parquet` must be compatible with existing candidate-bank/chunk-policy external candidate ingestion:

```text
id, well_id, row_idx, step, candidate, pred_tvt
```

No hidden target `TVT`, `Geology`, or formation columns may appear in candidate output.

## Core Algorithm

For each well:

1. Compress rows into steps, keeping partial tails.
2. Build a test-safe coordinate proxy from `TVT_input` bridge.
3. For every `scale a` and `offset`:
   - `path_tvt = center + a * (coord - center) + offset`
   - `template_gr = interp(typewell_GR, path_tvt)`
   - score template against hidden horizontal GR.
4. Select top templates.
5. Build low/high/mid envelope from selected paths.
6. Expand midpoint/low/high candidates back to hidden rows.

## First GO Checks

- normal GR envelope beats shuffled/zero envelope on raw row RMSE or candidate oracle.
- envelope candidates improve candidate-bank oracle.
- no p95/worst blow-up when later used by chunk-DP.

