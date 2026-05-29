# Template Envelope v0

Promising-track implementation of the late Kaggle 699853 idea:

```text
TW_GR(a * tvt + offset) templates
    -> select plausible scaled/offset templates
    -> build min/max TVT envelope
    -> use envelope midpoint as candidate path
```

This is deliberately a **candidate/corridor generator**, not a neural model and not a hard selector.

## Why this exists

Our previous exact-ridge GR matching, MTP, soft-segment and TraceBack experiments showed that raw GR often does not form a reliable single true ridge. The new hypothesis is that the stable object is a broader TVT corridor: find min/max plausible offsets and use the midpoint.

## Schema safety

`envelope_candidates.parquet` contains only:

```text
id, well_id, row_idx, step, candidate, pred_tvt
```

It does not store hidden target `TVT`, `Geology`, or formation columns.

## Run

```bash
make template-envelope
```

Smoke example:

```bash
make template-envelope TEMPLATE_ENVELOPE_OUTPUT=artifacts/template_envelope_v0_smoke TEMPLATE_ENVELOPE_K_WELLS=20
```

Then candidates can be used by the existing external-candidate path:

```bash
make candidate-bank TRACEBACK_CANDIDATES=artifacts/template_envelope_v0/envelope_candidates.parquet
make chunk-policy TRACEBACK_CANDIDATES=artifacts/template_envelope_v0/envelope_candidates.parquet
```

The variable is still named `TRACEBACK_CANDIDATES` in the older infrastructure, but the schema is generic enough for envelope candidates.

