# ROGII Competition Description

Competition: `rogii-wellbore-geology-prediction`

Kaggle host: ROGII

Start date: May 5, 2026

Final submission deadline: August 5, 2026 at 23:59 UTC

## Goal

Predict `TVT` (True Vertical Thickness, ft) for hidden evaluation intervals in
horizontal wells. The task models the geological position of each 1 ft point
along a lateral wellbore from trajectory, Gamma Ray logs, vertical reference
logs, and formation context.

The practical objective is better geosteering: keep a horizontal well inside
favorable geology, reduce wasted drilling, and support faster automated
interpretation during drilling operations.

## Data

The competition provides `train/` and `test/` directories. Each well is
identified by an 8-character hash and has:

- `{WELLNAME}__horizontal_well.csv`: measured-depth trajectory, coordinates,
  predicted formation depths, Gamma Ray, `TVT_input`, and train-only `TVT`.
- `{WELLNAME}__typewell.csv`: vertical reference log indexed by `TVT`, with
  `GR` and `Geology`.
- train-only `{WELLNAME}.png`: well-path and geological cross-section
  visualization.

Main horizontal fields:

- `MD`: measured depth along the wellbore, ft.
- `X`, `Y`: horizontal coordinates, ft.
- `Z`: true vertical depth, ft.
- `GR`: Gamma Ray, API.
- `TVT_input`: known `TVT` copied as a feature, with NaN in the hidden zone.
- `TVT`: target, hidden in the evaluation zone.
- `ANCC`, `ASTNU`, `ASTNL`, `EGFDU`, `EGFDL`, `BUDA`: train-only predicted
  geological formation surfaces.

### Train and test sizes

- Train: `773` wells with full `TVT` known. Stays the same locally and on
  Kaggle.
- Test: **about 200 hidden wells** when the kernel is rerun on the Kaggle
  evaluation infrastructure.

### CRITICAL: the visible `test/` folder is NOT the real test set

The `test/` directory shipped with the dataset contains only a handful of
train-like example wells. They exist so you can author and dry-run a kernel
locally without hitting NaN columns. The real evaluation happens on a hidden
**~200-well** test set that Kaggle substitutes in at submission time.

Implications, in plain language:

- A `submission.csv` produced from local `data/test/` is not your real
  submission. It is a local-only debug artifact. The real submission is
  whatever your kernel produces when Kaggle re-runs it on the hidden wells.
- Any analysis that assumes "public LB is computed on 3 wells" because
  there are 3 files in `data/test/` is wrong. Public LB is a subset of the
  ~200 hidden wells, not of the visible debug wells.
- "LB grinding" by submitting many slightly different CSVs is not a
  realistic strategy here. The competition is a code competition, not a
  CSV competition, and the metric is computed over a couple hundred wells.
- Robust generalization across many wells beats per-well bespoke tuning.
  GBM-style or feature-engineering strategies that average over many train
  wells should be expected to transfer well; bespoke "solve each test well
  by hand" strategies do not have a sound statistical justification on a
  ~200-well evaluation.

## Evaluation

Metric: RMSE over hidden test rows.

Submission format:

```csv
id,tvt
000d7d20_1442,0.0
000d7d20_1443,0.0
```

The `id` format is `{WELLNAME}_{row_index}`.

## Code Competition Constraints

Submissions must be made through Kaggle Notebooks:

- CPU runtime <= 9 hours.
- GPU runtime <= 9 hours.
- Internet disabled.
- External public data and pretrained models are allowed if freely and publicly
  available.
- Output file must be named `submission.csv`.

Our working strategy is local full training plus inference-only Kaggle submit,
because full CPU training on Kaggle is too slow for the current ensemble.

### What "submit" actually means here

This is a **code competition**, not a CSV upload competition. Concretely:

- You upload a Kaggle kernel (notebook) plus any model artifacts you trained
  locally.
- Kaggle reruns the kernel on the hidden ~200-well test set with the local
  `test/` directory swapped out for the real hidden one.
- Your `submission.csv` is whatever that rerun produces. CSVs you generated
  locally on the visible debug wells are not, by themselves, a valid
  submission to the real test.
- This codebase already follows that flow: `make submit` builds a kernel
  from the trained model, packages source code, and pushes it through
  `rogii/kaggle_submit.py`.
