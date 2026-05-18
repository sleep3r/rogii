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

The visible `test/` folder contains only a few train-like example wells. On
Kaggle rerun it is replaced with the hidden test set.

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
