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

## Data layout

The competition ships `train/`, `test/`, and `sample_submission.csv`. Wells are
identified by 8-character hex hashes. Each well has up to three files:

- `{WELL}__horizontal_well.csv`: the row-level table, one row per ~1 ft along
  the lateral.
- `{WELL}__typewell.csv`: a vertical reference well in the same area indexed
  by TVT.
- `{WELL}.png`: visualization, train-only.

### `__horizontal_well.csv` columns

| Column        | Available on   | Notes                                               |
| ------------- | -------------- | --------------------------------------------------- |
| `MD`          | train + test   | Measured depth along bore, ft. Monotonic.           |
| `X`, `Y`      | train + test   | Horizontal coordinates, ft.                         |
| `Z`           | train + test   | True vertical depth of the bit, ft.                 |
| `GR`          | train + test   | Gamma Ray reading, API. Has NaNs (sensor gaps).     |
| `TVT_input`   | train + test   | Known `TVT` copied as a feature, NaN in hidden zone.|
| `TVT`         | **train only** | Target. NaN-replaced in test by the platform.       |
| `ANCC`        | **train only** | Top of ANCC formation depth, ft.                    |
| `ASTNU`       | **train only** | Top of ASTNU formation depth, ft.                   |
| `ASTNL`       | **train only** | Top of ASTNL formation depth, ft.                   |
| `EGFDU`       | **train only** | Top of EGFDU formation depth, ft.                   |
| `EGFDL`       | **train only** | Top of EGFDL formation depth, ft.                   |
| `BUDA`        | **train only** | Top of BUDA formation depth, ft.                    |

The six formation surfaces are absent in the test horizontal CSVs. **Any
feature that reads them directly from the dataframe will degrade or fail on
test.** Use spatial imputation from train neighbors via
`rogii.spatial.KaggleTopContext.impute_formations` when you need formation
context at test time, or design features that do not depend on these columns.

### `__typewell.csv` columns

| Column     | Available on | Notes                                            |
| ---------- | ------------ | ------------------------------------------------ |
| `TVT`      | train + test | Vertical reference TVT grid, ft.                 |
| `GR`       | train + test | Gamma Ray vs TVT in the reference vertical well. |
| `Geology`  | train + test | Formation label (string).                        |

Typewell is the only data source that links GR shape to TVT directly. Most
classical features in this repo are built around aligning the horizontal GR
trace to typewell GR-vs-TVT.

### Train and test sizes

| Source        | Count                | Notes                                            |
| ------------- | -------------------- | ------------------------------------------------ |
| Train         | `773` wells          | Stays the same locally and on Kaggle.            |
| Test (hidden) | ~`200` wells         | Substituted at submission time on Kaggle.        |
| Test (visible)| `3` example wells    | Local debug only, not used for scoring.          |

Train descriptive stats (computed locally):

- Rows per well: min `2058`, median `6576`, max `12141`. Total `5,092,255` rows.
- Hidden ratio per well: min `19.8%`, median `74.0%`, max `87.5%`. Most wells
  have a long hidden tail and a short known head/tail of `TVT_input`.
- GR NaN ratio per well: min `0.7%`, median `27.7%`, max `80.1%`. GR has
  meaningful gaps; smoothing / interpolation is part of every feature.

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

Metric: **RMSE over hidden test rows**. Lower is better.

Submission format:

```csv
id,tvt
000d7d20_1442,0.0
000d7d20_1443,0.0
```

- The `id` column is `{WELL}_{row_index}` where `row_index` is the position
  of the row inside the horizontal CSV (zero-based on what the platform
  ships, the file order is fixed).
- `sample_submission.csv` lists exactly which rows must be predicted; the
  rows are the hidden subset.

The leaderboard (LB) shows a partial public score on a subset of the hidden
test wells. The final ranking uses the private subset. Both are computed over
the full hidden set across many wells, so per-well variance is averaged out.

### Current leaderboard snapshot (2026-05-20)

```text
1. Jacoby Jaeger       8.239   9 submits
2. Ehimen Nathaniel    8.801  17 submits
3. Virtute             8.947  57 submits
4. kitsune             9.057  53 submits
5. Takahiro Saito      9.102  54 submits
...
us (schema10 isolated): 10.084
```

`baseline = last_known_tvt` (predict `TVT_input` at the last known row for the
whole hidden interval) is the trivial floor. On our local 773-well OOF that
baseline is ~`11.5` ft.

## Code Competition Constraints

Submissions must be made through Kaggle Notebooks (kernels):

- CPU runtime <= 9 hours.
- GPU runtime <= 9 hours.
- Internet disabled at run time.
- External public data and pretrained models are allowed if freely and publicly
  available and packaged into the kernel.
- Output file must be named `submission.csv` and placed at the kernel working
  directory.

Our working strategy is local full training plus inference-only Kaggle submit,
because full CPU training on Kaggle is too slow for the current ensemble.

### What "submit" actually means here

This is a **code competition**, not a CSV upload competition. Concretely:

- You upload a Kaggle kernel (notebook) plus any model artifacts you trained
  locally (as a Kaggle Dataset attached to the kernel).
- Kaggle reruns the kernel on the hidden ~200-well test set with the local
  `test/` directory swapped out for the real hidden one.
- Your `submission.csv` is whatever that rerun produces. CSVs you generated
  locally on the visible debug wells are not, by themselves, a valid
  submission to the real test.
- This codebase already follows that flow: `make submit` builds a kernel
  from the trained model, packages source code, and pushes it through
  `rogii/kaggle_submit.py`. The kernel reads `/kaggle/input/rogii-...` data
  and emits `/kaggle/working/submission.csv`.

### Implications for any new module

- Anything that needs to run inside the Kaggle kernel must work without
  internet, without GPU at inference (unless we mark the kernel GPU), and
  with the train-only columns absent from the test horizontal CSVs.
- Heavy training should happen locally; the kernel is for inference and any
  cheap per-well preprocessing.
- Model artifacts must be packageable as a Kaggle Dataset and the kernel must
  load them deterministically from `/kaggle/input/...`.
