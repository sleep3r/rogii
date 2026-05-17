# ROGII - Wellbore Geology Prediction

Working repository for the Kaggle competition
`rogii-wellbore-geology-prediction`.

The goal is to predict `TVT` for hidden evaluation intervals in horizontal
wellbores. The repo currently contains a configurable gradient boosting
training pipeline and local Kaggle mining tools for tracking public
notebooks/discussions.

## Quick Start

```bash
make install-deps
make unzip-data
make quick-train
make train
```

The project uses `uv`, `.python-version`, `pyproject.toml`, and `uv.lock`.
Python and Kaggle commands are run through `uv run` by default, which avoids
stale global binaries.

## Setup

Install `uv` if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create or update the environment:

```bash
make install-deps
```

Kaggle auth uses the new access-token flow:

```bash
mkdir -p ~/.kaggle
echo KGAT_xxx > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
```

Check that the project Kaggle CLI is working:

```bash
uv run kaggle --version
uv run kaggle competitions files -c rogii-wellbore-geology-prediction
```

## Data

Download the competition archive:

```bash
make download-data
```

Download and unpack into `data/`:

```bash
make unzip-data
```

`make train` depends on `ensure-data`, so it will unzip the archive if
`data/train` or `data/test` is missing.

## Training

Run the main training config:

```bash
make train
```

Equivalent command:

```bash
uv run python train.py --config configs/hgb.yml
```

For a fast smoke-test on the small public sample:

```bash
make quick-train
```

Training pipeline:

- builds row-level features from trajectory, GR, `TVT_input`, and typewell logs;
- uses a flat TVT baseline as the physical anchor;
- adds public Kaggle top-solution alignment signals;
- trains `HistGradientBoostingRegressor` on residuals;
- validates with grouped CV by well;
- tunes residual blend weight from OOF predictions;
- writes `submission.csv`;
- saves model, feature list, metrics, and config under `artifacts/`.

## Kaggle Public Top Signals

On 2026-05-17, public notebook mining found the strongest current signals in:

- https://www.kaggle.com/code/nihilisticneuralnet/9-251-rogii-wellbore-geology-prediction-dwt-based
- https://www.kaggle.com/code/romantamrazov/rogii-super-solution-lb-top-3
- https://www.kaggle.com/code/ravaghi/wellbore-geology-prediction-hill-climbing
- https://www.kaggle.com/code/ravaghi/wellbore-geology-prediction-lightgbm
- https://www.kaggle.com/code/cdeotte/xgb-starter-cv-15

These notebooks point to the same core idea: this is less a generic tabular
problem and more a GR/typewell alignment problem. The current `train.py`
implements a portable version of that idea:

- beam-style matching of hidden horizontal-well GR against the typewell GR;
- multi-scale normalized cross-correlation anchors;
- low-resolution constrained DTW alignment;
- spatial KNN priors from train formation surfaces (`ANCC`, `ASTNU`, `ASTNL`,
  `EGFDU`, `EGFDL`, `BUDA`);
- dense ANCC spatial prior sampled from train rows;
- consensus features measuring disagreement between alignment signals.

The exact public top notebooks use heavier stacks such as LightGBM, CatBoost,
XGBoost, GPU code, hill climbing, and notebook-specific artifacts. This repo
keeps the implementation local and Kaggle-notebook friendly by feeding the
alignment signals into the existing scikit-learn residual model.

The feature block is controlled by:

```yaml
features:
  include_kaggle_top_signals: true
  kaggle_top:
    beam_configs:
      - [20.0, 144.0, 2, cons]
      - [8.0, 64.0, 2, loose]
      - [14.0, 90.0, 5, sm5]
      - [25.0, 180.0, 2, stiff]
    ncc_windows: [8, 15, 25]
    dtw_enabled: true
```

Latest smoke-test on the public sample:

```text
Flat RMSE: 11.40534
CV RMSE:   10.11490
Best residual_weight: 0.75
Kaggle top-signal features: 34
```

Latest full local train run on visible train wells:

```text
Train rows: 3,783,989
Features:   122
CV RMSE:    16.63554
Flat CV:    19.06126
Best residual_weight: 0.75
Kaggle top-signal features: 38
```

Configs:

- `configs/hgb.yml` - main training run;
- `configs/quick.yml` - small sample smoke-test.

Override the config:

```bash
make train CONFIG=configs/quick.yml
```

## Kaggle Research DB

Two local Codex skills were added for mining competition knowledge into SQLite:

- `kaggle-code-miner` - scans Kaggle notebooks/kernels;
- `kaggle-discussion-miner` - scans Kaggle topics/comments or CSV exports.

Default database:

```text
.kaggle_mining/ideas.sqlite
```

Mine public notebooks:

```bash
uv run python /Users/alexander/.codex/skills/kaggle-code-miner/scripts/mine_kaggle_code.py \
  --competition rogii-wellbore-geology-prediction \
  --db .kaggle_mining/ideas.sqlite \
  --work-dir .kaggle_mining/code \
  --pull-limit 100 \
  --kaggle-cmd "uv run kaggle"
```

Mine discussions:

```bash
uv run python /Users/alexander/.codex/skills/kaggle-discussion-miner/scripts/mine_kaggle_discussions.py \
  --competition rogii-wellbore-geology-prediction \
  --db .kaggle_mining/ideas.sqlite \
  --kaggle-cmd "uv run kaggle"
```

If Kaggle discussion API returns `403/404`, export topics/messages manually and
use the miner's `--topics-csv` fallback.

Useful query:

```bash
sqlite3 .kaggle_mining/ideas.sqlite \
  "select source_type, source_ref, idea_type, summary, score from ideas order by score desc limit 30;"
```

## Submit

This is a Kaggle code competition, so submissions should reference a committed
Kaggle notebook output.

```bash
make submit NOTEBOOK=<NOTEBOOK> VERSION=<VERSION> MESSAGE="Message"
```

Example:

```bash
make submit NOTEBOOK=rogii-hgb VERSION=3 MESSAGE="hgb top-signal model"
```

Expands to:

```bash
uv run kaggle competitions submit \
  -c rogii-wellbore-geology-prediction \
  -f submission.csv \
  -k sleep3r/rogii-hgb \
  -v 3 \
  -m "hgb top-signal model"
```

## Ignored Outputs

The following are intentionally ignored:

- `.venv/`
- `data/`
- `artifacts/`
- `.kaggle_mining/`
- `submission.csv`

## Project Layout

```text
.
├── .python-version
├── Makefile
├── README.md
├── pyproject.toml
├── uv.lock
├── configs/
│   ├── hgb.yml
│   └── quick.yml
├── train.py
└── data/                 # ignored
```
