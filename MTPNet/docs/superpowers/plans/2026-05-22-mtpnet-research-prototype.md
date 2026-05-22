# MTPNet Research Prototype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone MTPNet research prototype that trains and evaluates a CNN/MTP model on local GR heatmap windows, with config-controlled `k_wells` where `-1` means all wells.

**Architecture:** MTPNet is a separate Python project under `/Users/alexander/Desktop/rogii/MTPNet`. It reads Kaggle-style well CSVs, builds compressed local heatmap windows, trains a compact PyTorch CNN with MTP loss, and writes checkpoints plus window-level metrics. The design keeps IO, config, heatmap/window generation, model/loss, and train/eval orchestration in separate modules so fold-safe OOF, Docker, and sequential tracking can be added later.

**Tech Stack:** Python 3.11, numpy, pandas, PyYAML, PyTorch, pyarrow, pytest, Make.

---

## File Structure

- Create `/Users/alexander/Desktop/rogii/MTPNet/pyproject.toml`: package metadata and dependencies.
- Create `/Users/alexander/Desktop/rogii/MTPNet/Makefile`: developer commands for install, copy-data, smoke, train, eval, test.
- Create `/Users/alexander/Desktop/rogii/MTPNet/configs/mtp_smoke.yml`: tiny run config with `k_wells: 3`.
- Create `/Users/alexander/Desktop/rogii/MTPNet/configs/mtp_v0.yml`: full prototype config with `k_wells: -1`.
- Create `/Users/alexander/Desktop/rogii/MTPNet/mtpnet/__init__.py`: package marker.
- Create `/Users/alexander/Desktop/rogii/MTPNet/mtpnet/config.py`: config dataclasses, YAML loading, path resolution, `k_wells` validation.
- Create `/Users/alexander/Desktop/rogii/MTPNet/mtpnet/io.py`: well discovery, CSV loading, optional data copying.
- Create `/Users/alexander/Desktop/rogii/MTPNet/mtpnet/heatmap.py`: GR interpolation, path rasterization, SDF, channel building.
- Create `/Users/alexander/Desktop/rogii/MTPNet/mtpnet/windows.py`: window generation, well-level train/valid split, in-memory dataset.
- Create `/Users/alexander/Desktop/rogii/MTPNet/mtpnet/model.py`: compact CNN-MTP model.
- Create `/Users/alexander/Desktop/rogii/MTPNet/mtpnet/loss.py`: MTP loss and metrics.
- Create `/Users/alexander/Desktop/rogii/MTPNet/mtpnet/train.py`: training loop, checkpointing, predictions, metrics.
- Create `/Users/alexander/Desktop/rogii/MTPNet/mtpnet/eval.py`: load checkpoint and evaluate saved/ rebuilt validation windows.
- Create `/Users/alexander/Desktop/rogii/MTPNet/mtpnet/cli.py`: `copy-data`, `train`, and `eval` subcommands.
- Create tests under `/Users/alexander/Desktop/rogii/MTPNet/tests/`.

## Task 1: Project Scaffold And Config

**Files:**
- Create: `/Users/alexander/Desktop/rogii/MTPNet/pyproject.toml`
- Create: `/Users/alexander/Desktop/rogii/MTPNet/Makefile`
- Create: `/Users/alexander/Desktop/rogii/MTPNet/configs/mtp_smoke.yml`
- Create: `/Users/alexander/Desktop/rogii/MTPNet/configs/mtp_v0.yml`
- Create: `/Users/alexander/Desktop/rogii/MTPNet/mtpnet/__init__.py`
- Create: `/Users/alexander/Desktop/rogii/MTPNet/mtpnet/config.py`
- Test: `/Users/alexander/Desktop/rogii/MTPNet/tests/test_config.py`

- [ ] **Step 1: Write failing config tests**

Create `/Users/alexander/Desktop/rogii/MTPNet/tests/test_config.py`:

```python
from pathlib import Path

import pytest
import yaml

from mtpnet.config import load_config


def write_config(tmp_path: Path, k_wells: int) -> Path:
    path = tmp_path / "config.yml"
    path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "data_dir": "data",
                    "train_dir": "data/train",
                    "test_dir": "data/test",
                    "k_wells": k_wells,
                },
                "run": {"name": "unit"},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_k_wells_minus_one_is_all(tmp_path: Path) -> None:
    cfg = load_config(write_config(tmp_path, -1))
    assert cfg.data.k_wells == -1
    assert cfg.data.use_all_wells is True


def test_positive_k_wells_is_limited(tmp_path: Path) -> None:
    cfg = load_config(write_config(tmp_path, 7))
    assert cfg.data.k_wells == 7
    assert cfg.data.use_all_wells is False


def test_zero_k_wells_fails(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="data.k_wells must be -1 or a positive integer"):
        load_config(write_config(tmp_path, 0))
```

- [ ] **Step 2: Run config tests and verify they fail**

Run from `/Users/alexander/Desktop/rogii/MTPNet`:

```bash
python -m pytest tests/test_config.py -q
```

Expected: FAIL because `mtpnet.config` does not exist.

- [ ] **Step 3: Create package scaffold and config implementation**

Create `/Users/alexander/Desktop/rogii/MTPNet/pyproject.toml`:

```toml
[project]
name = "mtpnet"
version = "0.1.0"
description = "Multi-trajectory heatmap inversion prototype for ROGII wellbore geology prediction."
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
    "pandas>=2.0",
    "pyarrow>=15.0",
    "pyyaml>=6.0",
    "torch>=2.2",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

Create `/Users/alexander/Desktop/rogii/MTPNet/mtpnet/__init__.py`:

```python
"""MTPNet research prototype package."""

__all__ = ["__version__"]
__version__ = "0.1.0"
```

Create `/Users/alexander/Desktop/rogii/MTPNet/mtpnet/config.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    data_dir: Path = Path("data")
    train_dir: Path | None = None
    test_dir: Path | None = None
    k_wells: int = -1
    copy_from: Path | None = None

    @property
    def use_all_wells(self) -> bool:
        return self.k_wells == -1


@dataclass(frozen=True)
class WindowConfig:
    rows_per_step: int = 32
    history_steps: int = 8
    future_steps: int = 16
    vertical_bins: int = 64
    vertical_radius_ft: float = 160.0
    stride_steps: int = 4
    max_windows_per_well: int = 64
    channels: tuple[str, ...] = (
        "gr_diff",
        "abs_gr_diff",
        "history_mask",
        "history_sdf",
        "finite_mask",
    )


@dataclass(frozen=True)
class ModelConfig:
    k_modes: int = 8
    conv_channels: tuple[int, ...] = (16, 32, 64, 128)
    hidden_dims: tuple[int, ...] = (512, 1024)
    dropout: float = 0.05


@dataclass(frozen=True)
class LossConfig:
    path_loss: str = "mae"
    alpha_cls: float = 0.2
    smooth_lambda: float = 0.01


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 64
    epochs: int = 8
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    device: str = "auto"
    num_workers: int = 0
    seed: int = 42


@dataclass(frozen=True)
class ValidationConfig:
    valid_fraction: float = 0.2
    seed: int = 42


@dataclass(frozen=True)
class RunConfig:
    name: str = "mtp_v0"
    output_dir: Path = Path("artifacts/mtp_v0")


@dataclass(frozen=True)
class MTPConfig:
    data: DataConfig = field(default_factory=DataConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    run: RunConfig = field(default_factory=RunConfig)


def _as_path(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    return Path(str(value))


def _tuple(value: Any, default: tuple[Any, ...]) -> tuple[Any, ...]:
    if value is None:
        return default
    return tuple(value)


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} config section must be a mapping")
    return value


def load_config(path: str | Path) -> MTPConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    data_raw = _section(raw, "data")
    k_wells = int(data_raw.get("k_wells", -1))
    if k_wells == 0 or k_wells < -1:
        raise ValueError("data.k_wells must be -1 or a positive integer")
    data = DataConfig(
        data_dir=Path(data_raw.get("data_dir", "data")),
        train_dir=_as_path(data_raw.get("train_dir")),
        test_dir=_as_path(data_raw.get("test_dir")),
        k_wells=k_wells,
        copy_from=_as_path(data_raw.get("copy_from")),
    )

    window_raw = _section(raw, "window")
    default_window = WindowConfig()
    window = WindowConfig(
        rows_per_step=int(window_raw.get("rows_per_step", default_window.rows_per_step)),
        history_steps=int(window_raw.get("history_steps", default_window.history_steps)),
        future_steps=int(window_raw.get("future_steps", default_window.future_steps)),
        vertical_bins=int(window_raw.get("vertical_bins", default_window.vertical_bins)),
        vertical_radius_ft=float(window_raw.get("vertical_radius_ft", default_window.vertical_radius_ft)),
        stride_steps=int(window_raw.get("stride_steps", default_window.stride_steps)),
        max_windows_per_well=int(window_raw.get("max_windows_per_well", default_window.max_windows_per_well)),
        channels=tuple(window_raw.get("channels", default_window.channels)),
    )

    model_raw = _section(raw, "model")
    default_model = ModelConfig()
    model = ModelConfig(
        k_modes=int(model_raw.get("k_modes", default_model.k_modes)),
        conv_channels=tuple(int(v) for v in _tuple(model_raw.get("conv_channels"), default_model.conv_channels)),
        hidden_dims=tuple(int(v) for v in _tuple(model_raw.get("hidden_dims"), default_model.hidden_dims)),
        dropout=float(model_raw.get("dropout", default_model.dropout)),
    )

    loss_raw = _section(raw, "loss")
    default_loss = LossConfig()
    loss = LossConfig(
        path_loss=str(loss_raw.get("path_loss", default_loss.path_loss)),
        alpha_cls=float(loss_raw.get("alpha_cls", default_loss.alpha_cls)),
        smooth_lambda=float(loss_raw.get("smooth_lambda", default_loss.smooth_lambda)),
    )

    train_raw = _section(raw, "train")
    default_train = TrainConfig()
    train = TrainConfig(
        batch_size=int(train_raw.get("batch_size", default_train.batch_size)),
        epochs=int(train_raw.get("epochs", default_train.epochs)),
        learning_rate=float(train_raw.get("learning_rate", default_train.learning_rate)),
        weight_decay=float(train_raw.get("weight_decay", default_train.weight_decay)),
        device=str(train_raw.get("device", default_train.device)),
        num_workers=int(train_raw.get("num_workers", default_train.num_workers)),
        seed=int(train_raw.get("seed", default_train.seed)),
    )

    validation_raw = _section(raw, "validation")
    default_validation = ValidationConfig()
    validation = ValidationConfig(
        valid_fraction=float(validation_raw.get("valid_fraction", default_validation.valid_fraction)),
        seed=int(validation_raw.get("seed", default_validation.seed)),
    )

    run_raw = _section(raw, "run")
    default_run = RunConfig()
    run = RunConfig(
        name=str(run_raw.get("name", default_run.name)),
        output_dir=Path(run_raw.get("output_dir", default_run.output_dir)),
    )

    return MTPConfig(
        data=data,
        window=window,
        model=model,
        loss=loss,
        train=train,
        validation=validation,
        run=run,
    )
```

Create `/Users/alexander/Desktop/rogii/MTPNet/configs/mtp_smoke.yml`:

```yaml
data:
  data_dir: data
  train_dir: data/train
  test_dir: data/test
  k_wells: 3
  copy_from: ../old/data

window:
  rows_per_step: 32
  history_steps: 8
  future_steps: 16
  vertical_bins: 64
  vertical_radius_ft: 160.0
  stride_steps: 8
  max_windows_per_well: 12
  channels: [gr_diff, abs_gr_diff, history_mask, history_sdf, finite_mask]

model:
  k_modes: 4
  conv_channels: [8, 16, 32]
  hidden_dims: [128, 256]
  dropout: 0.05

loss:
  path_loss: mae
  alpha_cls: 0.2
  smooth_lambda: 0.01

train:
  batch_size: 16
  epochs: 2
  learning_rate: 0.001
  weight_decay: 0.0001
  device: auto
  num_workers: 0
  seed: 42

validation:
  valid_fraction: 0.34
  seed: 42

run:
  name: mtp_smoke
  output_dir: artifacts/mtp_smoke
```

Create `/Users/alexander/Desktop/rogii/MTPNet/configs/mtp_v0.yml`:

```yaml
data:
  data_dir: data
  train_dir: data/train
  test_dir: data/test
  k_wells: -1
  copy_from: ../old/data

window:
  rows_per_step: 32
  history_steps: 8
  future_steps: 16
  vertical_bins: 64
  vertical_radius_ft: 160.0
  stride_steps: 4
  max_windows_per_well: 64
  channels: [gr_diff, abs_gr_diff, history_mask, history_sdf, finite_mask]

model:
  k_modes: 8
  conv_channels: [16, 32, 64, 128]
  hidden_dims: [512, 1024]
  dropout: 0.05

loss:
  path_loss: mae
  alpha_cls: 0.2
  smooth_lambda: 0.01

train:
  batch_size: 64
  epochs: 8
  learning_rate: 0.001
  weight_decay: 0.0001
  device: auto
  num_workers: 0
  seed: 42

validation:
  valid_fraction: 0.2
  seed: 42

run:
  name: mtp_v0
  output_dir: artifacts/mtp_v0
```

Create `/Users/alexander/Desktop/rogii/MTPNet/Makefile`:

```makefile
SHELL := /bin/bash
PYTHON ?= python
CONFIG ?= configs/mtp_v0.yml
SMOKE_CONFIG ?= configs/mtp_smoke.yml
RUN_DIR ?= artifacts/mtp_v0
COPY_SOURCE ?= ../old/data
COPY_TARGET ?= data

.PHONY: install test copy-data smoke train eval

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest -q

copy-data:
	$(PYTHON) -m mtpnet.cli copy-data --source $(COPY_SOURCE) --target $(COPY_TARGET)

smoke:
	$(PYTHON) -m mtpnet.cli train --config $(SMOKE_CONFIG)

train:
	$(PYTHON) -m mtpnet.cli train --config $(CONFIG)

eval:
	$(PYTHON) -m mtpnet.cli eval --run-dir $(RUN_DIR)
```

- [ ] **Step 4: Run config tests and verify they pass**

Run:

```bash
python -m pytest tests/test_config.py -q
```

Expected: `3 passed`.

- [ ] **Step 5: Commit scaffold and config**

Run:

```bash
git add pyproject.toml Makefile configs/mtp_smoke.yml configs/mtp_v0.yml mtpnet/__init__.py mtpnet/config.py tests/test_config.py
git commit -m "feat: scaffold MTPNet config"
```

## Task 2: Well IO And Data Copy Command

**Files:**
- Create: `/Users/alexander/Desktop/rogii/MTPNet/mtpnet/io.py`
- Create: `/Users/alexander/Desktop/rogii/MTPNet/mtpnet/cli.py`
- Test: `/Users/alexander/Desktop/rogii/MTPNet/tests/test_io.py`

- [ ] **Step 1: Write failing IO tests**

Create `/Users/alexander/Desktop/rogii/MTPNet/tests/test_io.py`:

```python
from pathlib import Path

import pandas as pd
import pytest

from mtpnet.config import DataConfig
from mtpnet.io import copy_data_tree, discover_wells


def make_well(root: Path, well_id: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": [f"{well_id}_0"], "TVT": [1.0], "TVT_input": [1.0], "GR": [10.0]}).to_csv(
        root / f"{well_id}__horizontal_well.csv", index=False
    )
    pd.DataFrame({"TVT": [1.0], "GR": [10.0]}).to_csv(
        root / f"{well_id}__typewell.csv", index=False
    )


def test_discover_wells_respects_positive_k(tmp_path: Path) -> None:
    train = tmp_path / "data" / "train"
    make_well(train, "b_well")
    make_well(train, "a_well")
    wells = discover_wells(DataConfig(data_dir=tmp_path / "data", train_dir=train, k_wells=1))
    assert [well.well_id for well in wells] == ["a_well"]


def test_discover_wells_minus_one_uses_all(tmp_path: Path) -> None:
    train = tmp_path / "data" / "train"
    make_well(train, "b_well")
    make_well(train, "a_well")
    wells = discover_wells(DataConfig(data_dir=tmp_path / "data", train_dir=train, k_wells=-1))
    assert [well.well_id for well in wells] == ["a_well", "b_well"]


def test_missing_typewell_fails(tmp_path: Path) -> None:
    train = tmp_path / "data" / "train"
    train.mkdir(parents=True)
    pd.DataFrame({"TVT": [1.0], "TVT_input": [1.0], "GR": [10.0]}).to_csv(
        train / "lonely__horizontal_well.csv", index=False
    )
    with pytest.raises(FileNotFoundError, match="Missing typewell"):
        discover_wells(DataConfig(data_dir=tmp_path / "data", train_dir=train, k_wells=-1))


def test_copy_data_tree_copies_known_subdirs(tmp_path: Path) -> None:
    source = tmp_path / "old_data"
    make_well(source / "train", "well")
    target = tmp_path / "new_data"
    copy_data_tree(source, target)
    assert (target / "train" / "well__horizontal_well.csv").exists()
    assert (target / "train" / "well__typewell.csv").exists()
```

- [ ] **Step 2: Run IO tests and verify they fail**

Run:

```bash
python -m pytest tests/test_io.py -q
```

Expected: FAIL because `mtpnet.io` does not exist.

- [ ] **Step 3: Implement well discovery and copy-data**

Create `/Users/alexander/Desktop/rogii/MTPNet/mtpnet/io.py`:

```python
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import DataConfig


@dataclass(frozen=True)
class WellPaths:
    well_id: str
    horizontal_path: Path
    typewell_path: Path


def well_id_from_horizontal(path: Path) -> str:
    suffix = "__horizontal_well.csv"
    if not path.name.endswith(suffix):
        raise ValueError(f"Not a horizontal well file: {path}")
    return path.name[: -len(suffix)]


def resolve_train_dir(config: DataConfig) -> Path:
    candidates = []
    if config.train_dir is not None:
        candidates.append(config.train_dir)
    candidates.append(config.data_dir / "train")
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No train directory found. Checked: {', '.join(str(p) for p in candidates)}")


def discover_wells(config: DataConfig) -> list[WellPaths]:
    train_dir = resolve_train_dir(config)
    horizontal_paths = sorted(train_dir.glob("*__horizontal_well.csv"), key=well_id_from_horizontal)
    if not horizontal_paths:
        raise FileNotFoundError(f"No horizontal well files found in {train_dir}")
    if config.k_wells > 0:
        horizontal_paths = horizontal_paths[: config.k_wells]

    wells: list[WellPaths] = []
    for horizontal_path in horizontal_paths:
        well_id = well_id_from_horizontal(horizontal_path)
        typewell_path = horizontal_path.with_name(f"{well_id}__typewell.csv")
        if not typewell_path.exists():
            raise FileNotFoundError(f"Missing typewell for {well_id}: {typewell_path}")
        wells.append(WellPaths(well_id=well_id, horizontal_path=horizontal_path, typewell_path=typewell_path))
    return wells


def load_well(well: WellPaths) -> tuple[pd.DataFrame, pd.DataFrame]:
    horizontal = pd.read_csv(well.horizontal_path)
    typewell = pd.read_csv(well.typewell_path)
    required_horizontal = {"TVT", "TVT_input", "GR"}
    required_typewell = {"TVT", "GR"}
    missing_horizontal = required_horizontal.difference(horizontal.columns)
    missing_typewell = required_typewell.difference(typewell.columns)
    if missing_horizontal:
        raise ValueError(f"{well.horizontal_path} missing columns: {sorted(missing_horizontal)}")
    if missing_typewell:
        raise ValueError(f"{well.typewell_path} missing columns: {sorted(missing_typewell)}")
    return horizontal, typewell


def copy_data_tree(source: str | Path, target: str | Path) -> None:
    source_path = Path(source)
    target_path = Path(target)
    if not source_path.exists():
        raise FileNotFoundError(f"Data source does not exist: {source_path}")
    target_path.mkdir(parents=True, exist_ok=True)
    for name in ("train", "test", "public_train", "public_test"):
        src = source_path / name
        if src.exists():
            dst = target_path / name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
    for filename in ("sample_submission.csv",):
        src = source_path / filename
        if src.exists():
            shutil.copy2(src, target_path / filename)
```

Create `/Users/alexander/Desktop/rogii/MTPNet/mtpnet/cli.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path

from .io import copy_data_tree


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MTPNet research prototype CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    copy_parser = sub.add_parser("copy-data", help="Copy Kaggle-style data into this project")
    copy_parser.add_argument("--source", type=Path, required=True)
    copy_parser.add_argument("--target", type=Path, required=True)

    train_parser = sub.add_parser("train", help="Train an MTPNet run")
    train_parser.add_argument("--config", type=Path, required=True)

    eval_parser = sub.add_parser("eval", help="Evaluate a trained MTPNet run")
    eval_parser.add_argument("--run-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "copy-data":
        copy_data_tree(args.source, args.target)
        print(f"Copied data from {args.source} to {args.target}", flush=True)
        return
    if args.command == "train":
        from .train import train_from_config

        train_from_config(args.config)
        return
    if args.command == "eval":
        from .eval import evaluate_run

        evaluate_run(args.run_dir)
        return
    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run IO tests and verify they pass**

Run:

```bash
python -m pytest tests/test_io.py -q
```

Expected: `4 passed`.

- [ ] **Step 5: Commit IO and CLI copy command**

Run:

```bash
git add mtpnet/io.py mtpnet/cli.py tests/test_io.py
git commit -m "feat: add MTPNet well IO"
```

## Task 3: Heatmap Channels And Window Dataset

**Files:**
- Create: `/Users/alexander/Desktop/rogii/MTPNet/mtpnet/heatmap.py`
- Create: `/Users/alexander/Desktop/rogii/MTPNet/mtpnet/windows.py`
- Test: `/Users/alexander/Desktop/rogii/MTPNet/tests/test_windows.py`

- [ ] **Step 1: Write failing heatmap/window tests**

Create `/Users/alexander/Desktop/rogii/MTPNet/tests/test_windows.py`:

```python
import numpy as np
import pandas as pd

from mtpnet.config import WindowConfig
from mtpnet.windows import build_windows_for_well, split_wells


def synthetic_horizontal(n: int = 160) -> pd.DataFrame:
    tvt = np.linspace(1000.0, 1040.0, n)
    gr = 80.0 + np.sin(np.linspace(0.0, 8.0, n)) * 20.0
    tvt_input = tvt.copy()
    tvt_input[n // 2 :] = np.nan
    return pd.DataFrame({"id": [f"row_{i}" for i in range(n)], "TVT": tvt, "TVT_input": tvt_input, "GR": gr})


def synthetic_typewell(n: int = 256) -> pd.DataFrame:
    tvt = np.linspace(960.0, 1080.0, n)
    gr = 80.0 + np.sin(np.linspace(0.0, 12.0, n)) * 20.0
    return pd.DataFrame({"TVT": tvt, "GR": gr})


def test_build_windows_shape_and_metadata() -> None:
    cfg = WindowConfig(
        rows_per_step=4,
        history_steps=4,
        future_steps=6,
        vertical_bins=32,
        vertical_radius_ft=60.0,
        stride_steps=2,
        max_windows_per_well=3,
    )
    windows = build_windows_for_well("well_a", synthetic_horizontal(), synthetic_typewell(), cfg)
    assert len(windows) > 0
    sample = windows[0]
    assert sample.x.shape == (5, 32, 10)
    assert sample.target_bins.shape == (6,)
    assert sample.target_tvt.shape == (6,)
    assert sample.well_id == "well_a"
    assert np.isfinite(sample.x).all()


def test_split_wells_keeps_well_boundaries() -> None:
    train, valid = split_wells(["a", "b", "c", "d", "e"], valid_fraction=0.4, seed=7)
    assert set(train).isdisjoint(valid)
    assert sorted(train + valid) == ["a", "b", "c", "d", "e"]
    assert len(valid) == 2
```

- [ ] **Step 2: Run window tests and verify they fail**

Run:

```bash
python -m pytest tests/test_windows.py -q
```

Expected: FAIL because `mtpnet.windows` does not exist.

- [ ] **Step 3: Implement heatmap helpers**

Create `/Users/alexander/Desktop/rogii/MTPNet/mtpnet/heatmap.py`:

```python
from __future__ import annotations

import numpy as np


KNOWN_CHANNELS = {"gr_diff", "abs_gr_diff", "history_mask", "history_sdf", "finite_mask"}


def fill_nan(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(arr)
    if finite.all():
        return arr.astype(np.float32), finite.astype(np.float32)
    if not finite.any():
        return np.zeros_like(arr, dtype=np.float32), finite.astype(np.float32)
    idx = np.arange(len(arr))
    filled = np.interp(idx, idx[finite], arr[finite]).astype(np.float32)
    return filled, finite.astype(np.float32)


def rasterize_path(path_bins: np.ndarray, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.float32)
    path = np.asarray(path_bins, dtype=np.float32)
    for x, y in enumerate(path[:width]):
        if not np.isfinite(y):
            continue
        y0 = int(np.clip(round(float(y)), 0, height - 1))
        mask[y0, x] = 1.0
        if y0 > 0:
            mask[y0 - 1, x] = 0.5
        if y0 + 1 < height:
            mask[y0 + 1, x] = 0.5
    return mask


def path_sdf(path_bins: np.ndarray, height: int, width: int) -> np.ndarray:
    yy = np.arange(height, dtype=np.float32)[:, None]
    out = np.zeros((height, width), dtype=np.float32)
    path = np.asarray(path_bins, dtype=np.float32)
    for x in range(width):
        if x < len(path) and np.isfinite(path[x]):
            out[:, x] = (yy[:, 0] - path[x]) / max(1.0, float(height))
    return out


def build_channels(
    horizontal_gr: np.ndarray,
    typewell_gr: np.ndarray,
    history_bins: np.ndarray,
    finite_steps: np.ndarray,
    channels: tuple[str, ...],
) -> np.ndarray:
    unknown = sorted(set(channels).difference(KNOWN_CHANNELS))
    if unknown:
        raise ValueError(f"Unknown input channels: {unknown}")
    h = np.asarray(horizontal_gr, dtype=np.float32)
    t = np.asarray(typewell_gr, dtype=np.float32)
    heatmap = h[None, :] - t[:, None]
    height, width = heatmap.shape
    history_mask = rasterize_path(history_bins, height=height, width=width)
    sdf = path_sdf(history_bins, height=height, width=width)
    finite = np.broadcast_to(np.asarray(finite_steps, dtype=np.float32)[None, :], (height, width))
    values = {
        "gr_diff": heatmap / 100.0,
        "abs_gr_diff": np.abs(heatmap) / 100.0,
        "history_mask": history_mask,
        "history_sdf": sdf,
        "finite_mask": finite,
    }
    return np.stack([values[name] for name in channels]).astype(np.float32)
```

- [ ] **Step 4: Implement window generation**

Create `/Users/alexander/Desktop/rogii/MTPNet/mtpnet/windows.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import WindowConfig
from .heatmap import build_channels, fill_nan


@dataclass(frozen=True)
class WindowSample:
    x: np.ndarray
    target_bins: np.ndarray
    target_tvt: np.ndarray
    well_id: str
    start_step: int
    center_tvt: float


def _compress(values: np.ndarray, rows_per_step: int) -> np.ndarray:
    usable = (len(values) // rows_per_step) * rows_per_step
    if usable <= 0:
        return np.empty(0, dtype=np.float32)
    return np.asarray(values[:usable], dtype=np.float32).reshape(-1, rows_per_step).mean(axis=1)


def _target_bins(target_tvt: np.ndarray, typewell_tvt_crop: np.ndarray) -> np.ndarray:
    return np.abs(typewell_tvt_crop[:, None] - target_tvt[None, :]).argmin(axis=0).astype(np.float32)


def _crop_typewell(typewell: pd.DataFrame, center_tvt: float, cfg: WindowConfig) -> tuple[np.ndarray, np.ndarray]:
    tvt = pd.to_numeric(typewell["TVT"], errors="coerce").to_numpy(dtype=np.float32)
    gr_raw = pd.to_numeric(typewell["GR"], errors="coerce").to_numpy(dtype=np.float32)
    gr, _ = fill_nan(gr_raw)
    order = np.argsort(np.abs(tvt - center_tvt))[: cfg.vertical_bins]
    order = np.sort(order)
    crop_tvt = tvt[order]
    crop_gr = gr[order]
    if len(crop_tvt) < cfg.vertical_bins:
        pad = cfg.vertical_bins - len(crop_tvt)
        crop_tvt = np.pad(crop_tvt, (0, pad), mode="edge")
        crop_gr = np.pad(crop_gr, (0, pad), mode="edge")
    return crop_tvt.astype(np.float32), crop_gr.astype(np.float32)


def build_windows_for_well(
    well_id: str,
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    cfg: WindowConfig,
) -> list[WindowSample]:
    total_steps = cfg.history_steps + cfg.future_steps
    tvt = pd.to_numeric(horizontal["TVT"], errors="coerce").to_numpy(dtype=np.float32)
    tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(dtype=np.float32)
    gr_raw = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(dtype=np.float32)
    gr, finite = fill_nan(gr_raw)

    comp_tvt = _compress(tvt, cfg.rows_per_step)
    comp_tvt_input = _compress(tvt_input, cfg.rows_per_step)
    comp_gr = _compress(gr, cfg.rows_per_step)
    comp_finite = _compress(finite, cfg.rows_per_step)
    if len(comp_tvt) < total_steps:
        return []

    hidden = ~np.isfinite(comp_tvt_input)
    hidden_steps = np.flatnonzero(hidden)
    if len(hidden_steps) == 0:
        return []
    first_hidden = int(hidden_steps[0])
    min_start = max(0, first_hidden - cfg.history_steps)
    max_start = len(comp_tvt) - total_steps
    starts = list(range(min_start, max_start + 1, max(1, cfg.stride_steps)))
    windows: list[WindowSample] = []
    for start in starts:
        hist_slice = slice(start, start + cfg.history_steps)
        fut_slice = slice(start + cfg.history_steps, start + total_steps)
        if not np.isfinite(comp_tvt[hist_slice]).all() or not np.isfinite(comp_tvt[fut_slice]).all():
            continue
        center_tvt = float(comp_tvt[start + cfg.history_steps - 1])
        crop_tvt, crop_gr = _crop_typewell(typewell, center_tvt, cfg)
        path_all = _target_bins(comp_tvt[start : start + total_steps], crop_tvt)
        history_bins = np.full(total_steps, np.nan, dtype=np.float32)
        history_bins[: cfg.history_steps] = path_all[: cfg.history_steps]
        x = build_channels(
            horizontal_gr=comp_gr[start : start + total_steps],
            typewell_gr=crop_gr,
            history_bins=history_bins,
            finite_steps=comp_finite[start : start + total_steps],
            channels=cfg.channels,
        )
        windows.append(
            WindowSample(
                x=x,
                target_bins=path_all[cfg.history_steps :].astype(np.float32),
                target_tvt=comp_tvt[fut_slice].astype(np.float32),
                well_id=well_id,
                start_step=start,
                center_tvt=center_tvt,
            )
        )
        if len(windows) >= cfg.max_windows_per_well:
            break
    return windows


def split_wells(well_ids: list[str], valid_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    ordered = sorted(well_ids)
    if len(ordered) < 2:
        return ordered, []
    rng = np.random.default_rng(seed)
    shuffled = np.array(ordered, dtype=object)
    rng.shuffle(shuffled)
    n_valid = max(1, int(round(len(shuffled) * valid_fraction)))
    valid = sorted(str(v) for v in shuffled[:n_valid])
    train = sorted(str(v) for v in shuffled[n_valid:])
    return train, valid


class WindowDataset(Dataset):
    def __init__(self, samples: list[WindowSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        return {
            "x": torch.from_numpy(sample.x).float(),
            "target_bins": torch.from_numpy(sample.target_bins).float(),
            "well_id": sample.well_id,
            "start_step": sample.start_step,
        }
```

- [ ] **Step 5: Run window tests and verify they pass**

Run:

```bash
python -m pytest tests/test_windows.py -q
```

Expected: `2 passed`.

- [ ] **Step 6: Commit heatmap/window code**

Run:

```bash
git add mtpnet/heatmap.py mtpnet/windows.py tests/test_windows.py
git commit -m "feat: build MTP heatmap windows"
```

## Task 4: MTP Model And Loss

**Files:**
- Create: `/Users/alexander/Desktop/rogii/MTPNet/mtpnet/model.py`
- Create: `/Users/alexander/Desktop/rogii/MTPNet/mtpnet/loss.py`
- Test: `/Users/alexander/Desktop/rogii/MTPNet/tests/test_model_loss.py`

- [ ] **Step 1: Write failing model/loss tests**

Create `/Users/alexander/Desktop/rogii/MTPNet/tests/test_model_loss.py`:

```python
import torch

from mtpnet.config import LossConfig, ModelConfig
from mtpnet.loss import mtp_loss
from mtpnet.model import MTPNet


def test_model_forward_shapes() -> None:
    model = MTPNet(in_channels=5, height=32, width=10, future_steps=6, cfg=ModelConfig(k_modes=4, conv_channels=(8, 16), hidden_dims=(32,)))
    paths, logits = model(torch.randn(3, 5, 32, 10))
    assert paths.shape == (3, 4, 6)
    assert logits.shape == (3, 4)


def test_mtp_loss_prefers_closest_mode_and_backpropagates() -> None:
    pred = torch.tensor([[[0.0, 0.0], [5.0, 5.0], [1.0, 1.0]]], requires_grad=True)
    logits = torch.zeros(1, 3, requires_grad=True)
    target = torch.tensor([[1.2, 1.1]])
    loss, metrics = mtp_loss(pred, logits, target, LossConfig(alpha_cls=0.2, smooth_lambda=0.0))
    assert metrics["best_k"].tolist() == [2]
    loss.backward()
    assert pred.grad is not None
    assert logits.grad is not None
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
python -m pytest tests/test_model_loss.py -q
```

Expected: FAIL because model/loss modules do not exist.

- [ ] **Step 3: Implement MTP model**

Create `/Users/alexander/Desktop/rogii/MTPNet/mtpnet/model.py`:

```python
from __future__ import annotations

import torch
from torch import Tensor, nn

from .config import ModelConfig


class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MTPNet(nn.Module):
    def __init__(self, in_channels: int, height: int, width: int, future_steps: int, cfg: ModelConfig) -> None:
        super().__init__()
        self.k_modes = cfg.k_modes
        self.future_steps = future_steps
        blocks: list[nn.Module] = []
        c_in = in_channels
        for index, c_out in enumerate(cfg.conv_channels):
            blocks.append(ConvBlock(c_in, c_out))
            blocks.append(ConvBlock(c_out, c_out))
            if index < len(cfg.conv_channels) - 1:
                blocks.append(nn.AvgPool2d(kernel_size=2, stride=2))
            c_in = c_out
        self.encoder = nn.Sequential(*blocks)
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, height, width)
            flat_dim = int(self.encoder(dummy).reshape(1, -1).shape[1])
        head: list[nn.Module] = []
        in_dim = flat_dim
        for hidden_dim in cfg.hidden_dims:
            head.extend(
                [
                    nn.Linear(in_dim, hidden_dim, bias=False),
                    nn.BatchNorm1d(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(cfg.dropout),
                ]
            )
            in_dim = hidden_dim
        self.head = nn.Sequential(*head)
        self.path_head = nn.Linear(in_dim, cfg.k_modes * future_steps)
        self.logit_head = nn.Linear(in_dim, cfg.k_modes)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        batch = x.shape[0]
        features = self.encoder(x).reshape(batch, -1)
        hidden = self.head(features)
        paths = self.path_head(hidden).reshape(batch, self.k_modes, self.future_steps)
        logits = self.logit_head(hidden)
        return paths, logits
```

- [ ] **Step 4: Implement MTP loss**

Create `/Users/alexander/Desktop/rogii/MTPNet/mtpnet/loss.py`:

```python
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from .config import LossConfig


def _path_error(pred: Tensor, target: Tensor, path_loss: str) -> Tensor:
    target_expanded = target[:, None, :]
    if path_loss == "mae":
        return torch.abs(pred - target_expanded).mean(dim=-1)
    if path_loss == "mse":
        return ((pred - target_expanded) ** 2).mean(dim=-1)
    raise ValueError(f"Unsupported path_loss: {path_loss}")


def _smoothness(paths: Tensor) -> Tensor:
    if paths.shape[-1] < 3:
        return paths.new_tensor(0.0)
    second = paths[:, 2:] - 2.0 * paths[:, 1:-1] + paths[:, :-2]
    return torch.abs(second).mean()


def mtp_loss(pred: Tensor, logits: Tensor, target: Tensor, cfg: LossConfig) -> tuple[Tensor, dict[str, Any]]:
    errors = _path_error(pred, target, cfg.path_loss)
    best_k = errors.argmin(dim=1)
    batch_index = torch.arange(pred.shape[0], device=pred.device)
    best_paths = pred[batch_index, best_k]
    reg_loss = _path_error(best_paths[:, None, :], target, cfg.path_loss).mean()
    cls_loss = F.cross_entropy(logits, best_k)
    smooth_loss = _smoothness(best_paths)
    loss = reg_loss + cfg.alpha_cls * cls_loss + cfg.smooth_lambda * smooth_loss
    metrics = {
        "loss": float(loss.detach().cpu()),
        "reg_loss": float(reg_loss.detach().cpu()),
        "cls_loss": float(cls_loss.detach().cpu()),
        "smooth_loss": float(smooth_loss.detach().cpu()),
        "best_k": best_k.detach().cpu(),
        "best_error": errors[batch_index, best_k].detach().cpu(),
    }
    return loss, metrics
```

- [ ] **Step 5: Run model/loss tests and verify they pass**

Run:

```bash
python -m pytest tests/test_model_loss.py -q
```

Expected: `2 passed`.

- [ ] **Step 6: Commit model and loss**

Run:

```bash
git add mtpnet/model.py mtpnet/loss.py tests/test_model_loss.py
git commit -m "feat: add CNN MTP model"
```

## Task 5: Training, Evaluation, And Artifacts

**Files:**
- Create: `/Users/alexander/Desktop/rogii/MTPNet/mtpnet/train.py`
- Create: `/Users/alexander/Desktop/rogii/MTPNet/mtpnet/eval.py`
- Modify: `/Users/alexander/Desktop/rogii/MTPNet/mtpnet/cli.py`
- Test: `/Users/alexander/Desktop/rogii/MTPNet/tests/test_train_smoke.py`

- [ ] **Step 1: Write failing one-batch training test**

Create `/Users/alexander/Desktop/rogii/MTPNet/tests/test_train_smoke.py`:

```python
from pathlib import Path

import pandas as pd
import yaml

from mtpnet.train import train_from_config


def make_well(root: Path, well_id: str, shift: float) -> None:
    root.mkdir(parents=True, exist_ok=True)
    n = 160
    tvt = [1000.0 + shift + i * 0.5 for i in range(n)]
    gr = [80.0 + ((i % 20) - 10) * 1.5 for i in range(n)]
    tvt_input = [value if i < 80 else None for i, value in enumerate(tvt)]
    pd.DataFrame({"id": [f"{well_id}_{i}" for i in range(n)], "TVT": tvt, "TVT_input": tvt_input, "GR": gr}).to_csv(
        root / f"{well_id}__horizontal_well.csv", index=False
    )
    type_tvt = [970.0 + shift + i * 0.5 for i in range(260)]
    type_gr = [80.0 + ((i % 20) - 10) * 1.5 for i in range(260)]
    pd.DataFrame({"TVT": type_tvt, "GR": type_gr}).to_csv(root / f"{well_id}__typewell.csv", index=False)


def test_train_from_config_writes_metrics(tmp_path: Path) -> None:
    train_dir = tmp_path / "data" / "train"
    make_well(train_dir, "well_a", 0.0)
    make_well(train_dir, "well_b", 5.0)
    config = {
        "data": {"data_dir": str(tmp_path / "data"), "train_dir": str(train_dir), "k_wells": -1},
        "window": {
            "rows_per_step": 4,
            "history_steps": 4,
            "future_steps": 6,
            "vertical_bins": 32,
            "vertical_radius_ft": 60.0,
            "stride_steps": 4,
            "max_windows_per_well": 4,
            "channels": ["gr_diff", "abs_gr_diff", "history_mask", "history_sdf", "finite_mask"],
        },
        "model": {"k_modes": 3, "conv_channels": [8, 16], "hidden_dims": [32], "dropout": 0.0},
        "train": {"batch_size": 4, "epochs": 1, "learning_rate": 0.001, "device": "cpu", "num_workers": 0, "seed": 42},
        "validation": {"valid_fraction": 0.5, "seed": 42},
        "run": {"name": "unit", "output_dir": str(tmp_path / "artifacts" / "unit")},
    }
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    summary = train_from_config(config_path)
    assert summary["valid"]["num_windows"] > 0
    assert (tmp_path / "artifacts" / "unit" / "metrics.json").exists()
    assert (tmp_path / "artifacts" / "unit" / "checkpoints" / "best.pt").exists()
```

- [ ] **Step 2: Run training test and verify it fails**

Run:

```bash
python -m pytest tests/test_train_smoke.py -q
```

Expected: FAIL because `mtpnet.train` does not exist.

- [ ] **Step 3: Implement train/eval orchestration**

Create `/Users/alexander/Desktop/rogii/MTPNet/mtpnet/train.py` with these functions:

```python
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from .config import MTPConfig, load_config
from .io import discover_wells, load_well
from .loss import mtp_loss
from .model import MTPNet
from .windows import WindowDataset, WindowSample, build_windows_for_well, split_wells


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_all_windows(cfg: MTPConfig) -> list[WindowSample]:
    samples: list[WindowSample] = []
    skipped: list[str] = []
    for well in discover_wells(cfg.data):
        horizontal, typewell = load_well(well)
        well_samples = build_windows_for_well(well.well_id, horizontal, typewell, cfg.window)
        if not well_samples:
            skipped.append(well.well_id)
            continue
        samples.extend(well_samples)
    if not samples:
        raise RuntimeError(f"No windows were built. Skipped wells: {skipped[:10]}")
    return samples


def split_samples(samples: list[WindowSample], cfg: MTPConfig) -> tuple[list[WindowSample], list[WindowSample]]:
    wells = sorted({sample.well_id for sample in samples})
    train_wells, valid_wells = split_wells(wells, cfg.validation.valid_fraction, cfg.validation.seed)
    valid_set = set(valid_wells)
    train = [sample for sample in samples if sample.well_id not in valid_set]
    valid = [sample for sample in samples if sample.well_id in valid_set]
    if not train or not valid:
        raise RuntimeError(f"Invalid split: train_windows={len(train)} valid_windows={len(valid)}")
    return train, valid


def _loader(samples: list[WindowSample], cfg: MTPConfig, shuffle: bool) -> DataLoader:
    return DataLoader(
        WindowDataset(samples),
        batch_size=cfg.train.batch_size,
        shuffle=shuffle,
        num_workers=cfg.train.num_workers,
    )


def _evaluate(model: MTPNet, samples: list[WindowSample], cfg: MTPConfig, device: torch.device) -> tuple[dict[str, Any], pd.DataFrame]:
    model.eval()
    rows: list[dict[str, Any]] = []
    errors_top1: list[float] = []
    errors_weighted: list[float] = []
    errors_oracle: list[float] = []
    errors_top3: list[float] = []
    best_modes: list[int] = []
    with torch.no_grad():
        for batch in _loader(samples, cfg, shuffle=False):
            x = batch["x"].to(device)
            target = batch["target_bins"].to(device)
            paths, logits = model(x)
            prob = F.softmax(logits, dim=1)
            err = torch.sqrt(((paths - target[:, None, :]) ** 2).mean(dim=-1))
            top1 = prob.argmax(dim=1)
            batch_idx = torch.arange(paths.shape[0], device=device)
            weighted = (paths * prob[:, :, None]).sum(dim=1)
            top3_idx = torch.topk(prob, k=min(3, prob.shape[1]), dim=1).indices
            top3_err = torch.gather(err, 1, top3_idx).min(dim=1).values
            oracle_err, best_k = err.min(dim=1)
            errors_top1.extend(err[batch_idx, top1].cpu().tolist())
            errors_weighted.extend(torch.sqrt(((weighted - target) ** 2).mean(dim=-1)).cpu().tolist())
            errors_oracle.extend(oracle_err.cpu().tolist())
            errors_top3.extend(top3_err.cpu().tolist())
            best_modes.extend(best_k.cpu().tolist())
            for i in range(paths.shape[0]):
                rows.append(
                    {
                        "well_id": batch["well_id"][i],
                        "start_step": int(batch["start_step"][i]),
                        "top1_mode": int(top1[i].cpu()),
                        "best_mode": int(best_k[i].cpu()),
                        "top1_rmse_bins": float(err[i, top1[i]].cpu()),
                        "oracle_rmse_bins": float(oracle_err[i].cpu()),
                    }
                )
    mode_hist = {str(k): int(v) for k, v in zip(*np.unique(np.array(best_modes), return_counts=True), strict=False)}
    metrics = {
        "num_windows": len(samples),
        "top1_rmse_bins": float(np.mean(errors_top1)),
        "weighted_mean_rmse_bins": float(np.mean(errors_weighted)),
        "oracle_topk_rmse_bins": float(np.mean(errors_oracle)),
        "oracle_top3_rmse_bins": float(np.mean(errors_top3)),
        "best_mode_mae_bins": float(np.mean(errors_oracle)),
        "mode_usage_histogram": mode_hist,
    }
    return metrics, pd.DataFrame(rows)


def train_from_config(config_path: str | Path) -> dict[str, Any]:
    cfg = load_config(config_path)
    set_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    output_dir = cfg.run.output_dir
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    samples = build_all_windows(cfg)
    train_samples, valid_samples = split_samples(samples, cfg)
    first = samples[0]
    model = MTPNet(
        in_channels=first.x.shape[0],
        height=first.x.shape[1],
        width=first.x.shape[2],
        future_steps=cfg.window.future_steps,
        cfg=cfg.model,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay)
    best_valid = float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in _loader(train_samples, cfg, shuffle=True):
            optimizer.zero_grad(set_to_none=True)
            x = batch["x"].to(device)
            target = batch["target_bins"].to(device)
            paths, logits = model(x)
            loss, _ = mtp_loss(paths, logits, target, cfg.loss)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        valid_metrics, _ = _evaluate(model, valid_samples, cfg, device)
        valid_score = valid_metrics["oracle_topk_rmse_bins"]
        history.append({"epoch": float(epoch), "train_loss": float(np.mean(losses)), "valid_oracle_topk_rmse_bins": valid_score})
        if valid_score < best_valid:
            best_valid = valid_score
            torch.save({"model": model.state_dict(), "config_path": str(config_path)}, checkpoint_dir / "best.pt")

    valid_metrics, pred_frame = _evaluate(model, valid_samples, cfg, device)
    train_metrics, _ = _evaluate(model, train_samples, cfg, device)
    summary = {
        "train": train_metrics,
        "valid": valid_metrics,
        "history": history,
        "num_train_wells": len({s.well_id for s in train_samples}),
        "num_valid_wells": len({s.well_id for s in valid_samples}),
    }
    (output_dir / "config_resolved.yml").write_text(yaml.safe_dump({"config_path": str(config_path)}, sort_keys=False), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pred_frame.to_parquet(output_dir / "window_predictions.parquet", index=False)
    print(json.dumps(summary["valid"], indent=2), flush=True)
    return summary
```

Create `/Users/alexander/Desktop/rogii/MTPNet/mtpnet/eval.py`:

```python
from __future__ import annotations

import json
from pathlib import Path


def evaluate_run(run_dir: str | Path) -> dict[str, object]:
    path = Path(run_dir) / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    metrics = json.loads(path.read_text(encoding="utf-8"))
    print(json.dumps(metrics.get("valid", metrics), indent=2), flush=True)
    return metrics
```

Leave `/Users/alexander/Desktop/rogii/MTPNet/mtpnet/cli.py` as created in Task 2; it already imports `train_from_config` and `evaluate_run` lazily.

- [ ] **Step 4: Run training smoke test and verify it passes**

Run:

```bash
python -m pytest tests/test_train_smoke.py -q
```

Expected: `1 passed`.

- [ ] **Step 5: Commit training and evaluation**

Run:

```bash
git add mtpnet/train.py mtpnet/eval.py tests/test_train_smoke.py
git commit -m "feat: train and evaluate MTPNet prototype"
```

## Task 6: End-To-End Smoke With Real Data

**Files:**
- Modify only if needed: `/Users/alexander/Desktop/rogii/MTPNet/configs/mtp_smoke.yml`
- Modify only if needed: files touched in previous tasks

- [ ] **Step 1: Install project locally if imports fail**

Run from `/Users/alexander/Desktop/rogii/MTPNet`:

```bash
python -m pip install -e ".[dev]"
```

Expected: package installs successfully. If the environment already imports local packages through `pythonpath`, this step is still safe.

- [ ] **Step 2: Copy data into MTPNet**

Run:

```bash
make copy-data COPY_SOURCE=../old/data COPY_TARGET=data
```

Expected: `data/train` exists and contains Kaggle well CSVs.

- [ ] **Step 3: Run unit tests**

Run:

```bash
make test
```

Expected: all tests pass.

- [ ] **Step 4: Run real-data smoke train**

Run:

```bash
make smoke
```

Expected:

- `artifacts/mtp_smoke/checkpoints/best.pt` exists;
- `artifacts/mtp_smoke/metrics.json` exists;
- `artifacts/mtp_smoke/window_predictions.parquet` exists;
- CLI prints valid metrics including `oracle_topk_rmse_bins`.

- [ ] **Step 5: Inspect smoke metrics**

Run:

```bash
make eval RUN_DIR=artifacts/mtp_smoke
```

Expected: JSON includes `top1_rmse_bins`, `weighted_mean_rmse_bins`, `oracle_topk_rmse_bins`, `oracle_top3_rmse_bins`, and `mode_usage_histogram`.

- [ ] **Step 6: Commit smoke-ready fixes**

If no code changes were needed after Task 5, skip this commit. If smoke uncovered small fixes, commit only those MTPNet files:

```bash
git status --short
git add mtpnet configs tests Makefile pyproject.toml
git commit -m "fix: make MTPNet smoke run pass"
```

## Task 7: Final Verification And Handoff

**Files:**
- No required file changes unless verification reveals a defect.

- [ ] **Step 1: Run full focused test suite**

Run:

```bash
python -m pytest tests -q
```

Expected: all tests pass.

- [ ] **Step 2: Confirm Git status only contains intentional files**

Run from `/Users/alexander/Desktop/rogii`:

```bash
git status --short
```

Expected: MTPNet implementation files may be modified or committed. Existing unrelated deletions/moves outside MTPNet should not be reverted.

- [ ] **Step 3: Report results**

Final report should include:

- path to `/Users/alexander/Desktop/rogii/MTPNet`;
- commands run;
- test result summary;
- smoke metric highlights from `artifacts/mtp_smoke/metrics.json`;
- note that `k_wells=-1` is supported in config and positive values limit wells.

## Self-Review Notes

- Spec coverage: project boundary, config, `k_wells=-1`, optional data copy, heatmap windows, channels, CNN-MTP model, MTP loss, train/eval commands, window metrics, well-level validation split, error handling, and smoke verification are covered by Tasks 1-7.
- No unfinished task markers remain. Later milestones like sequential tracking, fold-safe OOF, B2/A SDF priors, Docker, and submission generation are intentionally excluded from this first implementation and named in the design spec as extensions.
- Type consistency: `WindowConfig.channels`, `DataConfig.k_wells`, `MTPNet.forward`, `mtp_loss`, `train_from_config`, and `evaluate_run` use consistent names across tasks.
