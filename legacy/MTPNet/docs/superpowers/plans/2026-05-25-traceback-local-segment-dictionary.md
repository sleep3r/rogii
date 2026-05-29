# TraceBack Local Segment Dictionary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build TraceBack v0: a schema-safe local GR event matching audit and candidate generator that tests whether short GR signatures improve candidate-bank oracle beyond global heatmap matching.

**Architecture:** Split the experiment into focused modules: event extraction, dictionary construction, event matching, band/candidate generation, and a CLI/report orchestrator. The first deliverable is diagnostic: strict normal-vs-shuffled metrics and candidate-bank oracle, not a final submit selector.

**Tech Stack:** Python, NumPy, pandas, matplotlib/Pillow for figures, existing `mtpnet` utilities, `pytest`, existing schema guard in `mtpnet.schema_safe`.

---

## File Structure

Create:

- `mtpnet/traceback_events.py`  
  Compression, robust GR normalization, local patch extraction, event extraction.

- `mtpnet/traceback_dictionary.py`  
  Same-well known dictionary, typewell dictionary, fold-safe train-well dictionary helpers.

- `mtpnet/traceback_match.py`  
  Patch score functions, normal/shuffled/zero/location-only variants, event-level metrics.

- `mtpnet/traceback_candidates.py`  
  Event anchor aggregation, broad band generation, candidate path generation, candidate-bank oracle wrappers.

- `mtpnet/traceback.py`  
  CLI orchestration, artifact writing, report and figure generation.

- `tests/test_traceback.py`  
  Unit and smoke tests for all TraceBack pieces.

Modify:

- `Makefile`  
  Add `TRACEBACK_*` variables and `traceback` target.

Optional later, only after GO:

- `mtpnet/candidate_bank.py`  
  Add an explicit TraceBack candidate input path.

- `policy_solver/README.md`  
  Document TraceBack as an optional candidate source after candidate oracle improves.

---

## Task 1: TraceBack Public Types And Schema Guard

**Files:**
- Create: `mtpnet/traceback_events.py`
- Test: `tests/test_traceback.py`

- [ ] **Step 1: Write failing tests for config and schema guard**

Add to `tests/test_traceback.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_traceback_config_defaults_are_schema_safe() -> None:
    from mtpnet.traceback_events import TracebackConfig

    cfg = TracebackConfig()

    assert cfg.rows_per_step == 32
    assert cfg.patch_radii == (3, 5, 9, 15)
    assert cfg.min_prominence_z > 0.0


def test_traceback_schema_guard_rejects_forbidden_columns() -> None:
    from mtpnet.traceback_events import assert_traceback_feature_schema_safe

    with pytest.raises(ValueError, match="forbidden"):
        assert_traceback_feature_schema_safe(["MD", "GR", "TVT", "Geology"])

    assert_traceback_feature_schema_safe(["MD", "X", "Y", "Z", "GR", "TVT_input"])
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
uv run --extra dev pytest -q tests/test_traceback.py::test_traceback_config_defaults_are_schema_safe tests/test_traceback.py::test_traceback_schema_guard_rejects_forbidden_columns
```

Expected: fail with `ModuleNotFoundError: No module named 'mtpnet.traceback_events'`.

- [ ] **Step 3: Implement minimal config and schema guard**

Create `mtpnet/traceback_events.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

from .schema_safe import assert_schema_safe_columns


@dataclass(frozen=True)
class TracebackConfig:
    rows_per_step: int = 32
    patch_radii: tuple[int, ...] = (3, 5, 9, 15)
    min_patch_points: int = 3
    min_prominence_z: float = 0.75
    max_events_per_well: int = 256
    vertical_step_ft: float = 5.0
    top_k_matches: int = 10
    seed: int = 42


def assert_traceback_feature_schema_safe(columns: list[str] | tuple[str, ...] | set[str]) -> None:
    assert_schema_safe_columns(columns, context="TraceBack feature builder")
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
uv run --extra dev pytest -q tests/test_traceback.py::test_traceback_config_defaults_are_schema_safe tests/test_traceback.py::test_traceback_schema_guard_rejects_forbidden_columns
```

Expected: both tests pass.

- [ ] **Step 5: Commit**

```bash
git add mtpnet/traceback_events.py tests/test_traceback.py
git commit -m "feat: add traceback config and schema guard"
```

---

## Task 2: Compression, Patch Extraction, And Event Extraction

**Files:**
- Modify: `mtpnet/traceback_events.py`
- Test: `tests/test_traceback.py`

- [ ] **Step 1: Write failing tests for compression and synthetic event extraction**

Append to `tests/test_traceback.py`:

```python
def test_compress_well_rows_keeps_partial_tail_and_known_mask() -> None:
    from mtpnet.traceback_events import compress_well_rows

    frame = pd.DataFrame(
        {
            "id": [f"w_{i}" for i in range(5)],
            "well_id": ["w"] * 5,
            "row_idx": np.arange(5),
            "MD": np.arange(5, dtype=float),
            "X": np.zeros(5),
            "Y": np.zeros(5),
            "Z": -np.arange(5, dtype=float),
            "GR": [1.0, 3.0, np.nan, 7.0, 9.0],
            "TVT_input": [100.0, 101.0, np.nan, np.nan, np.nan],
            "TVT": [100.0, 101.0, 102.0, 103.0, 104.0],
        }
    )

    comp = compress_well_rows(frame, rows_per_step=2)

    assert comp["step"].tolist() == [0, 1, 2]
    assert comp["row_start"].tolist() == [0, 2, 4]
    assert comp["row_end"].tolist() == [1, 3, 4]
    assert comp["known_mask"].tolist() == [True, False, False]
    assert np.isfinite(comp["GR_filled"]).all()


def test_extract_traceback_events_finds_peak_and_trough() -> None:
    from mtpnet.traceback_events import TracebackConfig, extract_traceback_events

    comp = pd.DataFrame(
        {
            "well_id": ["w"] * 9,
            "step": np.arange(9),
            "row_start": np.arange(9),
            "row_end": np.arange(9),
            "MD": np.arange(9, dtype=float),
            "X": np.zeros(9),
            "Y": np.zeros(9),
            "Z": -np.arange(9, dtype=float),
            "GR_filled": [0.0, 1.0, 5.0, 1.0, 0.0, -1.0, -4.0, -1.0, 0.0],
            "GR_finite_frac": np.ones(9),
            "TVT_input": [100, 101, 102, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            "known_mask": [True, True, True, False, False, False, False, False, False],
        }
    )

    events = extract_traceback_events(comp, TracebackConfig(patch_radii=(1,), min_prominence_z=0.5))

    assert {"peak", "trough"}.issubset(set(events["event_type"]))
    assert set(events["well_id"]) == {"w"}
    assert {"patch_gr", "patch_dgr", "patch_radius"}.issubset(events.columns)
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
uv run --extra dev pytest -q tests/test_traceback.py::test_compress_well_rows_keeps_partial_tail_and_known_mask tests/test_traceback.py::test_extract_traceback_events_finds_peak_and_trough
```

Expected: fail because `compress_well_rows` and `extract_traceback_events` are undefined.

- [ ] **Step 3: Implement compression and event extraction**

Add to `mtpnet/traceback_events.py`:

```python
from typing import Any

import numpy as np
import pandas as pd

from .heatmap import fill_nan


def _robust_z(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=np.float32)
    med = float(np.nanmedian(arr[finite]))
    mad = float(np.nanmedian(np.abs(arr[finite] - med)))
    scale = 1.4826 * mad if mad > 1e-6 else float(np.nanstd(arr[finite]))
    if not np.isfinite(scale) or scale <= 1e-6:
        scale = 1.0
    return np.where(np.isfinite(arr), (arr - med) / scale, 0.0).astype(np.float32)


def _mean_or_nan(values: np.ndarray) -> float:
    finite = np.isfinite(values)
    if not finite.any():
        return float("nan")
    return float(np.nanmean(values[finite]))


def compress_well_rows(frame: pd.DataFrame, *, rows_per_step: int) -> pd.DataFrame:
    assert_traceback_feature_schema_safe(["MD", "X", "Y", "Z", "GR", "TVT_input"])
    well = frame.sort_values("row_idx").reset_index(drop=True).copy()
    gr_raw = pd.to_numeric(well["GR"], errors="coerce").to_numpy(dtype=np.float32)
    gr_filled, finite_mask = fill_nan(gr_raw)
    well["GR_filled_row"] = gr_filled
    well["GR_finite_row"] = finite_mask

    rows: list[dict[str, Any]] = []
    for step, start in enumerate(range(0, len(well), rows_per_step)):
        chunk = well.iloc[start : start + rows_per_step]
        tvt_input = pd.to_numeric(chunk["TVT_input"], errors="coerce")
        known = tvt_input.notna()
        rows.append(
            {
                "well_id": str(chunk["well_id"].iloc[0]),
                "step": int(step),
                "row_start": int(chunk["row_idx"].min()),
                "row_end": int(chunk["row_idx"].max()),
                "MD": _mean_or_nan(pd.to_numeric(chunk["MD"], errors="coerce").to_numpy()),
                "X": _mean_or_nan(pd.to_numeric(chunk["X"], errors="coerce").to_numpy()),
                "Y": _mean_or_nan(pd.to_numeric(chunk["Y"], errors="coerce").to_numpy()),
                "Z": _mean_or_nan(pd.to_numeric(chunk["Z"], errors="coerce").to_numpy()),
                "GR_filled": _mean_or_nan(chunk["GR_filled_row"].to_numpy()),
                "GR_finite_frac": float(np.mean(chunk["GR_finite_row"].to_numpy(dtype=bool))),
                "TVT_input": _mean_or_nan(tvt_input.to_numpy(dtype=np.float64)),
                "known_mask": bool(known.any()),
                "true_TVT": _mean_or_nan(pd.to_numeric(chunk.get("TVT", pd.Series(np.nan, index=chunk.index)), errors="coerce").to_numpy()),
            }
        )
    out = pd.DataFrame(rows)
    out["GR_z"] = _robust_z(out["GR_filled"].to_numpy())
    out["dGR_z"] = np.gradient(out["GR_z"].to_numpy(dtype=np.float32)).astype(np.float32)
    return out


def _patch(values: np.ndarray, center: int, radius: int) -> np.ndarray:
    out = np.full(radius * 2 + 1, np.nan, dtype=np.float32)
    lo = max(0, center - radius)
    hi = min(len(values), center + radius + 1)
    dst_lo = lo - (center - radius)
    out[dst_lo : dst_lo + (hi - lo)] = values[lo:hi]
    return out


def extract_traceback_events(comp: pd.DataFrame, cfg: TracebackConfig) -> pd.DataFrame:
    if comp.empty:
        return pd.DataFrame()
    gr = comp["GR_z"].to_numpy(dtype=np.float32) if "GR_z" in comp else _robust_z(comp["GR_filled"].to_numpy())
    dgr = np.gradient(gr).astype(np.float32)
    rows: list[dict[str, Any]] = []
    for i in range(1, len(comp) - 1):
        left, cur, right = float(gr[i - 1]), float(gr[i]), float(gr[i + 1])
        event_type = ""
        prominence = 0.0
        if cur > left and cur > right:
            event_type = "peak"
            prominence = cur - max(left, right)
        elif cur < left and cur < right:
            event_type = "trough"
            prominence = min(left, right) - cur
        if not event_type or prominence < cfg.min_prominence_z:
            continue
        for radius in cfg.patch_radii:
            patch_gr = _patch(gr, i, radius)
            if np.isfinite(patch_gr).sum() < cfg.min_patch_points:
                continue
            rows.append(
                {
                    "well_id": str(comp["well_id"].iloc[i]),
                    "step": int(comp["step"].iloc[i]),
                    "row_start": int(comp["row_start"].iloc[i]),
                    "row_end": int(comp["row_end"].iloc[i]),
                    "MD": float(comp["MD"].iloc[i]),
                    "X": float(comp["X"].iloc[i]),
                    "Y": float(comp["Y"].iloc[i]),
                    "Z": float(comp["Z"].iloc[i]),
                    "known_mask": bool(comp["known_mask"].iloc[i]),
                    "TVT_input": float(comp["TVT_input"].iloc[i]) if np.isfinite(comp["TVT_input"].iloc[i]) else np.nan,
                    "true_TVT": float(comp["true_TVT"].iloc[i]) if "true_TVT" in comp and np.isfinite(comp["true_TVT"].iloc[i]) else np.nan,
                    "event_type": event_type,
                    "prominence": float(prominence),
                    "finite_frac": float(comp["GR_finite_frac"].iloc[i]),
                    "patch_radius": int(radius),
                    "patch_gr": patch_gr.astype(np.float32),
                    "patch_dgr": _patch(dgr, i, radius).astype(np.float32),
                }
            )
    events = pd.DataFrame(rows)
    if len(events) > cfg.max_events_per_well:
        events = events.sort_values("prominence", ascending=False).head(cfg.max_events_per_well)
    return events.reset_index(drop=True)
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
uv run --extra dev pytest -q tests/test_traceback.py::test_compress_well_rows_keeps_partial_tail_and_known_mask tests/test_traceback.py::test_extract_traceback_events_finds_peak_and_trough
```

Expected: both tests pass.

- [ ] **Step 5: Commit**

```bash
git add mtpnet/traceback_events.py tests/test_traceback.py
git commit -m "feat: extract local traceback events"
```

---

## Task 3: Same-Well, Typewell, And Fold-Safe Dictionaries

**Files:**
- Create: `mtpnet/traceback_dictionary.py`
- Test: `tests/test_traceback.py`

- [ ] **Step 1: Write failing dictionary tests**

Append to `tests/test_traceback.py`:

```python
def test_same_well_dictionary_uses_only_known_tvt_input_events() -> None:
    from mtpnet.traceback_dictionary import build_same_well_dictionary

    events = pd.DataFrame(
        {
            "well_id": ["w", "w"],
            "step": [1, 5],
            "known_mask": [True, False],
            "TVT_input": [101.0, np.nan],
            "true_TVT": [101.0, 105.0],
            "event_type": ["peak", "peak"],
            "prominence": [1.0, 1.0],
            "finite_frac": [1.0, 1.0],
            "patch_radius": [1, 1],
            "patch_gr": [np.array([0, 1, 0], dtype=np.float32)] * 2,
            "patch_dgr": [np.array([1, 0, -1], dtype=np.float32)] * 2,
        }
    )

    dictionary = build_same_well_dictionary(events)

    assert len(dictionary) == 1
    assert dictionary["source"].iloc[0] == "same_well_known"
    assert dictionary["source_TVT"].iloc[0] == 101.0


def test_fold_safe_dictionary_excludes_validation_wells() -> None:
    from mtpnet.traceback_dictionary import build_fold_safe_train_dictionary

    events = pd.DataFrame(
        {
            "well_id": ["train", "valid"],
            "step": [1, 1],
            "true_TVT": [101.0, 201.0],
            "known_mask": [False, False],
            "event_type": ["peak", "peak"],
            "prominence": [1.0, 1.0],
            "finite_frac": [1.0, 1.0],
            "patch_radius": [1, 1],
            "patch_gr": [np.array([0, 1, 0], dtype=np.float32)] * 2,
            "patch_dgr": [np.array([1, 0, -1], dtype=np.float32)] * 2,
        }
    )

    dictionary = build_fold_safe_train_dictionary(events, train_wells=["train"])

    assert dictionary["source_well_id"].tolist() == ["train"]
    assert dictionary["source_TVT"].tolist() == [101.0]
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
uv run --extra dev pytest -q tests/test_traceback.py::test_same_well_dictionary_uses_only_known_tvt_input_events tests/test_traceback.py::test_fold_safe_dictionary_excludes_validation_wells
```

Expected: fail because `mtpnet.traceback_dictionary` is missing.

- [ ] **Step 3: Implement dictionary builders**

Create `mtpnet/traceback_dictionary.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd


DICT_COLUMNS = [
    "source",
    "source_well_id",
    "source_step",
    "source_TVT",
    "event_type",
    "prominence",
    "finite_frac",
    "patch_radius",
    "patch_gr",
    "patch_dgr",
]


def _empty_dictionary() -> pd.DataFrame:
    return pd.DataFrame(columns=DICT_COLUMNS)


def _dictionary_from_events(events: pd.DataFrame, *, source: str, tvt_column: str) -> pd.DataFrame:
    if events.empty or tvt_column not in events:
        return _empty_dictionary()
    rows = []
    for _, row in events.iterrows():
        tvt = pd.to_numeric(pd.Series([row[tvt_column]]), errors="coerce").iloc[0]
        if not np.isfinite(tvt):
            continue
        rows.append(
            {
                "source": source,
                "source_well_id": str(row["well_id"]),
                "source_step": int(row["step"]),
                "source_TVT": float(tvt),
                "event_type": str(row["event_type"]),
                "prominence": float(row["prominence"]),
                "finite_frac": float(row["finite_frac"]),
                "patch_radius": int(row["patch_radius"]),
                "patch_gr": np.asarray(row["patch_gr"], dtype=np.float32),
                "patch_dgr": np.asarray(row["patch_dgr"], dtype=np.float32),
            }
        )
    return pd.DataFrame(rows, columns=DICT_COLUMNS)


def build_same_well_dictionary(events: pd.DataFrame) -> pd.DataFrame:
    known = events[events["known_mask"].astype(bool)].copy() if not events.empty else events
    return _dictionary_from_events(known, source="same_well_known", tvt_column="TVT_input")


def build_fold_safe_train_dictionary(events: pd.DataFrame, *, train_wells: list[str] | tuple[str, ...] | set[str]) -> pd.DataFrame:
    train_set = {str(well_id) for well_id in train_wells}
    train_events = events[events["well_id"].astype(str).isin(train_set)].copy()
    return _dictionary_from_events(train_events, source="fold_train", tvt_column="true_TVT")
```

- [ ] **Step 4: Add typewell dictionary test**

Append:

```python
def test_typewell_dictionary_samples_regular_tvt_grid() -> None:
    from mtpnet.traceback_dictionary import build_typewell_dictionary

    typewell = pd.DataFrame(
        {
            "TVT": [100.0, 105.0, 110.0, 115.0, 120.0],
            "GR": [0.0, 1.0, 4.0, 1.0, 0.0],
        }
    )

    dictionary = build_typewell_dictionary(typewell, vertical_step_ft=5.0, patch_radii=(1,), min_prominence_z=0.5)

    assert not dictionary.empty
    assert set(dictionary["source"]) == {"typewell"}
    assert dictionary["source_TVT"].between(100.0, 120.0).all()
```

- [ ] **Step 5: Implement typewell dictionary**

Append to `mtpnet/traceback_dictionary.py`:

```python
from .correlation_panel import _regular_typewell_grid
from .traceback_events import TracebackConfig, extract_traceback_events


def build_typewell_dictionary(
    typewell: pd.DataFrame,
    *,
    vertical_step_ft: float,
    patch_radii: tuple[int, ...],
    min_prominence_z: float,
) -> pd.DataFrame:
    grid, gr = _regular_typewell_grid(typewell, vertical_step_ft)
    if grid.size == 0:
        return _empty_dictionary()
    comp = pd.DataFrame(
        {
            "well_id": "typewell",
            "step": np.arange(grid.size, dtype=int),
            "row_start": np.arange(grid.size, dtype=int),
            "row_end": np.arange(grid.size, dtype=int),
            "MD": np.arange(grid.size, dtype=float),
            "X": np.zeros(grid.size),
            "Y": np.zeros(grid.size),
            "Z": np.zeros(grid.size),
            "GR_filled": gr.astype(np.float32),
            "GR_finite_frac": np.ones(grid.size),
            "TVT_input": grid.astype(np.float32),
            "known_mask": np.ones(grid.size, dtype=bool),
            "true_TVT": grid.astype(np.float32),
        }
    )
    cfg = TracebackConfig(
        rows_per_step=1,
        patch_radii=patch_radii,
        min_prominence_z=min_prominence_z,
        vertical_step_ft=vertical_step_ft,
    )
    events = extract_traceback_events(comp, cfg)
    out = _dictionary_from_events(events, source="typewell", tvt_column="true_TVT")
    out["source_well_id"] = "typewell"
    return out
```

- [ ] **Step 6: Run dictionary tests**

Run:

```bash
uv run --extra dev pytest -q tests/test_traceback.py::test_same_well_dictionary_uses_only_known_tvt_input_events tests/test_traceback.py::test_fold_safe_dictionary_excludes_validation_wells tests/test_traceback.py::test_typewell_dictionary_samples_regular_tvt_grid
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add mtpnet/traceback_dictionary.py tests/test_traceback.py
git commit -m "feat: build traceback event dictionaries"
```

---

## Task 4: Event Match Scoring And Shuffled Sanity Variants

**Files:**
- Create: `mtpnet/traceback_match.py`
- Test: `tests/test_traceback.py`

- [ ] **Step 1: Write failing score tests**

Append:

```python
def test_patch_score_ranks_exact_match_above_reverse_patch() -> None:
    from mtpnet.traceback_match import score_event_against_dictionary

    event = pd.Series(
        {
            "event_type": "peak",
            "prominence": 1.0,
            "finite_frac": 1.0,
            "patch_radius": 1,
            "patch_gr": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            "patch_dgr": np.array([1.0, 0.0, -1.0], dtype=np.float32),
        }
    )
    dictionary = pd.DataFrame(
        {
            "source": ["exact", "bad"],
            "source_well_id": ["a", "b"],
            "source_step": [0, 0],
            "source_TVT": [100.0, 200.0],
            "event_type": ["peak", "trough"],
            "prominence": [1.0, 1.0],
            "finite_frac": [1.0, 1.0],
            "patch_radius": [1, 1],
            "patch_gr": [
                np.array([0.0, 1.0, 0.0], dtype=np.float32),
                np.array([0.0, -1.0, 0.0], dtype=np.float32),
            ],
            "patch_dgr": [
                np.array([1.0, 0.0, -1.0], dtype=np.float32),
                np.array([-1.0, 0.0, 1.0], dtype=np.float32),
            ],
        }
    )

    scored = score_event_against_dictionary(event, dictionary, location_weight=0.0)

    assert scored.iloc[0]["source_TVT"] == 100.0
    assert scored.iloc[0]["score"] > scored.iloc[1]["score"]
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```bash
uv run --extra dev pytest -q tests/test_traceback.py::test_patch_score_ranks_exact_match_above_reverse_patch
```

Expected: fail because `mtpnet.traceback_match` is missing.

- [ ] **Step 3: Implement scoring**

Create `mtpnet/traceback_match.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    finite = np.isfinite(aa) & np.isfinite(bb)
    if finite.sum() < 3:
        return 0.0
    aa = aa[finite] - np.mean(aa[finite])
    bb = bb[finite] - np.mean(bb[finite])
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom <= 1e-9:
        return 0.0
    return float(np.dot(aa, bb) / denom)


def _mad_score(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    finite = np.isfinite(aa) & np.isfinite(bb)
    if not finite.any():
        return -10.0
    return -float(np.nanmedian(np.abs(aa[finite] - bb[finite])))


def score_event_against_dictionary(
    event: pd.Series,
    dictionary: pd.DataFrame,
    *,
    location_weight: float,
    bridge_tvt: float | None = None,
    location_sigma_ft: float = 80.0,
    top_k: int = 10,
) -> pd.DataFrame:
    if dictionary.empty:
        return pd.DataFrame()
    radius = int(event["patch_radius"])
    candidates = dictionary[dictionary["patch_radius"].astype(int) == radius].copy()
    if candidates.empty:
        return pd.DataFrame()
    rows = []
    event_gr = np.asarray(event["patch_gr"], dtype=np.float32)
    event_dgr = np.asarray(event["patch_dgr"], dtype=np.float32)
    for _, cand in candidates.iterrows():
        shape_corr = _corr(event_gr, cand["patch_gr"])
        dgr_corr = _corr(event_dgr, cand["patch_dgr"])
        mad = _mad_score(event_gr, cand["patch_gr"])
        event_type_hit = 1.0 if str(event["event_type"]) == str(cand["event_type"]) else 0.0
        prom_score = -abs(float(event["prominence"]) - float(cand["prominence"]))
        finite_weight = min(float(event["finite_frac"]), float(cand["finite_frac"]))
        location_score = 0.0
        if bridge_tvt is not None and np.isfinite(bridge_tvt):
            location_score = -abs(float(cand["source_TVT"]) - float(bridge_tvt)) / max(location_sigma_ft, 1.0)
        score = finite_weight * (
            shape_corr
            + 0.5 * dgr_corr
            + 0.3 * mad
            + 0.2 * event_type_hit
            + 0.15 * prom_score
            + location_weight * location_score
        )
        out = cand.to_dict()
        out.update(
            {
                "event_well_id": str(event.get("well_id", "")),
                "event_step": int(event.get("step", -1)),
                "score": float(score),
                "shape_corr": float(shape_corr),
                "dgr_corr": float(dgr_corr),
                "mad_score": float(mad),
                "event_type_hit": float(event_type_hit),
                "location_score": float(location_score),
            }
        )
        rows.append(out)
    return pd.DataFrame(rows).sort_values("score", ascending=False).head(top_k).reset_index(drop=True)
```

- [ ] **Step 4: Add sanity variant test**

Append:

```python
def test_make_sanity_events_supports_shuffled_and_zero_gr() -> None:
    from mtpnet.traceback_match import make_sanity_events

    events = pd.DataFrame(
        {
            "well_id": ["w"] * 2,
            "step": [1, 2],
            "patch_gr": [np.array([1, 2, 3], dtype=np.float32), np.array([4, 5, 6], dtype=np.float32)],
            "patch_dgr": [np.array([1, 1, 1], dtype=np.float32), np.array([2, 2, 2], dtype=np.float32)],
        }
    )

    variants = make_sanity_events(events, seed=7)

    assert set(variants) == {"normal_GR", "shuffled_hidden_GR", "zero_hidden_GR"}
    assert np.allclose(variants["zero_hidden_GR"].iloc[0]["patch_gr"], 0.0)
    assert len(variants["shuffled_hidden_GR"]) == len(events)
```

- [ ] **Step 5: Implement sanity variants and event metrics**

Append:

```python
def make_sanity_events(events: pd.DataFrame, *, seed: int) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    normal = events.copy()
    shuffled = events.copy()
    patches = list(shuffled["patch_gr"])
    dpatches = list(shuffled["patch_dgr"])
    order = rng.permutation(len(shuffled))
    shuffled["patch_gr"] = [patches[i] for i in order]
    shuffled["patch_dgr"] = [dpatches[i] for i in order]
    zero = events.copy()
    zero["patch_gr"] = [np.zeros_like(np.asarray(p, dtype=np.float32)) for p in zero["patch_gr"]]
    zero["patch_dgr"] = [np.zeros_like(np.asarray(p, dtype=np.float32)) for p in zero["patch_dgr"]]
    return {
        "normal_GR": normal,
        "shuffled_hidden_GR": shuffled,
        "zero_hidden_GR": zero,
    }


def evaluate_event_matches(matches: pd.DataFrame, *, tolerance_ft: float = 10.0) -> dict[str, float]:
    if matches.empty or "true_TVT" not in matches:
        return {
            "events": 0,
            "event_top1_rmse_ft": float("nan"),
            "event_top10_oracle_rmse_ft": float("nan"),
            "event_true_top10_rate_at_10ft": 0.0,
        }
    rows = []
    for _, group in matches.groupby(["event_well_id", "event_step"], sort=False):
        true_tvt = float(group["true_TVT"].iloc[0])
        if not np.isfinite(true_tvt):
            continue
        sqerr = (pd.to_numeric(group["source_TVT"], errors="coerce").to_numpy(dtype=np.float64) - true_tvt) ** 2
        rows.append(
            {
                "top1_sqerr": float(sqerr[0]),
                "top10_sqerr": float(np.nanmin(sqerr)),
                "top10_hit": bool(np.nanmin(np.sqrt(sqerr)) <= tolerance_ft),
            }
        )
    if not rows:
        return {
            "events": 0,
            "event_top1_rmse_ft": float("nan"),
            "event_top10_oracle_rmse_ft": float("nan"),
            "event_true_top10_rate_at_10ft": 0.0,
        }
    frame = pd.DataFrame(rows)
    return {
        "events": int(len(frame)),
        "event_top1_rmse_ft": float(np.sqrt(frame["top1_sqerr"].mean())),
        "event_top10_oracle_rmse_ft": float(np.sqrt(frame["top10_sqerr"].mean())),
        "event_true_top10_rate_at_10ft": float(frame["top10_hit"].mean()),
    }
```

- [ ] **Step 6: Run match tests**

Run:

```bash
uv run --extra dev pytest -q tests/test_traceback.py::test_patch_score_ranks_exact_match_above_reverse_patch tests/test_traceback.py::test_make_sanity_events_supports_shuffled_and_zero_gr
```

Expected: both pass.

- [ ] **Step 7: Commit**

```bash
git add mtpnet/traceback_match.py tests/test_traceback.py
git commit -m "feat: score traceback event matches"
```

---

## Task 5: Bands, Candidate Paths, And Candidate-Bank Oracle

**Files:**
- Create: `mtpnet/traceback_candidates.py`
- Test: `tests/test_traceback.py`

- [ ] **Step 1: Write failing band/candidate tests**

Append:

```python
def test_traceback_band_generation_covers_anchor_steps() -> None:
    from mtpnet.traceback_candidates import build_traceback_bands

    comp = pd.DataFrame({"well_id": ["w"] * 5, "step": np.arange(5), "true_TVT": [100, 105, 110, 115, 120]})
    anchors = pd.DataFrame(
        {
            "well_id": ["w", "w"],
            "step": [1, 3],
            "anchor_tvt": [105.0, 115.0],
            "confidence": [1.0, 1.0],
        }
    )

    bands = build_traceback_bands(comp, anchors, widths_ft=(40.0,))

    assert len(bands) == 5
    assert bands["band_center_tvt"].between(100.0, 120.0).all()
    assert set(bands["band_width_ft"]) == {40.0}


def test_traceback_candidates_do_not_require_hidden_tvt() -> None:
    from mtpnet.traceback_candidates import build_traceback_candidates

    hidden = pd.DataFrame(
        {
            "id": [f"w_{i}" for i in range(4)],
            "well_id": ["w"] * 4,
            "row_idx": np.arange(4),
            "step": np.arange(4),
            "TVT_input": [np.nan] * 4,
        }
    )
    bands = pd.DataFrame(
        {
            "well_id": ["w"] * 4,
            "step": np.arange(4),
            "band_center_tvt": [100.0, 103.0, 106.0, 109.0],
            "band_width_ft": [80.0] * 4,
            "candidate": ["traceback_band_w80"] * 4,
        }
    )

    candidates = build_traceback_candidates(hidden, bands)

    assert {"id", "well_id", "row_idx", "candidate", "pred_tvt"}.issubset(candidates.columns)
    assert set(candidates["candidate"]) == {"traceback_band_w80"}
    assert "TVT" not in candidates.columns
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
uv run --extra dev pytest -q tests/test_traceback.py::test_traceback_band_generation_covers_anchor_steps tests/test_traceback.py::test_traceback_candidates_do_not_require_hidden_tvt
```

Expected: fail because `mtpnet.traceback_candidates` is missing.

- [ ] **Step 3: Implement bands and candidates**

Create `mtpnet/traceback_candidates.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd


def matches_to_anchors(matches: pd.DataFrame, *, min_score_quantile: float = 0.75) -> pd.DataFrame:
    if matches.empty:
        return pd.DataFrame(columns=["well_id", "step", "anchor_tvt", "confidence"])
    top = matches.sort_values("score", ascending=False).groupby(["event_well_id", "event_step"], as_index=False).head(1)
    threshold = float(top["score"].quantile(min_score_quantile)) if len(top) else float("inf")
    top = top[top["score"] >= threshold].copy()
    if top.empty:
        return pd.DataFrame(columns=["well_id", "step", "anchor_tvt", "confidence"])
    score = top["score"].to_numpy(dtype=np.float64)
    denom = max(float(np.nanmax(score) - np.nanmin(score)), 1e-6)
    conf = (score - float(np.nanmin(score))) / denom
    return pd.DataFrame(
        {
            "well_id": top["event_well_id"].astype(str).to_numpy(),
            "step": top["event_step"].astype(int).to_numpy(),
            "anchor_tvt": pd.to_numeric(top["source_TVT"], errors="coerce").to_numpy(dtype=np.float64),
            "confidence": np.clip(conf, 0.05, 1.0),
        }
    )


def build_traceback_bands(
    comp: pd.DataFrame,
    anchors: pd.DataFrame,
    *,
    widths_ft: tuple[float, ...] = (40.0, 80.0, 120.0),
) -> pd.DataFrame:
    rows = []
    for well_id, well in comp.groupby("well_id", sort=False):
        well = well.sort_values("step")
        well_anchors = anchors[anchors["well_id"].astype(str) == str(well_id)].sort_values("step")
        steps = well["step"].to_numpy(dtype=np.float64)
        if well_anchors.empty:
            center = np.full(len(well), np.nan, dtype=np.float64)
        else:
            a_steps = well_anchors["step"].to_numpy(dtype=np.float64)
            a_tvt = well_anchors["anchor_tvt"].to_numpy(dtype=np.float64)
            center = np.interp(steps, a_steps, a_tvt, left=a_tvt[0], right=a_tvt[-1])
        for width in widths_ft:
            for step, c in zip(well["step"].to_numpy(dtype=int), center):
                rows.append(
                    {
                        "well_id": str(well_id),
                        "step": int(step),
                        "band_center_tvt": float(c) if np.isfinite(c) else np.nan,
                        "band_width_ft": float(width),
                        "candidate": f"traceback_band_w{int(width)}",
                    }
                )
    return pd.DataFrame(rows)


def build_traceback_candidates(hidden_rows: pd.DataFrame, bands: pd.DataFrame) -> pd.DataFrame:
    if hidden_rows.empty or bands.empty:
        return pd.DataFrame(columns=["id", "well_id", "row_idx", "step", "candidate", "pred_tvt"])
    hidden = hidden_rows.copy()
    if "step" not in hidden:
        hidden["step"] = hidden["row_idx"] // 32
    keep = ["id", "well_id", "row_idx", "step"]
    out_parts = []
    for candidate, band in bands.groupby("candidate", sort=False):
        merged = hidden[keep].merge(
            band[["well_id", "step", "band_center_tvt"]],
            on=["well_id", "step"],
            how="left",
        )
        merged["candidate"] = str(candidate)
        merged["pred_tvt"] = pd.to_numeric(merged["band_center_tvt"], errors="coerce")
        out_parts.append(merged[["id", "well_id", "row_idx", "step", "candidate", "pred_tvt"]])
    return pd.concat(out_parts, ignore_index=True)
```

- [ ] **Step 4: Run band/candidate tests**

Run:

```bash
uv run --extra dev pytest -q tests/test_traceback.py::test_traceback_band_generation_covers_anchor_steps tests/test_traceback.py::test_traceback_candidates_do_not_require_hidden_tvt
```

Expected: both pass.

- [ ] **Step 5: Commit**

```bash
git add mtpnet/traceback_candidates.py tests/test_traceback.py
git commit -m "feat: generate traceback bands and candidates"
```

---

## Task 6: CLI Orchestrator, Artifacts, And Report

**Files:**
- Create: `mtpnet/traceback.py`
- Modify: `Makefile`
- Test: `tests/test_traceback.py`

- [ ] **Step 1: Write failing smoke test**

Append:

```python
def test_traceback_smoke_writes_metrics_report_and_artifacts(tmp_path) -> None:
    from mtpnet.traceback import run_traceback_from_frames

    rows = []
    for well_id, offset in [("a", 0.0), ("b", 20.0)]:
        for i in range(16):
            tvt = 100.0 + offset + i
            rows.append(
                {
                    "id": f"{well_id}_{i}",
                    "well_id": well_id,
                    "row_idx": i,
                    "MD": 1000.0 + i,
                    "X": float(i),
                    "Y": 0.0,
                    "Z": -float(i),
                    "GR": [0, 1, 5, 1, 0, -1, -4, -1, 0, 1, 4, 1, 0, -1, -3, -1][i],
                    "TVT_input": tvt if i < 4 else np.nan,
                    "TVT": tvt,
                }
            )
    frame = pd.DataFrame(rows)
    typewell = pd.DataFrame({"TVT": np.linspace(90, 140, 20), "GR": np.sin(np.linspace(0, 8, 20))})

    metrics = run_traceback_from_frames(frame, typewell=typewell, output_dir=tmp_path, k_wells=-1, seed=3)

    assert metrics["wells"] == 2
    assert "normal_GR" in metrics["variants"]
    assert (tmp_path / "traceback_metrics.json").exists()
    assert (tmp_path / "traceback_report.md").exists()
    assert (tmp_path / "traceback_events.parquet").exists()
    assert (tmp_path / "traceback_event_matches.parquet").exists()
```

- [ ] **Step 2: Run smoke test and verify it fails**

Run:

```bash
uv run --extra dev pytest -q tests/test_traceback.py::test_traceback_smoke_writes_metrics_report_and_artifacts
```

Expected: fail because `mtpnet.traceback` is missing.

- [ ] **Step 3: Implement minimal orchestrator**

Create `mtpnet/traceback.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .correlation_panel import _regular_typewell_grid
from .heatmap import fill_nan
from .io import discover_wells, load_well
from .residual_stack import load_training_frame
from .traceback_candidates import build_traceback_bands, build_traceback_candidates, matches_to_anchors
from .traceback_dictionary import (
    build_fold_safe_train_dictionary,
    build_same_well_dictionary,
    build_typewell_dictionary,
)
from .traceback_events import TracebackConfig, compress_well_rows, extract_traceback_events
from .traceback_match import evaluate_event_matches, make_sanity_events, score_event_against_dictionary


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _first_typewell_from_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if {"TVT", "GR"}.issubset(frame.columns):
        valid = frame[pd.to_numeric(frame["TVT"], errors="coerce").notna()].copy()
        return valid[["TVT", "GR"]].dropna(subset=["TVT"]).sort_values("TVT")
    return pd.DataFrame({"TVT": [], "GR": []})


def run_traceback_from_frames(
    frame: pd.DataFrame,
    *,
    typewell: pd.DataFrame,
    output_dir: Path,
    k_wells: int = -1,
    seed: int = 42,
    cfg: TracebackConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or TracebackConfig(seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    wells = sorted(frame["well_id"].astype(str).unique())
    if k_wells > 0:
        wells = wells[:k_wells]
    frame = frame[frame["well_id"].astype(str).isin(wells)].copy()

    comp_parts = []
    event_parts = []
    for well_id, well in frame.groupby("well_id", sort=False):
        comp = compress_well_rows(well, rows_per_step=cfg.rows_per_step)
        comp_parts.append(comp)
        events = extract_traceback_events(comp, cfg)
        event_parts.append(events)
    comp_all = pd.concat(comp_parts, ignore_index=True) if comp_parts else pd.DataFrame()
    events_all = pd.concat(event_parts, ignore_index=True) if event_parts else pd.DataFrame()

    same_dict = build_same_well_dictionary(events_all)
    train_dict = build_fold_safe_train_dictionary(events_all, train_wells=wells)
    typewell_dict = build_typewell_dictionary(
        typewell,
        vertical_step_ft=cfg.vertical_step_ft,
        patch_radii=cfg.patch_radii,
        min_prominence_z=cfg.min_prominence_z,
    )
    dictionary = pd.concat([same_dict, typewell_dict, train_dict], ignore_index=True)

    variants = make_sanity_events(events_all, seed=seed) if not events_all.empty else {}
    variant_metrics: dict[str, Any] = {}
    match_parts = []
    for variant_name, variant_events in variants.items():
        rows = []
        for _, event in variant_events.iterrows():
            scored = score_event_against_dictionary(
                event,
                dictionary,
                location_weight=0.0,
                top_k=cfg.top_k_matches,
            )
            if scored.empty:
                continue
            scored["variant"] = variant_name
            scored["true_TVT"] = float(event["true_TVT"]) if "true_TVT" in event and np.isfinite(event["true_TVT"]) else np.nan
            rows.append(scored)
        matches = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        variant_metrics[variant_name] = evaluate_event_matches(matches)
        if not matches.empty:
            match_parts.append(matches)
    matches_all = pd.concat(match_parts, ignore_index=True) if match_parts else pd.DataFrame()

    normal_matches = matches_all[matches_all["variant"] == "normal_GR"].copy() if not matches_all.empty else pd.DataFrame()
    anchors = matches_to_anchors(normal_matches) if not normal_matches.empty else pd.DataFrame()
    bands = build_traceback_bands(comp_all, anchors) if not comp_all.empty else pd.DataFrame()
    hidden = frame[pd.to_numeric(frame["TVT_input"], errors="coerce").isna()].copy()
    candidates = build_traceback_candidates(hidden, bands) if not hidden.empty else pd.DataFrame()

    events_all.to_parquet(output_dir / "traceback_events.parquet", index=False)
    dictionary.to_parquet(output_dir / "traceback_dictionary.parquet", index=False)
    matches_all.to_parquet(output_dir / "traceback_event_matches.parquet", index=False)
    bands.to_parquet(output_dir / "traceback_bands.parquet", index=False)
    candidates.to_parquet(output_dir / "traceback_candidates.parquet", index=False)

    metrics = {
        "wells": int(len(wells)),
        "compressed_steps": int(len(comp_all)),
        "events": int(len(events_all)),
        "dictionary_entries": int(len(dictionary)),
        "matches": int(len(matches_all)),
        "candidate_rows": int(len(candidates)),
        "variants": variant_metrics,
    }
    (output_dir / "traceback_metrics.json").write_text(json.dumps(_json_safe(metrics), indent=2), encoding="utf-8")
    report = [
        "TRACEBACK_V0_REPORT",
        "",
        "summary:",
        "```json",
        json.dumps(_json_safe(metrics), indent=2),
        "```",
    ]
    (output_dir / "traceback_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/traceback_v0"))
    parser.add_argument("--rows-per-step", type=int, default=32)
    parser.add_argument("--vertical-step-ft", type=float, default=5.0)
    parser.add_argument("--patch-radii", default="3,5,9,15")
    parser.add_argument("--k-wells", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frame = load_training_frame(args.data_dir)
    typewell = _first_typewell_from_frame(frame)
    cfg = TracebackConfig(
        rows_per_step=args.rows_per_step,
        vertical_step_ft=args.vertical_step_ft,
        patch_radii=tuple(int(v) for v in args.patch_radii.split(",") if v),
        seed=args.seed,
    )
    metrics = run_traceback_from_frames(
        frame,
        typewell=typewell,
        output_dir=args.output_dir,
        k_wells=args.k_wells,
        seed=args.seed,
        cfg=cfg,
    )
    print(json.dumps(_json_safe(metrics), indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Add Makefile target**

Modify `Makefile`:

```make
TRACEBACK_OUTPUT ?= artifacts/traceback_v0
TRACEBACK_K_WELLS ?= -1
TRACEBACK_ROWS_PER_STEP ?= 32
TRACEBACK_VERTICAL_STEP_FT ?= 5.0
TRACEBACK_PATCH_RADII ?= 3,5,9,15
```

Add `traceback` to `.PHONY`.

Add target:

```make
traceback:
	$(UV) run --extra dev python -m mtpnet.traceback \
		--data-dir $(COPY_TARGET)/train \
		--output-dir $(TRACEBACK_OUTPUT) \
		--rows-per-step $(TRACEBACK_ROWS_PER_STEP) \
		--vertical-step-ft $(TRACEBACK_VERTICAL_STEP_FT) \
		--patch-radii $(TRACEBACK_PATCH_RADII) \
		--k-wells $(TRACEBACK_K_WELLS) \
		--seed $(SEED)
```

- [ ] **Step 5: Run smoke test**

Run:

```bash
uv run --extra dev pytest -q tests/test_traceback.py::test_traceback_smoke_writes_metrics_report_and_artifacts
```

Expected: pass.

- [ ] **Step 6: Run Make smoke**

Run:

```bash
make traceback TRACEBACK_OUTPUT=artifacts/traceback_smoke TRACEBACK_K_WELLS=20 TRACEBACK_PATCH_RADII=3,5
```

Expected: writes `traceback_metrics.json`, `traceback_report.md`, and parquet artifacts under `artifacts/traceback_smoke`.

- [ ] **Step 7: Commit**

```bash
git add mtpnet/traceback.py Makefile tests/test_traceback.py
git commit -m "feat: add traceback cli and artifacts"
```

---

## Task 7: Candidate-Bank Oracle And Report Figures

**Files:**
- Modify: `mtpnet/traceback_candidates.py`
- Modify: `mtpnet/traceback.py`
- Test: `tests/test_traceback.py`

- [ ] **Step 1: Write failing candidate oracle test**

Append:

```python
def test_traceback_candidate_oracle_improves_when_traceback_candidate_is_exact(tmp_path) -> None:
    from mtpnet.traceback_candidates import evaluate_traceback_candidate_oracle

    hidden = pd.DataFrame(
        {
            "id": ["a", "b"],
            "well_id": ["w", "w"],
            "row_idx": [0, 1],
            "TVT": [100.0, 110.0],
            "b2_tvt": [120.0, 130.0],
        }
    )
    traceback_candidates = pd.DataFrame(
        {
            "id": ["a", "b"],
            "well_id": ["w", "w"],
            "row_idx": [0, 1],
            "candidate": ["traceback_exact", "traceback_exact"],
            "pred_tvt": [100.0, 110.0],
        }
    )

    metrics = evaluate_traceback_candidate_oracle(hidden, traceback_candidates)

    assert metrics["b2_row_rmse"] > 0.0
    assert metrics["traceback_oracle_row_rmse"] == 0.0
    assert metrics["oracle_gain_ft"] > 0.0
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```bash
uv run --extra dev pytest -q tests/test_traceback.py::test_traceback_candidate_oracle_improves_when_traceback_candidate_is_exact
```

Expected: fail because `evaluate_traceback_candidate_oracle` is undefined.

- [ ] **Step 3: Implement candidate oracle**

Append to `mtpnet/traceback_candidates.py`:

```python
def _rmse(pred: np.ndarray, target: np.ndarray) -> float:
    finite = np.isfinite(pred) & np.isfinite(target)
    if not finite.any():
        return float("nan")
    return float(np.sqrt(np.mean((pred[finite] - target[finite]) ** 2)))


def evaluate_traceback_candidate_oracle(hidden_rows: pd.DataFrame, traceback_candidates: pd.DataFrame) -> dict[str, float | int]:
    if hidden_rows.empty:
        return {"rows": 0, "b2_row_rmse": float("nan"), "traceback_oracle_row_rmse": float("nan"), "oracle_gain_ft": 0.0}
    truth = hidden_rows[["id", "TVT", "b2_tvt"]].copy()
    b2_rmse = _rmse(truth["b2_tvt"].to_numpy(dtype=np.float64), truth["TVT"].to_numpy(dtype=np.float64))
    if traceback_candidates.empty:
        return {"rows": int(len(hidden_rows)), "b2_row_rmse": b2_rmse, "traceback_oracle_row_rmse": float("nan"), "oracle_gain_ft": 0.0}
    merged = traceback_candidates.merge(truth[["id", "TVT"]], on="id", how="inner")
    merged["sqerr"] = (pd.to_numeric(merged["pred_tvt"], errors="coerce") - pd.to_numeric(merged["TVT"], errors="coerce")) ** 2
    best = merged.groupby("id", as_index=False)["sqerr"].min()
    oracle_rmse = float(np.sqrt(best["sqerr"].mean())) if not best.empty else float("nan")
    return {
        "rows": int(len(hidden_rows)),
        "candidate_rows": int(len(traceback_candidates)),
        "b2_row_rmse": b2_rmse,
        "traceback_oracle_row_rmse": oracle_rmse,
        "oracle_gain_ft": float(b2_rmse - oracle_rmse) if np.isfinite(oracle_rmse) else 0.0,
    }
```

- [ ] **Step 4: Add oracle metrics to CLI report**

Modify `run_traceback_from_frames` in `mtpnet/traceback.py`:

```python
from .traceback_candidates import (
    build_traceback_bands,
    build_traceback_candidates,
    evaluate_traceback_candidate_oracle,
    matches_to_anchors,
)
```

After `candidates` are built:

```python
oracle_metrics = evaluate_traceback_candidate_oracle(hidden, candidates) if "TVT" in hidden.columns else {}
```

Add to `metrics`:

```python
"candidate_oracle": oracle_metrics,
```

Write `traceback_candidate_oracle.csv`:

```python
pd.DataFrame([oracle_metrics]).to_csv(output_dir / "traceback_candidate_oracle.csv", index=False)
```

- [ ] **Step 5: Run oracle test and smoke**

Run:

```bash
uv run --extra dev pytest -q tests/test_traceback.py::test_traceback_candidate_oracle_improves_when_traceback_candidate_is_exact tests/test_traceback.py::test_traceback_smoke_writes_metrics_report_and_artifacts
```

Expected: both pass.

- [ ] **Step 6: Run full traceback tests**

Run:

```bash
uv run --extra dev pytest -q tests/test_traceback.py
```

Expected: all TraceBack tests pass.

- [ ] **Step 7: Commit**

```bash
git add mtpnet/traceback_candidates.py mtpnet/traceback.py tests/test_traceback.py
git commit -m "feat: evaluate traceback candidate oracle"
```

---

## Task 8: Full Verification And First Diagnostic Run

**Files:**
- All TraceBack files
- `Makefile`

- [ ] **Step 1: Run all TraceBack tests**

Run:

```bash
uv run --extra dev pytest -q tests/test_traceback.py
```

Expected: all TraceBack tests pass.

- [ ] **Step 2: Run full repo tests**

Run:

```bash
uv run --extra dev pytest -q
```

Expected: full suite passes. Existing unrelated warnings are acceptable if they match current baseline warning categories.

- [ ] **Step 3: Run 80-well diagnostic**

Run:

```bash
make traceback TRACEBACK_OUTPUT=artifacts/traceback_v0_k80 TRACEBACK_K_WELLS=80 TRACEBACK_PATCH_RADII=3,5,9
```

Expected outputs:

```text
artifacts/traceback_v0_k80/traceback_metrics.json
artifacts/traceback_v0_k80/traceback_report.md
artifacts/traceback_v0_k80/traceback_event_matches.parquet
artifacts/traceback_v0_k80/traceback_candidates.parquet
artifacts/traceback_v0_k80/traceback_candidate_oracle.csv
```

- [ ] **Step 4: Read diagnostic metrics**

Run:

```bash
uv run --extra dev python - <<'PY'
import json
from pathlib import Path

metrics = json.loads(Path("artifacts/traceback_v0_k80/traceback_metrics.json").read_text())
print(json.dumps(metrics.get("variants", {}), indent=2))
print(json.dumps(metrics.get("candidate_oracle", {}), indent=2))
PY
```

Expected: prints normal/shuffled/zero event metrics and candidate oracle metrics.

- [ ] **Step 5: Apply decision rule**

Use this exact interpretation:

```text
GO:
  normal_GR event top10 beats shuffled_hidden_GR by >= 5 percentage points
  and candidate_oracle.oracle_gain_ft > 0

Weak GO:
  normal_GR beats shuffled by >= 3 percentage points
  or candidate_oracle.oracle_gain_ft > 0.05 on k80

NO-GO:
  normal_GR ~= shuffled_hidden_GR
  and candidate_oracle.oracle_gain_ft <= 0
```

- [ ] **Step 6: Commit final TraceBack implementation**

```bash
git add mtpnet/traceback*.py tests/test_traceback.py Makefile
git commit -m "feat: add traceback local segment dictionary experiment"
```

---

## Self-Review Checklist

- Spec coverage:
  - Event extraction: Task 2.
  - Same-well/typewell/fold-safe dictionaries: Task 3.
  - Normal/shuffled/zero sanity variants: Task 4.
  - Band and candidate generation: Task 5.
  - Artifacts/report/CLI/Make: Task 6.
  - Candidate oracle: Task 7.
  - Verification and decision rule: Task 8.

- Anti-leak coverage:
  - Schema guard in Task 1.
  - Same-well dictionary uses only known `TVT_input` in Task 3.
  - Fold-safe train dictionary excludes validation wells in Task 3.
  - Candidate generation test verifies hidden `TVT` is not required in Task 5.
  - Candidate oracle is diagnostic-only in Task 7.

- Known first-version limitation:
  - The initial CLI uses train events as dictionary entries in a simplified way for smoke and first audit. Before using results as final OOF, add explicit fold iteration around dictionaries so validation wells never contribute entries to their own train-well dictionary. Same-well known dictionary remains allowed.

