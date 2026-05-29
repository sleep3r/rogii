# Old Cleanup Leakfix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `old/` a small standalone ROGII project with test-safe training defaults, a readable Makefile, and a compact supported server path.

**Architecture:** Keep the current Python package layout, but remove Makefile support for stale research workflows. Keep Docker/spacebridge as a separate server block. Close train-test leakage by making direct-path features ignore current-well formation columns unless an explicit research flag enables them.

**Tech Stack:** Python 3.12, uv, pandas/numpy/scikit-learn/GBM libraries, pytest, ruff, Make.

---

### Task 1: Direct-Path Leakage Guard

**Files:**
- Modify: `rogii/path_features.py`
- Modify: `tests/test_path_features.py`

- [ ] Write or keep a failing regression test that perturbs hidden-row `ANCC/.../BUDA` columns and asserts direct-path features do not change by default.
- [ ] Run `uv run --group dev pytest tests/test_path_features.py::test_direct_path_features_ignore_current_well_formations_by_default -q` and confirm the test fails before the fix.
- [ ] Add a direct-path config gate named `allow_current_well_formations`; default it to `false`.
- [ ] When the gate is false, pass a copy of the horizontal frame with formation columns removed to the direct solver path builders.
- [ ] Keep an explicit opt-in path for research by setting `features.direct_path.allow_current_well_formations: true`.
- [ ] Run `uv run --group dev pytest tests/test_path_features.py -q`.

### Task 2: Test-Safe Config Defaults

**Files:**
- Modify: `configs/stack.yml`
- Modify: `configs/quick.yml`
- Modify: `configs/stack_gpu.yml`

- [ ] Ensure `features.direct_path.allow_current_well_formations: false` is explicit in production configs.
- [ ] Keep `top_state_predictions` because it is a test-safe distilled teacher artifact, not a raw train formation feature.
- [ ] Disable or remove stale HMM/drift configs from the supported Makefile path.

### Task 3: Simple Standalone Makefile With Server Block

**Files:**
- Replace: `Makefile`

- [ ] Replace the broad research Makefile with a compact interface: `help`, `install`, `data`, `train`, `train-gpu`, `train-server`, `quick`, `profile`, `submit`, `check`, `format`, `clean-cache`, `doctor`, `server-doctor`.
- [ ] Route all training through `uv run python -m rogii --config ...`.
- [ ] Keep server plumbing in a small dedicated block, not mixed with local defaults.
- [ ] Add a `help` target that documents the supported commands.

### Task 4: Project Hygiene

**Files:**
- Modify: `README.md`
- Modify: `.gitignore`
- Modify: `.dockerignore`
- Keep: `Dockerfile`
- Keep: `rogii/spacebridge_preflight.py`
- Create: `portainer.yml.example`

- [ ] Rewrite README around the standalone local workflow.
- [ ] Ignore `.DS_Store`, caches, artifacts, data, `catboost_info`, and generated submissions.
- [ ] Keep source code, configs, tests, `pyproject.toml`, and `uv.lock` as the project surface.

### Task 5: Verification

**Files:**
- Read: changed files

- [ ] Run `uv run --group dev pytest tests/test_path_features.py tests/test_modeling_blend.py -q`.
- [ ] Run `uv run --group dev ruff check .`.
- [ ] Run `make help`.
- [ ] Report exact commands and any remaining failures.
