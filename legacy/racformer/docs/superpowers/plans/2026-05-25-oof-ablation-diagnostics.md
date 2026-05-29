# OOF Ablation Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a post-training OOF ablation diagnostics runner for RAC-Former.

**Architecture:** Put pure metric and row-assembly logic in `oof_diagnostics.py`, call it from `infer.py`, and expose a Makefile target. Tests cover the pure functions with synthetic tensors and samples.

**Tech Stack:** Python 3.12, PyTorch, NumPy, pandas, pytest, ruff.

---

### Task 1: Pure Diagnostics Functions

**Files:**
- Create: `oof_diagnostics.py`
- Create: `tests/test_oof_diagnostics.py`

- [ ] Write failing tests for RMSE, bucket summaries, row-level variant assembly, and ablated materialization.
- [ ] Implement minimal pure functions in `oof_diagnostics.py`.
- [ ] Run `uv run pytest tests/test_oof_diagnostics.py -q`.

### Task 2: OOF Runner Integration

**Files:**
- Modify: `infer.py`
- Modify: `Makefile`

- [ ] Add `run_oof_diagnostics` that mirrors OOF fold loading and emits row CSV plus summary JSON.
- [ ] Add `--oof_diagnostics`, `--oof_diag_path`, and `--oof_summary_path` CLI args.
- [ ] Add `OOF_DIAG_PATH`, `OOF_SUMMARY_PATH`, and `oof-diagnostics` Makefile target.

### Task 3: Verification

**Files:**
- All changed files.

- [ ] Run focused pytest.
- [ ] Run ruff check.
- [ ] Inspect git diff for unrelated edits.
