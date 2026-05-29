# Template Envelope v0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create an isolated `promising/template_envelope_v0` experiment that exports test-safe envelope candidates from scaled/offset typewell GR templates.

**Architecture:** The package is split into small modules for config, data compression, template generation/scoring, envelope construction, and CLI/report writing. It outputs candidate parquet files compatible with the existing candidate-bank/chunk-policy external candidate path.

**Tech Stack:** Python, NumPy, pandas, pyarrow parquet through pandas, pytest, Makefile target.

---

### Task 1: Tests

**Files:**
- Create: `tests/test_template_envelope_v0.py`

- [ ] Write tests for schema guard, template path construction, score sanity, envelope construction, smoke artifacts, and no target leakage in candidate output.
- [ ] Run focused tests and verify they fail because package/functions are missing.

### Task 2: Package

**Files:**
- Create: `promising/__init__.py`
- Create: `promising/template_envelope_v0/__init__.py`
- Create: `promising/template_envelope_v0/config.py`
- Create: `promising/template_envelope_v0/data.py`
- Create: `promising/template_envelope_v0/templates.py`
- Create: `promising/template_envelope_v0/scoring.py`
- Create: `promising/template_envelope_v0/envelope.py`
- Create: `promising/template_envelope_v0/cli.py`
- Create: `promising/template_envelope_v0/README.md`

- [ ] Implement minimal code to pass tests.
- [ ] Keep candidate output schema-safe.

### Task 3: Make Target

**Files:**
- Modify: `Makefile`

- [ ] Add variables for template envelope output/grid.
- [ ] Add `template-envelope` target.

### Task 4: Verification

- [ ] Run `uv run --extra dev pytest tests/test_template_envelope_v0.py -q`.
- [ ] Run a smoke CLI on tiny synthetic test through pytest.
- [ ] Run `uv run --extra dev pytest -q` if focused tests pass.

