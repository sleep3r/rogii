# ANCC Top Teacher Distillation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add train-time ANCC formation teacher distillation to RAC-Former.

**Architecture:** `dataset.py` creates step-level labels from train-only ANCC, `model.py` exposes event and direction logits, and `loss.py` consumes masked labels with configurable weights. Existing inference inputs remain unchanged.

**Tech Stack:** Python 3.12, NumPy, pandas, PyTorch, pytest.

---

### Task 1: Teacher Labels

**Files:**
- Modify: `dataset.py`
- Test: `tests/test_ancc_teacher.py`

- [ ] Write failing tests for ANCC d-step aggregation and `RACDataset` batch keys.
- [ ] Add `top_teacher_eps`, `top_state_step`, `top_event_step`, and `top_teacher_mask`.
- [ ] Verify focused tests pass.

### Task 2: Model and Loss

**Files:**
- Modify: `model.py`
- Modify: `loss.py`
- Modify: `config.py`
- Test: `tests/test_top_distillation_loss.py`

- [ ] Write failing tests for output shapes and masked loss math.
- [ ] Add 3-class direction logits and top loss terms.
- [ ] Add `w_top_event` and `w_top_dir` defaults/config entries.
- [ ] Verify focused tests pass.

### Task 3: Configs and Verification

**Files:**
- Modify: `configs/*.yml`

- [ ] Add explicit top-loss weights to configs.
- [ ] Run focused pytest and import/smoke checks.
