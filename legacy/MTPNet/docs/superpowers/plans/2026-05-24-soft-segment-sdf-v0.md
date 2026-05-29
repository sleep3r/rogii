# Soft Segment SDF v0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a schema-safe MVP that trains a dense path posterior from raw/plane-coordinate heatmap channels and decodes it with smooth DP.

**Architecture:** Implement `mtpnet.soft_segment` as a focused module: config, sample/channel builder, small CNN segmentation model, vertical-KL loss, Viterbi decode, train/eval artifact writer. Reuse existing train frame loaders and schema-safe conventions; do not use forbidden geology/surface columns as model inputs.

**Tech Stack:** Python, PyTorch, NumPy, Pandas, PyArrow parquet, pytest, Makefile.

---

### Task 1: Dataset Channels And Targets

**Files:**
- Create: `mtpnet/soft_segment.py`
- Test: `tests/test_soft_segment.py`

- [ ] Write tests for `build_soft_segment_sample` producing `[C,H,W]`, finite channels, and target distribution summing to one per step.
- [ ] Implement `SoftSegmentConfig`, `SoftSegmentSample`, `vertical_soft_target`, and `build_soft_segment_sample`.
- [ ] Verify forbidden columns are never included as feature column names.

### Task 2: Loss And DP Decode

**Files:**
- Modify: `mtpnet/soft_segment.py`
- Test: `tests/test_soft_segment.py`

- [ ] Write tests that vertical KL is lower when logits peak near the target.
- [ ] Write tests that DP recovers a synthetic diagonal ridge and smooths a one-step spike.
- [ ] Implement `vertical_kl_loss`, `viterbi_decode_logprobs`, and path-to-TVt conversion helpers.

### Task 3: Model Forward

**Files:**
- Modify: `mtpnet/soft_segment.py`
- Test: `tests/test_soft_segment.py`

- [ ] Write test that `SoftSegmentNet` returns logits shaped `[B,H,W]`.
- [ ] Implement a small encoder-decoder CNN with native-resolution output.

### Task 4: Train/Eval CLI Smoke

**Files:**
- Modify: `mtpnet/soft_segment.py`
- Modify: `Makefile`
- Create: `configs/soft_segment_smoke.yml`
- Create: `configs/soft_segment_v0.yml`
- Test: `tests/test_soft_segment.py`

- [ ] Write tiny end-to-end smoke test that trains for one epoch and writes `metrics.json`, `window_predictions.parquet`, `report.md`.
- [ ] Implement `run_soft_segment`, argparse CLI, YAML loading, train/valid split, metrics and artifact writing.
- [ ] Add `make soft-segment-smoke` and `make soft-segment-train`.

### Task 5: Verification

- [ ] Run `uv run --extra dev pytest tests/test_soft_segment.py -q`.
- [ ] Run `uv run --extra dev pytest -q`.
- [ ] Run `make soft-segment-smoke`.
- [ ] Run `make codebase-bundle` and verify `rg -n "research_101|resources" artifacts/codebase_bundle.md` returns no matches.

