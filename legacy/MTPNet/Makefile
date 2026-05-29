SHELL := /bin/bash
PYTHON ?= python3
UV ?= uv
CONFIG ?= configs/mtp_v0.yml
SMOKE_CONFIG ?= configs/mtp_smoke.yml
RUN_DIR ?= artifacts/mtp_v0
OUTPUT ?=
MAX_FOLDS ?=
N_FOLDS ?= 5
SEED ?= 42
RANKER_VARIANT ?= conservative_regression
LOGIT_SOURCE ?= ranker
RANKER_LOGITS ?=
TAU_FT ?= 5.0
RANKER_BETA ?= 0.5
CORR_BETA ?=
FULL_STITCH ?=
COPY_SOURCE ?= ../old/data
COPY_TARGET ?= data
CODEBASE_BUNDLE ?= artifacts/codebase_bundle.md
CANDIDATES ?=
TOP_N ?= 30
PRIMARY_CANDIDATE ?=
CORR_OUTPUT ?= artifacts/corr_panel_v0
CORR_ROWS_PER_STEP ?= 32
CORR_VERTICAL_STEP_FT ?= 5.0
CORR_PATCH_RADIUS ?= 3
CORR_PATCH_RADII ?=
CORR_STRETCH_FACTORS ?= 1.0
CORR_ANCHOR_SOURCE ?= none
CORR_SEARCH_RADIUS_FT ?=
CORR_K_WELLS ?= -1
CORR_NO_SHUFFLED ?=
CONDITIONAL_CORR_OUTPUT ?= artifacts/conditional_corr_panel_v0
CONDITIONAL_CORR_LOCATION_SIGMA_FT ?= 80.0
CONDITIONAL_CORR_GR_WEIGHT ?= 1.0
CONDITIONAL_CORR_LOCATION_WEIGHT ?= 1.0
TYPEWELL_MISMATCH_OUTPUT ?= artifacts/typewell_mismatch_v0
TRUE_PATH_GR_OUTPUT ?= artifacts/true_path_gr_audit_v0
PLANE_COORD_OUTPUT ?= artifacts/plane_coordinate_audit_v0
TARGET_SHOOTOUT_OUTPUT ?= artifacts/target_shootout_v0
SOFT_SEGMENT_CONFIG ?= configs/soft_segment_v0.yml
SOFT_SEGMENT_SMOKE_CONFIG ?= configs/soft_segment_smoke.yml
SOFT_SEGMENT_V1_CONFIG ?= configs/soft_segment_v1_xcorr.yml
LOCATION_AWARE_CORR_OUTPUT ?= artifacts/location_aware_corr_v0
LOCATION_AWARE_SIGMA_FT ?= 80.0
LOCATION_AWARE_GR_WEIGHT ?= 1.0
LOCATION_AWARE_LOCATION_WEIGHT ?= 1.0
SHOOTOUT_N_FOLDS ?= 5
SHOOTOUT_N_ESTIMATORS ?= 500
SHOOTOUT_LEARNING_RATE ?= 0.05
SHOOTOUT_NUM_LEAVES ?= 64
TAIL_CLASSES_PATH ?= artifacts/tail_audit_v1/well_tail_audit.csv
FORMATION_CORR_OUTPUT ?= artifacts/formation_correlation_v0
PSEUDO_ZONE_OUTPUT ?= artifacts/pseudo_zone_v0
PSEUDO_ZONE_V1_OUTPUT ?= artifacts/pseudo_zone_v1
PSEUDO_ZONE_BINS ?= 128
PSEUDO_ZONE_EPOCHS ?= 6
PSEUDO_ZONE_HIDDEN_DIM ?= 64
PSEUDO_ZONE_BATCH_SIZE ?= 2048
PSEUDO_ZONE_LR ?= 0.001
PSEUDO_ZONE_KNOWN_TAIL_RADIUS_FT ?= 120.0
PSEUDO_ZONE_SOFT_WEIGHT ?= 0.5
DIAGNOSTIC_OUTPUT ?= artifacts/diagnostic_report_v1
DIAGNOSTIC_PDF ?= artifacts/diagnostic_report_v1/diagnostic_report.pdf

# PathFormer config overrides
PF_CONFIG ?= configs/pathformer_v0.yml
PF_SMOKE_CONFIG ?= configs/pathformer_smoke.yml
PF_EPOCHS ?=

# GeoAligner config overrides
GA_CONFIG ?= configs/geoaligner_v0.yml
GA_SMOKE_CONFIG ?= configs/geoaligner_smoke.yml
GA_EPOCHS ?=
GA_CHECKPOINT ?=

# PathPolicy oracle overrides
ORACLE_OUTPUT ?= artifacts/oracle_v0
ORACLE_K_WELLS ?= -1

# PathPolicy train/eval overrides
PP_CONFIG ?= configs/pathpolicy_v6.yml
PP_SMOKE_CONFIG ?= configs/pathpolicy_v6_smoke.yml
PP_CHECKPOINT ?= artifacts/pathpolicy_v6/best.pt
PP_OUTPUT ?= artifacts/pathpolicy_v6
PP_K_WELLS ?= -1
PP_COSTGBM_CHUNKS ?= artifacts/oracle_v1/oracle_chunks.parquet
PP_COSTGBM_OUTPUT ?= artifacts/costgbm_v1
RESIDUAL_OUTPUT ?= artifacts/residual_stack_v0
RESIDUAL_K_WELLS ?= -1
RESIDUAL_ROWS_PER_STEP ?= 32
RESIDUAL_FOLDS ?= 5
RESIDUAL_ITERATIONS ?= 700
RESIDUAL_LR ?= 0.04
RESIDUAL_DEPTH ?= 6
CANDIDATE_BANK_OUTPUT ?= artifacts/candidate_bank_v2
RESIDUAL_PREDICTIONS ?= $(RESIDUAL_OUTPUT)/oof_predictions.parquet
TRACEBACK_CANDIDATES ?=
CANDIDATE_SELECTOR_OUTPUT ?= artifacts/candidate_selector_v0
CANDIDATE_SELECTOR_V1_OUTPUT ?= artifacts/candidate_selector_v1_ranker_guard
CANDIDATE_SELECTOR_FOLDS ?= 5
CANDIDATE_SELECTOR_ITERATIONS ?= 900
CANDIDATE_SELECTOR_LR ?= 0.035
CANDIDATE_SELECTOR_DEPTH ?= 5
CANDIDATE_SELECTOR_PROGRESS_EVERY ?= 50
CANDIDATE_SELECTOR_GUARD_GAIN ?= 0.0
CANDIDATE_SELECTOR_DANGEROUS_GUARD_GAIN ?= 5.0
CHUNK_POLICY_OUTPUT ?= artifacts/chunk_ranker_dp_v1
CHUNK_POLICY_CHUNK_SIZE ?= 128
CHUNK_POLICY_ITERATIONS ?= 700
CHUNK_POLICY_LR ?= 0.04
CHUNK_POLICY_DEPTH ?= 5
CHUNK_POLICY_COST_TARGET ?= log_mse
CHUNK_POLICY_RANKER_TARGET ?= neg_log1p_regret
CHUNK_POLICY_RANKER_ALPHA ?= 5.0
CHUNK_POLICY_SWITCH_PENALTY ?= 500.0
CHUNK_POLICY_JUMP_PENALTY ?= 1.0
CHUNK_POLICY_SELFCAL_PROBES ?= 4
CHUNK_POLICY_USE_SHARED_TYPEWELL ?=
CHUNK_POLICY_SHARED_TYPEWELL_NEIGHBOURS ?= artifacts/shared_typewell_neighbour_v0/typewell_neighbours.parquet
CHUNK_POLICY_STEP_PREDICTIONS ?= artifacts/soft_segment_v1_xcorr/window_predictions.parquet
CHUNK_POLICY_LOCATION_AWARE_STEPS ?= artifacts/location_aware_corr_v0/location_aware_corr_steps.parquet
CHUNK_POLICY_EXTRA_ARGS ?=
POLICY_SOLVER_OUTPUT ?= artifacts/policy_solver_v0
SHARED_TYPEWELL_OUTPUT ?= artifacts/shared_typewell_neighbour_v0
SHARED_TYPEWELL_K_WELLS ?= -1
SHARED_TYPEWELL_FOLDS ?= 5
SHARED_TYPEWELL_TOP_K ?= 8
SHARED_TYPEWELL_SIGNATURE_BINS ?= 256
SHARED_TYPEWELL_MAX_SHIFT_BINS ?= 32
TRACEBACK_OUTPUT ?= artifacts/traceback_v0
TRACEBACK_K_WELLS ?= -1
TRACEBACK_ROWS_PER_STEP ?= 32
TRACEBACK_VERTICAL_STEP_FT ?= 5.0
TRACEBACK_PATCH_RADII ?= 3,5,9,15
TRACEBACK_MAX_MATCH_CANDIDATES ?= 2000
TRACEBACK_PROGRESS_EVERY ?= 250
TRACEBACK_USE_TRAIN_DICTIONARY ?=
TRACEBACK_LOCATION_WEIGHT ?= 1.0
TRACEBACK_LOCATION_SIGMA_FT ?= 120.0
TRACEBACK_ANCHOR_MIN_SCORE_QUANTILE ?= 0.8
TRACEBACK_ANCHOR_MIN_TOP1_GAP ?= 0.05
TRACEBACK_ANCHOR_MAX_BRIDGE_DELTA_FT ?= 160.0
TEMPLATE_ENVELOPE_OUTPUT ?= artifacts/template_envelope_v0
TEMPLATE_ENVELOPE_K_WELLS ?= -1
TEMPLATE_ENVELOPE_ROWS_PER_STEP ?= 32
TEMPLATE_ENVELOPE_SCALE_GRID ?= 0.5,0.7,0.85,0.95,1.0,1.05,1.15,1.3,1.5
TEMPLATE_ENVELOPE_OFFSET_GRID ?= -160,-140,-120,-100,-80,-60,-40,-20,0,20,40,60,80,100,120,140,160
TEMPLATE_ENVELOPE_TOP_N ?= 16
TEMPLATE_ENVELOPE_PROGRESS_EVERY ?= 100
TEMPLATE_ENVELOPE_COORD_SOURCE ?= z_aligned
TEMPLATE_ENVELOPE_MIN_SCORE ?= 0.1
TEMPLATE_ENVELOPE_REQUIRE_BEATS_SHUFFLED ?= 0
ANNOTATION_TOP_OUTPUT ?= artifacts/annotation_top_audit_v0
ANNOTATION_TOP_K_WELLS ?= -1
ANNOTATION_TOP_CONTROL_SPACING_ROWS ?= 323
ANNOTATION_TOP_PANEL_WELLS ?= 8
ANNOTATION_TOP_PROGRESS_EVERY ?= 100
TOP_STATE_OUTPUT ?= artifacts/top_state_teacher_v0
TOP_STATE_K_WELLS ?= -1
TOP_STATE_ROWS_PER_STEP ?= 32
TOP_STATE_FOLDS ?= 5
TOP_STATE_ITERATIONS ?= 400
TOP_STATE_LR ?= 0.05
TOP_STATE_DEPTH ?= 5
TOP_STATE_TARGET_TOP ?= ANCC
TOP_STATE_FLAT_EPSILON ?= 0.05
TOP_STATE_PANEL_WELLS ?= 6
TOP_STATE_PROGRESS_EVERY ?= 100
CHUNK_POLICY_TOP_STATE_PREDICTIONS ?= $(TOP_STATE_OUTPUT)/top_state_oof_predictions.parquet

# K-segment per-well constant-offset candidate
K_SEGMENT_OFFSET_OUTPUT ?= artifacts/k_segment_offset_v0
K_SEGMENT_OFFSET_K ?= 5
K_SEGMENT_OFFSET_K_WELLS ?= -1
K_SEGMENT_OFFSET_ITERATIONS ?= 800
K_SEGMENT_OFFSET_LR ?= 0.04
K_SEGMENT_OFFSET_DEPTH ?= 6
CHUNK_POLICY_K_OFFSET_PREDICTIONS ?= $(K_SEGMENT_OFFSET_OUTPUT)/k_offset_oof_predictions.parquet

# dTVT state-model candidate (per-row PyTorch NN with dual loss)
DTVT_STATE_OUTPUT ?= artifacts/dtvt_state_model_v0
DTVT_STATE_K_WELLS ?= -1
DTVT_STATE_EPOCHS ?= 25
DTVT_STATE_HIDDEN_DIM ?= 256
DTVT_STATE_LR ?= 5e-4
DTVT_STATE_ALPHA_LOCAL ?= 1.0
DTVT_STATE_BETA_GLOBAL ?= 5e-3
DTVT_STATE_MICRO_BATCH ?= 4
DTVT_STATE_DEVICE ?= auto
CHUNK_POLICY_DTVT_STATE_PREDICTIONS ?= $(DTVT_STATE_OUTPUT)/dtvt_state_oof_predictions.parquet

# Discrete global-offset grid experiment: TVT = anchor - dZ + offset * row_delta.
DISCRETE_OFFSET_OUTPUT ?= artifacts/discrete_offset_v0
DISCRETE_OFFSET_K_WELLS ?= -1
DISCRETE_OFFSET_GRID ?= -0.16:0.16:0.001
DISCRETE_OFFSET_ITERATIONS ?= 300
DISCRETE_OFFSET_LR ?= 0.04
DISCRETE_OFFSET_DEPTH ?= 5

# Discrete offset posterior / MDN-style diagnostic.
OFFSET_MDN_OUTPUT ?= artifacts/offset_mdn_v0
OFFSET_MDN_K_WELLS ?= -1
OFFSET_MDN_GRID ?= $(DISCRETE_OFFSET_GRID)
OFFSET_MDN_TEMPERATURES ?= 0.25,0.5,1.0,2.0
OFFSET_MDN_ITERATIONS ?= 300
OFFSET_MDN_LR ?= 0.04
OFFSET_MDN_DEPTH ?= 5

# Differentiable offset refinement from posterior shortlist.
OFFSET_TTO_OUTPUT ?= artifacts/offset_tto_v0
OFFSET_TTO_POSTERIOR ?= $(OFFSET_MDN_OUTPUT)/offset_mdn_oof_candidate_scores.parquet
OFFSET_TTO_GRID ?= $(DISCRETE_OFFSET_GRID)
OFFSET_TTO_SCORE_COLUMN ?= pred_log_mse
OFFSET_TTO_TOP_K ?= 10
OFFSET_TTO_K_WELLS ?= -1
OFFSET_TTO_STEPS ?= 60
OFFSET_TTO_LR ?= 0.05
OFFSET_TTO_MAX_DELTA ?= 0.015
OFFSET_TTO_OFFSET_L2 ?= 0.05

# LDT-like transformer over candidate lattice: (offset × lookahead span).
LATTICE_OFFSET_OUTPUT ?= artifacts/lattice_offset_v0
LATTICE_OFFSET_K_WELLS ?= 100
LATTICE_OFFSET_GRID ?= -0.16:0.16:0.008
LATTICE_OFFSET_SPANS ?= 64,128,256
LATTICE_OFFSET_STATE_STRIDE ?= 128
LATTICE_OFFSET_EPOCHS ?= 4
LATTICE_OFFSET_D_MODEL ?= 96
LATTICE_OFFSET_LAYERS ?= 2
LATTICE_OFFSET_BATCH_SIZE ?= 64
LATTICE_OFFSET_BEAM_SIZE ?= 8
LATTICE_OFFSET_BRANCH_TOP_K ?= 4
LATTICE_OFFSET_ON_POLICY_ROUNDS ?= 1
LATTICE_OFFSET_ON_POLICY_STATE_STRIDE ?= 0
LATTICE_OFFSET_ON_POLICY_MAX_WELLS ?= 0

.PHONY: install test copy-data smoke train eval stitch rank rank-crossfit mtp-ranker-crossfit track track-audit tail-audit corr-panel conditional-corr typewell-mismatch true-path-gr-audit plane-coordinate-audit annotation-top-audit top-state-teacher k-segment-offset dtvt-state-model discrete-offset offset-mdn offset-tto lattice-offset candidate-audit target-shootout soft-segment-smoke soft-segment-train soft-segment-v1 location-aware-corr formation-corr pseudo-zone pseudo-zone-v1 pseudo-zone-corr ga-zone-eval diagnostic-report diagnostic-pdf oof mtp-oof codebase-bundle collect-code pf-train pf-smoke pf-eval ga-train ga-smoke ga-eval ga-report oracle oracle-smoke pp-smoke pp-train pp-eval pp-costgbm residual-stack candidate-bank candidate-selector candidate-selector-v1 chunk-policy policy-solver shared-typewell-neighbour traceback template-envelope

install:
	$(UV) sync --extra dev

test:
	$(UV) run --extra dev pytest -q

copy-data:
	$(UV) run --extra dev python -m mtpnet.cli copy-data --source $(COPY_SOURCE) --target $(COPY_TARGET)

smoke:
	$(UV) run --extra dev python -m mtpnet.cli train --config $(SMOKE_CONFIG)

train:
	$(UV) run --extra dev python -m mtpnet.cli train --config $(CONFIG)

eval:
	$(UV) run --extra dev python -m mtpnet.cli eval --run-dir $(RUN_DIR)

stitch:
	$(UV) run --extra dev python -m mtpnet.cli stitch --run-dir $(RUN_DIR)

rank:
	$(UV) run --extra dev python -m mtpnet.cli rank --run-dir $(RUN_DIR)

rank-crossfit:
	$(UV) run --extra dev python -m mtpnet.cli rank-crossfit --run-dir $(RUN_DIR) --n-folds $(N_FOLDS) --seed $(SEED) --ranker-variant $(RANKER_VARIANT) $(if $(OUTPUT),--output-dir $(OUTPUT),)

mtp-ranker-crossfit: rank-crossfit

track:
	$(UV) run --extra dev python -m mtpnet.cli track --run-dir $(RUN_DIR) --logit-source $(LOGIT_SOURCE) --tau-ft $(TAU_FT) --ranker-beta $(RANKER_BETA) $(if $(CORR_BETA),--corr-beta $(CORR_BETA),) $(if $(RANKER_LOGITS),--ranker-logits $(RANKER_LOGITS),)

track-audit:
	$(UV) run --extra dev python -m mtpnet.cli track-audit --run-dir $(RUN_DIR)

tail-audit:
	$(UV) run --extra dev python -m mtpnet.cli tail-audit $(if $(RUN_DIR),--run-dir $(RUN_DIR),) $(if $(CANDIDATES),--candidates $(CANDIDATES),) $(if $(OUTPUT),--output-dir $(OUTPUT),) $(if $(PRIMARY_CANDIDATE),--primary-candidate $(PRIMARY_CANDIDATE),) --top-n $(TOP_N)

corr-panel:
	$(UV) run --extra dev python -m mtpnet.cli corr-panel --data-dir $(COPY_TARGET) --output-dir $(CORR_OUTPUT) --rows-per-step $(CORR_ROWS_PER_STEP) --vertical-step-ft $(CORR_VERTICAL_STEP_FT) --patch-radius $(CORR_PATCH_RADIUS) $(if $(CORR_PATCH_RADII),--patch-radii $(CORR_PATCH_RADII),) --stretch-factors $(CORR_STRETCH_FACTORS) --anchor-source $(CORR_ANCHOR_SOURCE) $(if $(CORR_SEARCH_RADIUS_FT),--search-radius-ft $(CORR_SEARCH_RADIUS_FT),) --k-wells $(CORR_K_WELLS) $(if $(CORR_NO_SHUFFLED),--no-shuffled,)

conditional-corr:
	$(UV) run --extra dev python -m mtpnet.conditional_corr_panel --data-dir $(COPY_TARGET) --output-dir $(CONDITIONAL_CORR_OUTPUT) --rows-per-step $(CORR_ROWS_PER_STEP) --vertical-step-ft $(CORR_VERTICAL_STEP_FT) --patch-radius $(CORR_PATCH_RADIUS) $(if $(CORR_PATCH_RADII),--patch-radii $(CORR_PATCH_RADII),) --stretch-factors $(CORR_STRETCH_FACTORS) --location-sigma-ft $(CONDITIONAL_CORR_LOCATION_SIGMA_FT) --gr-weight $(CONDITIONAL_CORR_GR_WEIGHT) --location-weight $(CONDITIONAL_CORR_LOCATION_WEIGHT) --k-wells $(CORR_K_WELLS) --seed $(SEED)

typewell-mismatch:
	$(UV) run --extra dev python -m mtpnet.cli typewell-mismatch --data-dir $(COPY_TARGET) --output-dir $(TYPEWELL_MISMATCH_OUTPUT) --rows-per-step $(CORR_ROWS_PER_STEP) --vertical-step-ft $(CORR_VERTICAL_STEP_FT) --patch-radius $(CORR_PATCH_RADIUS) $(if $(CORR_PATCH_RADII),--patch-radii $(CORR_PATCH_RADII),) --stretch-factors $(CORR_STRETCH_FACTORS) --k-wells $(CORR_K_WELLS)

true-path-gr-audit:
	$(UV) run --extra dev python -m mtpnet.cli true-path-gr-audit --data-dir $(COPY_TARGET) --output-dir $(TRUE_PATH_GR_OUTPUT) --rows-per-step $(CORR_ROWS_PER_STEP) --vertical-step-ft $(CORR_VERTICAL_STEP_FT) --patch-radius $(CORR_PATCH_RADIUS) $(if $(CORR_PATCH_RADII),--patch-radii $(CORR_PATCH_RADII),) --stretch-factors $(CORR_STRETCH_FACTORS) --k-wells $(CORR_K_WELLS) --seed $(SEED) $(if $(TAIL_CLASSES_PATH),--tail-classes-path $(TAIL_CLASSES_PATH),)

plane-coordinate-audit:
	$(UV) run --extra dev python -m mtpnet.plane_coordinate_audit --data-dir $(COPY_TARGET) --output-dir $(PLANE_COORD_OUTPUT) --rows-per-step $(CORR_ROWS_PER_STEP) --vertical-step-ft $(CORR_VERTICAL_STEP_FT) --k-wells $(CORR_K_WELLS) --seed $(SEED) $(if $(TAIL_CLASSES_PATH),--tail-classes-path $(TAIL_CLASSES_PATH),)

annotation-top-audit:
	$(UV) run --extra dev python -m mtpnet.annotation_top_audit \
		--data-dir $(COPY_TARGET)/train \
		--output-dir $(ANNOTATION_TOP_OUTPUT) \
		--k-wells $(ANNOTATION_TOP_K_WELLS) \
		--control-spacing-rows $(ANNOTATION_TOP_CONTROL_SPACING_ROWS) \
		--n-panel-wells $(ANNOTATION_TOP_PANEL_WELLS) \
		--progress-every $(ANNOTATION_TOP_PROGRESS_EVERY)

top-state-teacher:
	$(UV) run --extra dev python -m mtpnet.top_state_teacher \
		--data-dir $(COPY_TARGET)/train \
		--output-dir $(TOP_STATE_OUTPUT) \
		--rows-per-step $(TOP_STATE_ROWS_PER_STEP) \
		--n-folds $(TOP_STATE_FOLDS) \
		--seed $(SEED) \
		--k-wells $(TOP_STATE_K_WELLS) \
		--target-top-column $(TOP_STATE_TARGET_TOP) \
		--flat-epsilon-ft $(TOP_STATE_FLAT_EPSILON) \
		--iterations $(TOP_STATE_ITERATIONS) \
		--learning-rate $(TOP_STATE_LR) \
		--depth $(TOP_STATE_DEPTH) \
		--n-panel-wells $(TOP_STATE_PANEL_WELLS) \
		--progress-every $(TOP_STATE_PROGRESS_EVERY)

k-segment-offset:
	$(UV) run --extra dev python -m mtpnet.k_segment_offset \
		--data-dir $(COPY_TARGET)/train \
		--output-dir $(K_SEGMENT_OFFSET_OUTPUT) \
		--top-state-path $(TOP_STATE_OUTPUT)/top_state_oof_predictions.parquet \
		--n-folds $(TOP_STATE_FOLDS) \
		--seed $(SEED) \
		--k-wells $(K_SEGMENT_OFFSET_K_WELLS) \
		--K $(K_SEGMENT_OFFSET_K) \
		--rows-per-step $(TOP_STATE_ROWS_PER_STEP) \
		--iterations $(K_SEGMENT_OFFSET_ITERATIONS) \
		--learning-rate $(K_SEGMENT_OFFSET_LR) \
		--depth $(K_SEGMENT_OFFSET_DEPTH)

dtvt-state-model:
	$(UV) run --extra dev python -m mtpnet.dtvt_state_model \
		--data-dir $(COPY_TARGET)/train \
		--output-dir $(DTVT_STATE_OUTPUT) \
		--top-state-path $(TOP_STATE_OUTPUT)/top_state_oof_predictions.parquet \
		--n-folds $(TOP_STATE_FOLDS) \
		--seed $(SEED) \
		--k-wells $(DTVT_STATE_K_WELLS) \
		--rows-per-step $(TOP_STATE_ROWS_PER_STEP) \
		--epochs $(DTVT_STATE_EPOCHS) \
		--hidden-dim $(DTVT_STATE_HIDDEN_DIM) \
		--learning-rate $(DTVT_STATE_LR) \
		--alpha-local $(DTVT_STATE_ALPHA_LOCAL) \
		--beta-global $(DTVT_STATE_BETA_GLOBAL) \
		--micro-batch-wells $(DTVT_STATE_MICRO_BATCH) \
		--device $(DTVT_STATE_DEVICE)

discrete-offset:
	$(UV) run --extra dev python -m mtpnet.discrete_offset \
		--data-dir $(COPY_TARGET)/train \
		--output-dir $(DISCRETE_OFFSET_OUTPUT) \
		--offset-grid="$(DISCRETE_OFFSET_GRID)" \
		--n-folds $(TOP_STATE_FOLDS) \
		--seed $(SEED) \
		--k-wells $(DISCRETE_OFFSET_K_WELLS) \
		--iterations $(DISCRETE_OFFSET_ITERATIONS) \
		--learning-rate $(DISCRETE_OFFSET_LR) \
		--depth $(DISCRETE_OFFSET_DEPTH)

offset-mdn:
	$(UV) run --extra dev python -m mtpnet.offset_mdn \
		--data-dir $(COPY_TARGET)/train \
		--output-dir $(OFFSET_MDN_OUTPUT) \
		--offset-grid="$(OFFSET_MDN_GRID)" \
		--temperatures "$(OFFSET_MDN_TEMPERATURES)" \
		--n-folds $(TOP_STATE_FOLDS) \
		--seed $(SEED) \
		--k-wells $(OFFSET_MDN_K_WELLS) \
		--iterations $(OFFSET_MDN_ITERATIONS) \
		--learning-rate $(OFFSET_MDN_LR) \
		--depth $(OFFSET_MDN_DEPTH)

offset-tto:
	$(UV) run --extra dev python -m mtpnet.offset_tto \
		--data-dir $(COPY_TARGET)/train \
		--output-dir $(OFFSET_TTO_OUTPUT) \
		--posterior-scores-path $(OFFSET_TTO_POSTERIOR) \
		--offset-grid="$(OFFSET_TTO_GRID)" \
		--score-column $(OFFSET_TTO_SCORE_COLUMN) \
		--top-k $(OFFSET_TTO_TOP_K) \
		--n-folds $(TOP_STATE_FOLDS) \
		--seed $(SEED) \
		--k-wells $(OFFSET_TTO_K_WELLS) \
		--steps $(OFFSET_TTO_STEPS) \
		--learning-rate $(OFFSET_TTO_LR) \
		--max-delta $(OFFSET_TTO_MAX_DELTA) \
		--offset-l2 $(OFFSET_TTO_OFFSET_L2)

lattice-offset:
	$(UV) run --extra dev python -m mtpnet.lattice_offset \
		--data-dir $(COPY_TARGET)/train \
		--output-dir $(LATTICE_OFFSET_OUTPUT) \
		--offset-grid="$(LATTICE_OFFSET_GRID)" \
		--spans "$(LATTICE_OFFSET_SPANS)" \
		--state-stride $(LATTICE_OFFSET_STATE_STRIDE) \
		--n-folds $(TOP_STATE_FOLDS) \
		--seed $(SEED) \
		--k-wells $(LATTICE_OFFSET_K_WELLS) \
		--epochs $(LATTICE_OFFSET_EPOCHS) \
			--batch-size $(LATTICE_OFFSET_BATCH_SIZE) \
			--d-model $(LATTICE_OFFSET_D_MODEL) \
			--n-layers $(LATTICE_OFFSET_LAYERS) \
			--beam-size $(LATTICE_OFFSET_BEAM_SIZE) \
			--branch-top-k $(LATTICE_OFFSET_BRANCH_TOP_K) \
			--on-policy-rounds $(LATTICE_OFFSET_ON_POLICY_ROUNDS) \
			--on-policy-state-stride $(LATTICE_OFFSET_ON_POLICY_STATE_STRIDE) \
			--on-policy-max-wells $(LATTICE_OFFSET_ON_POLICY_MAX_WELLS)

# Per-chunk audit of new candidates vs b2 / bank oracle. Lightweight: no model training.
CANDIDATE_AUDIT_BASELINE ?= artifacts/chunk_ranker_dp_v2_selfcal_c512/chunk_policy_dataset.parquet
CANDIDATE_AUDIT_CHUNK_SIZE ?= 512
CANDIDATE_AUDIT_OUTPUT ?= artifacts/candidate_audit/audit.json

candidate-audit:
	$(UV) run --extra dev python -m mtpnet.candidate_audit \
		--baseline $(CANDIDATE_AUDIT_BASELINE) \
		--data-dir $(COPY_TARGET)/train \
		--chunk-size $(CANDIDATE_AUDIT_CHUNK_SIZE) \
		--k-offset $(CHUNK_POLICY_K_OFFSET_PREDICTIONS) \
		--dtvt-state $(CHUNK_POLICY_DTVT_STATE_PREDICTIONS) \
		--output $(CANDIDATE_AUDIT_OUTPUT)

target-shootout:
	$(UV) run --extra dev python -m mtpnet.target_shootout --data-dir $(COPY_TARGET) --output-dir $(TARGET_SHOOTOUT_OUTPUT) --n-folds $(SHOOTOUT_N_FOLDS) --n-estimators $(SHOOTOUT_N_ESTIMATORS) --learning-rate $(SHOOTOUT_LEARNING_RATE) --num-leaves $(SHOOTOUT_NUM_LEAVES) --k-wells $(CORR_K_WELLS) --seed $(SEED) $(if $(TAIL_CLASSES_PATH),--tail-classes-path $(TAIL_CLASSES_PATH),)

soft-segment-smoke:
	$(UV) run --extra dev python -m mtpnet.soft_segment --config $(SOFT_SEGMENT_SMOKE_CONFIG)

soft-segment-train:
	$(UV) run --extra dev python -m mtpnet.soft_segment --config $(SOFT_SEGMENT_CONFIG)

soft-segment-v1:
	$(UV) run --extra dev python -m mtpnet.soft_segment --config $(SOFT_SEGMENT_V1_CONFIG)

location-aware-corr:
	$(UV) run --extra dev python -m mtpnet.location_aware_corr \
		--data-dir $(COPY_TARGET) \
		--output-dir $(LOCATION_AWARE_CORR_OUTPUT) \
		--rows-per-step $(CORR_ROWS_PER_STEP) \
		--vertical-step-ft $(CORR_VERTICAL_STEP_FT) \
		--location-sigma-ft $(LOCATION_AWARE_SIGMA_FT) \
		--gr-weight $(LOCATION_AWARE_GR_WEIGHT) \
		--location-weight $(LOCATION_AWARE_LOCATION_WEIGHT) \
		--k-wells $(CORR_K_WELLS) \
		--seed $(SEED)

formation-corr:
	$(UV) run --extra dev python -m mtpnet.cli formation-corr --data-dir $(COPY_TARGET) --output-dir $(FORMATION_CORR_OUTPUT) --rows-per-step $(CORR_ROWS_PER_STEP) --vertical-step-ft $(CORR_VERTICAL_STEP_FT) --patch-radius $(CORR_PATCH_RADIUS) $(if $(CORR_PATCH_RADII),--patch-radii $(CORR_PATCH_RADII),) --stretch-factors $(CORR_STRETCH_FACTORS) --k-wells $(CORR_K_WELLS)

pseudo-zone:
	$(UV) run --extra dev python -m mtpnet.cli pseudo-zone --data-dir $(COPY_TARGET) --output-dir $(PSEUDO_ZONE_OUTPUT) --rows-per-step $(CORR_ROWS_PER_STEP) --n-bins $(PSEUDO_ZONE_BINS) --n-folds $(N_FOLDS) --k-wells $(CORR_K_WELLS)

pseudo-zone-v1:
	$(UV) run --extra dev python -m mtpnet.cli pseudo-zone-v1 --data-dir $(COPY_TARGET) --output-dir $(PSEUDO_ZONE_V1_OUTPUT) --rows-per-step $(CORR_ROWS_PER_STEP) --vertical-step-ft $(CORR_VERTICAL_STEP_FT) --n-bins $(PSEUDO_ZONE_BINS) --n-folds $(N_FOLDS) --k-wells $(CORR_K_WELLS) --seed $(SEED) --neural-epochs $(PSEUDO_ZONE_EPOCHS) --neural-hidden-dim $(PSEUDO_ZONE_HIDDEN_DIM) --neural-batch-size $(PSEUDO_ZONE_BATCH_SIZE) --neural-lr $(PSEUDO_ZONE_LR) $(if $(CORR_PATCH_RADII),--patch-radii $(CORR_PATCH_RADII),) --stretch-factors $(CORR_STRETCH_FACTORS) --known-tail-radius-ft $(PSEUDO_ZONE_KNOWN_TAIL_RADIUS_FT) --soft-zone-weight $(PSEUDO_ZONE_SOFT_WEIGHT)

pseudo-zone-corr:
	$(UV) run --extra dev python -m mtpnet.cli pseudo-zone-corr --data-dir $(COPY_TARGET) --output-dir $(PSEUDO_ZONE_V1_OUTPUT) --rows-per-step $(CORR_ROWS_PER_STEP) --vertical-step-ft $(CORR_VERTICAL_STEP_FT) --n-bins $(PSEUDO_ZONE_BINS) --n-folds $(N_FOLDS) --k-wells $(CORR_K_WELLS) --seed $(SEED) --neural-epochs $(PSEUDO_ZONE_EPOCHS) --neural-hidden-dim $(PSEUDO_ZONE_HIDDEN_DIM) --neural-batch-size $(PSEUDO_ZONE_BATCH_SIZE) --neural-lr $(PSEUDO_ZONE_LR) $(if $(CORR_PATCH_RADII),--patch-radii $(CORR_PATCH_RADII),) --stretch-factors $(CORR_STRETCH_FACTORS) --known-tail-radius-ft $(PSEUDO_ZONE_KNOWN_TAIL_RADIUS_FT) --soft-zone-weight $(PSEUDO_ZONE_SOFT_WEIGHT)

ga-zone-eval:
	$(UV) run --extra dev python -m mtpnet.cli ga-zone-eval --data-dir $(COPY_TARGET) --output-dir $(PSEUDO_ZONE_V1_OUTPUT) --rows-per-step $(CORR_ROWS_PER_STEP) --vertical-step-ft $(CORR_VERTICAL_STEP_FT) --n-bins $(PSEUDO_ZONE_BINS) --n-folds $(N_FOLDS) --k-wells $(CORR_K_WELLS) --seed $(SEED) --neural-epochs $(PSEUDO_ZONE_EPOCHS) --neural-hidden-dim $(PSEUDO_ZONE_HIDDEN_DIM) --neural-batch-size $(PSEUDO_ZONE_BATCH_SIZE) --neural-lr $(PSEUDO_ZONE_LR) $(if $(CORR_PATCH_RADII),--patch-radii $(CORR_PATCH_RADII),) --stretch-factors $(CORR_STRETCH_FACTORS) --known-tail-radius-ft $(PSEUDO_ZONE_KNOWN_TAIL_RADIUS_FT) --soft-zone-weight $(PSEUDO_ZONE_SOFT_WEIGHT)

diagnostic-report:
	$(UV) run --extra dev python -m mtpnet.cli diagnostic-report --artifacts-dir artifacts --output-dir $(DIAGNOSTIC_OUTPUT)

diagnostic-pdf:
	$(UV) run --extra dev python -m mtpnet.cli diagnostic-pdf --report-path $(DIAGNOSTIC_OUTPUT)/diagnostic_report.md --figures-dir $(DIAGNOSTIC_OUTPUT)/figures --output-path $(DIAGNOSTIC_PDF)

oof:
	$(UV) run --extra dev python -m mtpnet.cli oof --config $(CONFIG) --n-folds $(N_FOLDS) --seed $(SEED) --logit-source $(LOGIT_SOURCE) $(if $(CORR_BETA),--corr-beta $(CORR_BETA),) $(if $(OUTPUT),--output-dir $(OUTPUT),) $(if $(MAX_FOLDS),--max-folds $(MAX_FOLDS),) $(if $(FULL_STITCH),--full-stitch,)

mtp-oof: oof

codebase-bundle:
	$(UV) run --extra dev python -m mtpnet.codebase_bundle --root . --output $(CODEBASE_BUNDLE)

collect-code: codebase-bundle

# ── PathFormer targets ──────────────────────────────────────────────────────

pf-smoke:
	$(UV) run --extra dev python -m pathformer.train --config $(PF_SMOKE_CONFIG) --epochs 1

pf-train:
	$(UV) run --extra dev python -m pathformer.train --config $(PF_CONFIG) $(if $(PF_EPOCHS),--epochs $(PF_EPOCHS),)

pf-eval:
	@echo "pf-eval: re-run pf-train then inspect artifacts/pathformer_v0/eval_metrics.json"

# ── GeoAligner targets ──────────────────────────────────────────────────────

ga-smoke:
	$(UV) run --extra dev python -m geoaligner.train --config $(GA_SMOKE_CONFIG) --epochs 1

ga-train:
	$(UV) run --extra dev python -m geoaligner.train --config $(GA_CONFIG) $(if $(GA_EPOCHS),--epochs $(GA_EPOCHS),)

ga-eval:
	$(UV) run --extra dev python -m geoaligner.evaluate --config $(GA_CONFIG) $(if $(GA_CHECKPOINT),--checkpoint $(GA_CHECKPOINT),)

ga-report:
	@echo "GeoAligner report: inspect artifacts/geoaligner_v0/geoaligner_report.md"

# ── PathPolicy oracle targets ───────────────────────────────────────────────

oracle-smoke:
	$(UV) run --extra dev python -m pathpolicy.oracle --k_wells 20 --output_dir artifacts/oracle_smoke

oracle:
	$(UV) run --extra dev python -m pathpolicy.oracle --k_wells $(ORACLE_K_WELLS) --output_dir $(ORACLE_OUTPUT)

# ── PathPolicy train/eval targets ───────────────────────────────────────────

pp-smoke:
	$(UV) run --extra dev python -m pathpolicy.train --config $(PP_SMOKE_CONFIG)

pp-train:
	$(UV) run --extra dev python -m pathpolicy.train --config $(PP_CONFIG) $(if $(PP_OUTPUT),--output_dir $(PP_OUTPUT),)

pp-eval:
	$(UV) run --extra dev python -m pathpolicy.evaluate \
		--checkpoint $(PP_CHECKPOINT) \
		--output_dir $(PP_OUTPUT) \
		$(if $(PP_K_WELLS),--k_wells $(PP_K_WELLS),)

pp-costgbm:
	$(UV) run --extra dev python -m pathpolicy.costgbm \
		--oracle_chunks $(PP_COSTGBM_CHUNKS) \
		--output_dir    $(PP_COSTGBM_OUTPUT)

# ── Schema-safe residual/candidate-bank targets ────────────────────────────

residual-stack:
	$(UV) run --extra dev python -m mtpnet.residual_stack \
		--data-dir $(COPY_TARGET)/train \
		--output-dir $(RESIDUAL_OUTPUT) \
		--rows-per-step $(RESIDUAL_ROWS_PER_STEP) \
		--n-folds $(RESIDUAL_FOLDS) \
		--seed $(SEED) \
		--k-wells $(RESIDUAL_K_WELLS) \
		--iterations $(RESIDUAL_ITERATIONS) \
		--learning-rate $(RESIDUAL_LR) \
		--depth $(RESIDUAL_DEPTH)

candidate-bank:
	$(UV) run --extra dev python -m mtpnet.candidate_bank \
		--data-dir $(COPY_TARGET)/train \
		--output-dir $(CANDIDATE_BANK_OUTPUT) \
		--residual-predictions $(RESIDUAL_PREDICTIONS) \
		$(if $(TRACEBACK_CANDIDATES),--traceback-candidates $(TRACEBACK_CANDIDATES),) \
		--primary-candidate residual_stack_v0 \
		--k-wells $(RESIDUAL_K_WELLS) \
		--streaming-oracle

candidate-selector:
	$(UV) run --extra dev python -m mtpnet.candidate_selector \
		--data-dir $(COPY_TARGET)/train \
		--output-dir $(CANDIDATE_SELECTOR_OUTPUT) \
		--residual-predictions $(RESIDUAL_PREDICTIONS) \
		$(if $(TRACEBACK_CANDIDATES),--traceback-candidates $(TRACEBACK_CANDIDATES),) \
		--n-folds $(CANDIDATE_SELECTOR_FOLDS) \
		--seed $(SEED) \
		--k-wells $(RESIDUAL_K_WELLS) \
		--iterations $(CANDIDATE_SELECTOR_ITERATIONS) \
		--learning-rate $(CANDIDATE_SELECTOR_LR) \
		--depth $(CANDIDATE_SELECTOR_DEPTH) \
		--progress-every $(CANDIDATE_SELECTOR_PROGRESS_EVERY)

candidate-selector-v1:
	$(UV) run --extra dev python -m mtpnet.candidate_selector \
		--data-dir $(COPY_TARGET)/train \
		--output-dir $(CANDIDATE_SELECTOR_V1_OUTPUT) \
		--residual-predictions $(RESIDUAL_PREDICTIONS) \
		$(if $(TRACEBACK_CANDIDATES),--traceback-candidates $(TRACEBACK_CANDIDATES),) \
		--n-folds $(CANDIDATE_SELECTOR_FOLDS) \
		--seed $(SEED) \
		--k-wells $(RESIDUAL_K_WELLS) \
		--iterations $(CANDIDATE_SELECTOR_ITERATIONS) \
		--learning-rate $(CANDIDATE_SELECTOR_LR) \
		--depth $(CANDIDATE_SELECTOR_DEPTH) \
		--progress-every $(CANDIDATE_SELECTOR_PROGRESS_EVERY) \
		--mode ranker_guard \
		--guard-gain-threshold $(CANDIDATE_SELECTOR_GUARD_GAIN) \
		--dangerous-guard-gain-threshold $(CANDIDATE_SELECTOR_DANGEROUS_GUARD_GAIN)

chunk-policy:
	$(UV) run --extra dev python -m mtpnet.chunk_policy \
		--data-dir $(COPY_TARGET)/train \
		--output-dir $(CHUNK_POLICY_OUTPUT) \
		--residual-predictions $(RESIDUAL_PREDICTIONS) \
		$(if $(TRACEBACK_CANDIDATES),--traceback-candidates $(TRACEBACK_CANDIDATES),) \
		--n-folds $(CANDIDATE_SELECTOR_FOLDS) \
		--seed $(SEED) \
		--k-wells $(RESIDUAL_K_WELLS) \
		--chunk-size $(CHUNK_POLICY_CHUNK_SIZE) \
		--iterations $(CHUNK_POLICY_ITERATIONS) \
		--learning-rate $(CHUNK_POLICY_LR) \
		--depth $(CHUNK_POLICY_DEPTH) \
		--ranker-alpha $(CHUNK_POLICY_RANKER_ALPHA) \
		--switch-penalty $(CHUNK_POLICY_SWITCH_PENALTY) \
		--jump-penalty $(CHUNK_POLICY_JUMP_PENALTY) \
		--selfcal-max-probes $(CHUNK_POLICY_SELFCAL_PROBES) \
		--shared-typewell-neighbours $(CHUNK_POLICY_SHARED_TYPEWELL_NEIGHBOURS) \
		$(if $(CHUNK_POLICY_STEP_PREDICTIONS),--step-predictions $(CHUNK_POLICY_STEP_PREDICTIONS),) \
		$(if $(CHUNK_POLICY_LOCATION_AWARE_STEPS),--location-aware-steps $(CHUNK_POLICY_LOCATION_AWARE_STEPS),) \
		$(if $(CHUNK_POLICY_TOP_STATE_PREDICTIONS),--top-state-predictions $(CHUNK_POLICY_TOP_STATE_PREDICTIONS),) \
		$(if $(CHUNK_POLICY_K_OFFSET_PREDICTIONS),--k-offset-predictions $(CHUNK_POLICY_K_OFFSET_PREDICTIONS),) \
		$(if $(CHUNK_POLICY_DTVT_STATE_PREDICTIONS),--dtvt-state-predictions $(CHUNK_POLICY_DTVT_STATE_PREDICTIONS),) \
		--progress-every $(CANDIDATE_SELECTOR_PROGRESS_EVERY) \
		$(if $(CHUNK_POLICY_USE_SHARED_TYPEWELL),--use-shared-typewell-priors,) \
		$(CHUNK_POLICY_EXTRA_ARGS)

policy-solver:
	$(UV) run --extra dev python -m policy_solver.train \
		--data-dir $(COPY_TARGET)/train \
		--output-dir $(POLICY_SOLVER_OUTPUT) \
		--residual-predictions $(RESIDUAL_PREDICTIONS) \
		--n-folds $(CANDIDATE_SELECTOR_FOLDS) \
		--seed $(SEED) \
		--k-wells $(RESIDUAL_K_WELLS) \
		--chunk-size $(CHUNK_POLICY_CHUNK_SIZE) \
		--iterations $(CHUNK_POLICY_ITERATIONS) \
		--learning-rate $(CHUNK_POLICY_LR) \
		--depth $(CHUNK_POLICY_DEPTH) \
		--ranker-alpha $(CHUNK_POLICY_RANKER_ALPHA) \
		--switch-penalty $(CHUNK_POLICY_SWITCH_PENALTY) \
		--jump-penalty $(CHUNK_POLICY_JUMP_PENALTY) \
		--selfcal-max-probes $(CHUNK_POLICY_SELFCAL_PROBES) \
		--shared-typewell-neighbours $(CHUNK_POLICY_SHARED_TYPEWELL_NEIGHBOURS) \
		$(if $(CHUNK_POLICY_STEP_PREDICTIONS),--step-predictions $(CHUNK_POLICY_STEP_PREDICTIONS),) \
		$(if $(CHUNK_POLICY_LOCATION_AWARE_STEPS),--location-aware-steps $(CHUNK_POLICY_LOCATION_AWARE_STEPS),) \
		$(if $(CHUNK_POLICY_TOP_STATE_PREDICTIONS),--top-state-predictions $(CHUNK_POLICY_TOP_STATE_PREDICTIONS),) \
		--progress-every $(CANDIDATE_SELECTOR_PROGRESS_EVERY) \
		$(if $(CHUNK_POLICY_USE_SHARED_TYPEWELL),--use-shared-typewell-priors,) \
		$(CHUNK_POLICY_EXTRA_ARGS)

shared-typewell-neighbour:
	$(UV) run --extra dev python -m mtpnet.shared_typewell_neighbour \
		--data-dir $(COPY_TARGET)/train \
		--output-dir $(SHARED_TYPEWELL_OUTPUT) \
		--rows-per-step $(RESIDUAL_ROWS_PER_STEP) \
		--n-folds $(SHARED_TYPEWELL_FOLDS) \
		--seed $(SEED) \
		--k-wells $(SHARED_TYPEWELL_K_WELLS) \
		--n-signature-bins $(SHARED_TYPEWELL_SIGNATURE_BINS) \
		--max-shift-bins $(SHARED_TYPEWELL_MAX_SHIFT_BINS) \
		--top-k-neighbours $(SHARED_TYPEWELL_TOP_K)

traceback:
	$(UV) run --extra dev python -m mtpnet.traceback \
		--data-dir $(COPY_TARGET)/train \
		--output-dir $(TRACEBACK_OUTPUT) \
		--rows-per-step $(TRACEBACK_ROWS_PER_STEP) \
		--vertical-step-ft $(TRACEBACK_VERTICAL_STEP_FT) \
		--patch-radii $(TRACEBACK_PATCH_RADII) \
		--k-wells $(TRACEBACK_K_WELLS) \
		--max-match-candidates $(TRACEBACK_MAX_MATCH_CANDIDATES) \
		--progress-every $(TRACEBACK_PROGRESS_EVERY) \
		--location-weight $(TRACEBACK_LOCATION_WEIGHT) \
		--location-sigma-ft $(TRACEBACK_LOCATION_SIGMA_FT) \
		--anchor-min-score-quantile $(TRACEBACK_ANCHOR_MIN_SCORE_QUANTILE) \
		--anchor-min-top1-gap $(TRACEBACK_ANCHOR_MIN_TOP1_GAP) \
		--anchor-max-bridge-delta-ft $(TRACEBACK_ANCHOR_MAX_BRIDGE_DELTA_FT) \
		$(if $(TRACEBACK_USE_TRAIN_DICTIONARY),--use-train-dictionary,) \
		--seed $(SEED)

template-envelope:
	$(UV) run --extra dev python -m promising.template_envelope_v0.cli \
		--data-dir $(COPY_TARGET)/train \
		--output-dir $(TEMPLATE_ENVELOPE_OUTPUT) \
		--rows-per-step $(TEMPLATE_ENVELOPE_ROWS_PER_STEP) \
		--k-wells $(TEMPLATE_ENVELOPE_K_WELLS) \
		--seed $(SEED) \
		--scale-grid=$(TEMPLATE_ENVELOPE_SCALE_GRID) \
		--offset-grid=$(TEMPLATE_ENVELOPE_OFFSET_GRID) \
		--top-n-templates $(TEMPLATE_ENVELOPE_TOP_N) \
		--progress-every $(TEMPLATE_ENVELOPE_PROGRESS_EVERY) \
		--coord-source $(TEMPLATE_ENVELOPE_COORD_SOURCE) \
		--min-template-score $(TEMPLATE_ENVELOPE_MIN_SCORE) \
		$(if $(filter 1 true yes,$(TEMPLATE_ENVELOPE_REQUIRE_BEATS_SHUFFLED)),--require-beats-shuffled,)
