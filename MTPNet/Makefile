SHELL := /bin/bash
PYTHON ?= python3
UV ?= uv
CONFIG ?= configs/mtp_v0.yml
SMOKE_CONFIG ?= configs/mtp_smoke.yml
RUN_DIR ?= artifacts/mtp_v0
COPY_SOURCE ?= ../old/data
COPY_TARGET ?= data
CODEBASE_BUNDLE ?= artifacts/codebase_bundle.md

.PHONY: install test copy-data smoke train eval stitch rank track codebase-bundle collect-code

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

track:
	$(UV) run --extra dev python -m mtpnet.cli track --run-dir $(RUN_DIR)

codebase-bundle:
	$(UV) run --extra dev python -m mtpnet.codebase_bundle --root . --output $(CODEBASE_BUNDLE)

collect-code: codebase-bundle
