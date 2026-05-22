SHELL := /bin/bash
PYTHON ?= python3
UV ?= uv
CONFIG ?= configs/mtp_v0.yml
SMOKE_CONFIG ?= configs/mtp_smoke.yml
RUN_DIR ?= artifacts/mtp_v0
COPY_SOURCE ?= ../old/data
COPY_TARGET ?= data

.PHONY: install test copy-data smoke train eval

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
