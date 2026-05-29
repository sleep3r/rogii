KAGGLE_CMD ?= kaggle
PYTHON ?= python3

KERNEL_DIR ?= kernel
KERNEL_REF ?= sleep3r/rogii-kojimar-pf-beam-tabicl-submit
COMPETITION ?= rogii-wellbore-geology-prediction
SUBMISSION_FILE ?= submission.csv
OUTPUT_DIR ?= artifacts/output
MESSAGE ?= Kojimar PF Beam TabICL inference stack
WAIT_TIMEOUT ?= 7200
POLL_INTERVAL ?= 120

.PHONY: help push status wait output validate submit-code open-log clean

help:
	@printf '%s\n' \
		'Targets:' \
		'  make push                  Push kernel/ to Kaggle' \
		'  make status                Show current kernel status' \
		'  make wait                  Wait until Kaggle run completes/errors' \
		'  make output                Download output and validate submission.csv' \
		'  make submit-code VERSION=N Submit kernel version N to the competition' \
		'  make clean                 Remove local downloaded outputs'

push:
	$(KAGGLE_CMD) kernels push -p $(KERNEL_DIR)

status:
	$(KAGGLE_CMD) kernels status $(KERNEL_REF)

wait:
	$(PYTHON) scripts/wait_kernel.py \
		--kaggle-cmd "$(KAGGLE_CMD)" \
		--kernel "$(KERNEL_REF)" \
		--timeout "$(WAIT_TIMEOUT)" \
		--poll-interval "$(POLL_INTERVAL)"

output:
	rm -rf "$(OUTPUT_DIR)"
	mkdir -p "$(OUTPUT_DIR)"
	$(KAGGLE_CMD) kernels output "$(KERNEL_REF)" -p "$(OUTPUT_DIR)"
	$(PYTHON) scripts/validate_submission.py "$(OUTPUT_DIR)/$(SUBMISSION_FILE)"

validate:
	$(PYTHON) scripts/validate_submission.py "$(OUTPUT_DIR)/$(SUBMISSION_FILE)"

submit-code:
	@test -n "$(VERSION)" || (echo 'Set VERSION=N, for example: make submit-code VERSION=2' && exit 2)
	$(KAGGLE_CMD) competitions submit "$(COMPETITION)" \
		-f "$(SUBMISSION_FILE)" \
		-k "$(KERNEL_REF)" \
		-v "$(VERSION)" \
		-m "$(MESSAGE)"

open-log:
	@find "$(OUTPUT_DIR)" -type f -name '*.log' -print

clean:
	rm -rf artifacts/output
