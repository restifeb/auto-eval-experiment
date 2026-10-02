UV_AUTH = UV_INDEX_GEN_AI_INTERNAL_USERNAME=oauth2accesstoken UV_INDEX_GEN_AI_INTERNAL_PASSWORD="$$(gcloud auth print-access-token)"
DATASET ?= data/test_auto_eval_intern.csv
PRESET ?= default

.PHONY: install test lint run experiment

install:
	@command -v gcloud >/dev/null 2>&1 || { echo "gcloud is required to install gllm-evals. Install it and run 'gcloud auth login'."; exit 1; }
	@$(UV_AUTH) uv sync

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

run:
	uv run python -m auto_eval --dataset "$(DATASET)" --preset $(PRESET)

experiment:
	uv run python -m experiments.calibration_study
