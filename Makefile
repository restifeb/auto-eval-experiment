UV_AUTH = UV_INDEX_GEN_AI_INTERNAL_USERNAME=oauth2accesstoken UV_INDEX_GEN_AI_INTERNAL_PASSWORD="$$(gcloud auth print-access-token)"
PROJECT ?= pipeline
PRESET ?= default
ifeq ($(PROJECT),agent)
DATASET ?= data/agent/test_case_agent_evaluation.csv
else
DATASET ?= data/pipeline/test_auto_eval_intern.csv
endif

.PHONY: install test lint format run presets experiment

install:
	@command -v gcloud >/dev/null 2>&1 || { echo "gcloud is required to install gllm-evals. Install it and run 'gcloud auth login'."; exit 1; }
	@$(UV_AUTH) uv sync

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

run:
	uv run auto-eval --project $(PROJECT) --dataset "$(DATASET)" --preset $(PRESET)

presets:
	uv run auto-eval --project $(PROJECT) --list-presets

experiment:
	uv run python -m experiments.pipeline_calibration
