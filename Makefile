# Pheasant Swarm Search
#
# Every target here is laptop-local. Nothing reaches a hosted database, a
# message broker, or a remote telemetry collector.

UV ?= uv
CONFIG ?= configs/experiment.example.yaml
RUN ?=

.DEFAULT_GOAL := help

.PHONY: help
help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

.PHONY: bootstrap
bootstrap:  ## Create the virtualenv and install the package with dev extras
	./scripts/bootstrap.sh

.PHONY: doctor
doctor:  ## Verify laptop, models, providers and the Pheasant MCP capability map
	$(UV) run pheasant-lab doctor --config $(CONFIG)

.PHONY: test
test:  ## Run the offline test suite
	$(UV) run pytest

.PHONY: lint
lint:  ## Lint and format-check
	$(UV) run ruff check src tests
	$(UV) run ruff format --check src tests

.PHONY: format
format:  ## Apply formatting
	$(UV) run ruff format src tests
	$(UV) run ruff check --fix src tests

.PHONY: schemas
schemas:  ## Regenerate schemas/*.json from the Pydantic models
	$(UV) run python scripts/export_schemas.py --check-clean

.PHONY: demo
demo:  ## End-to-end offline run against fixtures and a mock Pheasant server
	$(UV) run pheasant-lab demo --output-root runs

.PHONY: verify
verify:  ## Verify a run's hashes, pairing, leakage and gates (RUN=<run_id>)
	$(UV) run pheasant-lab verify --run $(RUN)

.PHONY: replay
replay:  ## Rebuild projections and metrics from raw events (RUN=<run_id>)
	$(UV) run pheasant-lab replay --run $(RUN)

.PHONY: report
report:  ## Render Markdown/CSV/JSON reports (RUN=<run_id>)
	$(UV) run pheasant-lab report --run $(RUN)

.PHONY: check
check: lint schemas test  ## Everything CI runs
