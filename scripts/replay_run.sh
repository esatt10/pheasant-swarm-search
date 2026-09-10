#!/usr/bin/env bash
# Rebuild every projection from the raw events alone.
#
# The raw JSONL is authoritative; the DuckDB projection is disposable. This
# never writes to the trace it rebuilds from.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN="${1:?usage: replay_run.sh <run_id> [--config path]}"
CONFIG="${2:-configs/experiment.example.yaml}"
uv run pheasant-lab replay --run "$RUN" --config "$CONFIG" "${@:3}"
