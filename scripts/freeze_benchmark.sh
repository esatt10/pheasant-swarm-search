#!/usr/bin/env bash
# Build and freeze the question set for a run, then report the leakage checks.
#
# Nothing after this point may change the questions: they are content-hashed
# into benchmark-manifest.json and `verify` re-derives every digest.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN="${1:?usage: freeze_benchmark.sh <run_id> [--config path]}"
CONFIG="${2:-configs/experiment.example.yaml}"
uv run pheasant-lab freeze-benchmark --run "$RUN" --config "$CONFIG" "${@:3}"
