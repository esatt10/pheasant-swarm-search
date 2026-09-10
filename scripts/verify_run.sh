#!/usr/bin/env bash
# Verify a run: checksums, sequence continuity, payload digests, the frozen
# benchmark, lineage resolvability, leakage and the hard gates.
#
# Exit 1 means at least one finding. An `INCOMPLETE` gate verdict is one of
# them: a skipped gate and a failed gate are equally disqualifying for a
# result somebody will publish.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN="${1:?usage: verify_run.sh <run_id> [--config path]}"
CONFIG="${2:-configs/experiment.example.yaml}"
uv run pheasant-lab verify --run "$RUN" --config "$CONFIG" "${@:3}"
