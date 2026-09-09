#!/usr/bin/env bash
# Refuse a paid run that cannot work, before it costs anything.
#
# Exit 2 means "refused": a required capability is missing, a model has no
# price, a prompt is absent, or the region is unreachable.
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="${1:-configs/experiment.example.yaml}"
exec uv run pheasant-lab doctor --config "$CONFIG" "${@:2}"
