#!/usr/bin/env bash
# Create the virtualenv and install the package with its dev extras.
#
# Everything this lab needs is local: Python, uv, and (optionally) a running
# Pheasant. Nothing here provisions a database, a broker or a collector.
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. See https://docs.astral.sh/uv/ , or use:" >&2
  echo "  python3.12 -m venv .venv && .venv/bin/pip install -e '.[dev]'" >&2
  exit 1
fi

uv venv --python 3.12
uv pip install -e ".[dev]"

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "wrote .env from .env.example — fill in your model and Pheasant settings"
fi

echo
echo "next:"
echo "  uv run pheasant-lab demo --config configs/demo.yaml   # offline, free, end to end"
echo "  uv run pheasant-lab doctor --config configs/experiment.example.yaml"
