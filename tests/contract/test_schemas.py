"""The published schemas, validated against a real run.

A schema nothing checks is documentation of an intention. These tests validate
the artifacts a run actually produced, so a record that drifts from its
published shape turns a build red rather than surprising whoever reads the run
next.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

jsonschema = pytest.importorskip("jsonschema")

REPO = Path(__file__).resolve().parents[2]
SCHEMAS = REPO / "schemas"


def load(name: str) -> dict:
    return json.loads((SCHEMAS / f"{name}.schema.json").read_text())


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_the_exported_schemas_are_not_stale():
    completed = subprocess.run(
        [sys.executable, "scripts/export_schemas.py", "--check-clean"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_every_schema_is_itself_valid():
    for path in sorted(SCHEMAS.glob("*.schema.json")):
        payload = json.loads(path.read_text())
        jsonschema.Draft202012Validator.check_schema(payload)


@pytest.mark.parametrize(
    ("schema_name", "relative", "jsonl"),
    [
        ("run-manifest", "run-manifest.json", False),
        ("event", "raw/events.jsonl", True),
        ("error", "raw/errors.jsonl", True),
        ("source", "raw/sources.jsonl", True),
        ("claim", "raw/claims.jsonl", True),
        ("ingest-receipt", "raw/ingest-receipts.jsonl", True),
        ("benchmark", "benchmark/questions.jsonl", True),
        ("answer", "raw/answers.jsonl", True),
        ("metric-result", "metrics/per-query.jsonl", True),
        ("refinement-candidate", "metrics/refinement-candidates.jsonl", True),
    ],
)
def test_a_real_run_validates_against_its_schema(demo_run, schema_name, relative, jsonl):
    run = demo_run[0]
    path = run / relative
    if not path.is_file():
        pytest.skip(f"{relative} was not produced by this run")
    validator = jsonschema.Draft202012Validator(load(schema_name))
    rows = read_jsonl(path) if jsonl else [json.loads(path.read_text())]
    if not rows:
        pytest.skip(f"{relative} is empty")
    for row in rows:
        errors = sorted(validator.iter_errors(row), key=lambda error: error.path)
        assert not errors, f"{relative}: {errors[0].message} at {list(errors[0].path)}"


def test_the_metric_schema_refuses_a_zero_where_evidence_was_missing():
    """The rule the metric contract exists for, as a schema constraint."""

    validator = jsonschema.Draft202012Validator(load("metric-result"))
    bad = {
        "metric_result_id": "metric-1",
        "metric": "fact_f1",
        "version": "1",
        "classification": "primary",
        "formula": "2TP / (2TP + FP + FN)",
        "status": "insufficient_evidence",
        "value": 0.0,
        "claim_supported": "x",
        "claim_not_supported": "y",
        "limitation": "z",
    }
    assert list(validator.iter_errors(bad)), "0.0 with insufficient_evidence must not validate"
    bad["value"] = None
    assert not list(validator.iter_errors(bad))


def test_the_event_schema_requires_a_payload_digest():
    validator = jsonschema.Draft202012Validator(load("event"))
    event = {
        "event_id": "event-1",
        "schema_version": 1,
        "occurred_at": "2026-01-01T00:00:00Z",
        "run_id": "run-1",
        "trace_id": "0" * 32,
        "span_id": "0" * 16,
        "sequence": 1,
        "event_type": "run.started",
        "status": "succeeded",
        "config_digest": "sha256:" + "0" * 64,
    }
    assert list(validator.iter_errors(event))
    event["payload_digest"] = "sha256:" + "0" * 64
    assert not list(validator.iter_errors(event))
