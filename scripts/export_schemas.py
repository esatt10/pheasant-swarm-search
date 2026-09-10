#!/usr/bin/env python3
"""Export ``schemas/*.json`` — the shapes an outside reader can rely on.

The schemas are authored here, in one place, and written out. They are not
decoration: ``tests/contract/test_schemas.py`` validates a real run's artifacts
against them, so a record that drifts from its published shape turns a build
red rather than surprising whoever reads the run next.

    python scripts/export_schemas.py              # write the files
    python scripts/export_schemas.py --check-clean  # fail if they would change
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "schemas"

DRAFT = "https://json-schema.org/draft/2020-12/schema"
DIGEST = {"type": "string", "pattern": "^sha256:[0-9a-f]{64}$"}
NULLABLE_STRING = {"type": ["string", "null"]}
NULLABLE_NUMBER = {"type": ["number", "null"]}


def schema(title: str, description: str, properties: dict[str, Any], required: list[str], **extra: Any) -> dict[str, Any]:
    return {
        "$schema": DRAFT,
        "$id": f"https://github.com/esatt10/pheasant-deep-research/schemas/{title}.schema.json",
        "title": title,
        "description": description,
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": True,
        **extra,
    }


SCHEMAS: dict[str, dict[str, Any]] = {
    "run-manifest": schema(
        "run-manifest",
        "The document every later comparison is judged against. Two runs are comparable only "
        "when their config_digest matches, or when every differing resolved field has been "
        "enumerated.",
        {
            "schema_version": {"const": 1},
            "run_id": {"type": "string", "pattern": "^run-[0-9a-f]{16}$"},
            "experiment_name": {"type": "string"},
            "created_at": {"type": "string"},
            "nonce": {"type": "string"},
            "config_digest": DIGEST,
            "package_version": {"type": "string"},
            "command": {"type": "string"},
            "seed": {"type": "integer"},
            "arms": {"type": "array", "items": {"enum": ["S0", "C0", "P0", "P1", "P2"]}},
            "topics": {"type": "array", "items": {"type": "string"}},
            "cost_budget_usd": {"type": "number", "exclusiveMinimum": 0},
            "models": {"type": "object"},
            "pheasant": {"type": "object"},
            "replay": {"type": "object"},
            "privacy": {"type": "object"},
            "environment": {"type": "object"},
            "resolved_config": {"type": "object"},
            "status": {"type": "string"},
        },
        ["schema_version", "run_id", "config_digest", "arms", "models", "resolved_config"],
    ),
    "event": schema(
        "event",
        "The universal event envelope. Sequence is monotonic per run and payload_digest is a "
        "digest of the payload as recorded, so a payload edited after the fact stops matching.",
        {
            "event_id": {"type": "string"},
            "schema_version": {"const": 1},
            "occurred_at": {"type": "string"},
            "recorded_at": {"type": "string"},
            "run_id": {"type": "string"},
            "trace_id": {"type": "string", "pattern": "^[0-9a-f]{32}$"},
            "span_id": {"type": "string", "pattern": "^[0-9a-f]{16}$"},
            "parent_span_id": {"type": ["string", "null"]},
            "sequence": {"type": "integer", "minimum": 1},
            "agent_id": NULLABLE_STRING,
            "agent_role": NULLABLE_STRING,
            "topic_id": NULLABLE_STRING,
            "question_id": NULLABLE_STRING,
            "arm_id": {"type": ["string", "null"]},
            "event_type": {"type": "string"},
            "status": {"enum": ["started", "succeeded", "failed", "partial", "skipped"]},
            "input_refs": {"type": "array", "items": {"type": "string"}},
            "output_refs": {"type": "array", "items": {"type": "string"}},
            "payload": {"type": "object"},
            "payload_digest": DIGEST,
            "config_digest": DIGEST,
        },
        [
            "event_id",
            "schema_version",
            "occurred_at",
            "run_id",
            "trace_id",
            "span_id",
            "sequence",
            "event_type",
            "status",
            "payload_digest",
            "config_digest",
        ],
    ),
    "error": schema(
        "error",
        "One recorded attempt. Every attempt is recorded, including the ones a retry later "
        "rescued: a run that only recorded terminal failures would look healthier the flakier "
        "it was. Secrets never appear here - bodies are digests, messages are redacted.",
        {
            "error_id": {"type": "string"},
            "run_id": {"type": "string"},
            "event_id": NULLABLE_STRING,
            "trace_id": NULLABLE_STRING,
            "span_id": NULLABLE_STRING,
            "occurred_at": {"type": "string"},
            "stage": {
                "enum": [
                    "discovery",
                    "acquisition",
                    "extraction",
                    "mcp",
                    "ingest",
                    "index",
                    "retrieval",
                    "answer",
                    "evaluation",
                    "report",
                ]
            },
            "class": {
                "enum": [
                    "timeout",
                    "transport",
                    "authentication",
                    "rate_limit",
                    "schema",
                    "tool",
                    "model",
                    "budget",
                    "data",
                    "invariant",
                    "unknown",
                ]
            },
            "component": {"type": "string"},
            "operation": {"type": "string"},
            "retryable": {"type": ["boolean", "null"]},
            "attempt": {"type": "integer", "minimum": 1},
            "exception_type": {"type": "string"},
            "message_redacted": {"type": "string"},
            "stack_digest": {"type": ["string", "null"]},
            "request_digest": {"type": ["string", "null"]},
            "response_digest": {"type": ["string", "null"]},
            "resolution": {"enum": ["retried", "recovered", "skipped", "terminal", "unresolved"]},
            "backoff_seconds": NULLABLE_NUMBER,
            "impact": {
                "type": "object",
                "properties": {
                    "affected_queries": {"type": "array", "items": {"type": "string"}},
                    "excluded_from_metrics": {"type": "boolean"},
                    "comparability": {"enum": ["none", "partial", "invalidated"]},
                },
                "required": ["affected_queries", "excluded_from_metrics", "comparability"],
            },
        },
        ["error_id", "run_id", "occurred_at", "stage", "class", "component", "operation", "attempt", "impact"],
    ),
    "source": schema(
        "source",
        "One literature candidate, wherever it got to. Rejected and blocked candidates are "
        "retained with their reason code so the same failure is not rediscovered every round.",
        {
            "source_id": {"type": "string"},
            "candidate_id": {"type": "string"},
            "run_id": {"type": "string"},
            "provider": {"type": "string"},
            "title": {"type": "string"},
            "stable_identifier": NULLABLE_STRING,
            "canonical_url": NULLABLE_STRING,
            "abstract": NULLABLE_STRING,
            "authors": {"type": "array", "items": {"type": "string"}},
            "published_at": NULLABLE_STRING,
            "source_type": {"type": "string"},
            "license": {"type": "string"},
            "family_key": {"type": "string"},
            "family_key_derived": {"type": "boolean"},
            "content_digest": DIGEST,
            "peer_reviewed": {"type": "boolean"},
            "state": {
                "enum": [
                    "discovered",
                    "validated",
                    "acquired",
                    "extracted",
                    "submitted",
                    "indexed",
                    "verified",
                    "rejected",
                    "blocked",
                    "failed_retryable",
                    "failed_terminal",
                ]
            },
            "reason_code": NULLABLE_STRING,
            "discovery_event_id": NULLABLE_STRING,
            "acquisition_event_id": NULLABLE_STRING,
            "researcher_agent_id": NULLABLE_STRING,
            "artifact_id": NULLABLE_STRING,
            "idempotency_key": NULLABLE_STRING,
            "provenance_complete": {"type": "boolean"},
            "facet_ids": {"type": "array", "items": {"type": "string"}},
        },
        ["source_id", "run_id", "provider", "title", "state", "content_digest", "family_key"],
    ),
    "claim": schema(
        "claim",
        "One atomic extracted claim. `confidence_basis` decides eligibility: a "
        "researcher_inference claim may guide collection and can never be an expected-answer "
        "operand.",
        {
            "claim_id": {"type": "string"},
            "run_id": {"type": "string"},
            "claim_text": {"type": "string", "minLength": 1},
            "claim_type": {
                "enum": ["observation", "result", "method", "definition", "limitation", "contradiction"]
            },
            "source_id": {"type": "string"},
            "locator": {"type": "string"},
            "quotation_digest": {"type": ["string", "null"]},
            "support": {"enum": ["supports", "contradicts", "qualifies", "mentions"]},
            "confidence_basis": {"enum": ["direct_text", "structured_metadata", "researcher_inference"]},
            "eligible": {"type": "boolean"},
            "researcher_agent_id": {"type": "string"},
            "subtopic_id": {"type": "string"},
            "facet_ids": {"type": "array", "items": {"type": "string"}},
            "extracted_at": {"type": "string"},
        },
        ["claim_id", "claim_text", "claim_type", "source_id", "locator", "support", "confidence_basis", "eligible"],
    ),
    "ingest-receipt": schema(
        "ingest-receipt",
        "One item's receipt. `accepted` and `indexed` are separate dispositions because they "
        "are separate facts: acceptance is not searchability. The file is append-only, so a "
        "reader folds it by idempotency_key and the last row wins.",
        {
            "receipt_id": {"type": "string"},
            "run_id": {"type": "string"},
            "source_id": {"type": "string"},
            "idempotency_key": {"type": "string"},
            "submission_id": NULLABLE_STRING,
            "status": {"type": "string"},
            "artifact_id": NULLABLE_STRING,
            "document_id": NULLABLE_STRING,
            "content_digest": {"type": ["string", "null"]},
            "accepted_content_digest": {"type": ["string", "null"]},
            "digest_matches": {"type": ["boolean", "null"]},
            "deduplicated": {"type": "boolean"},
            "dedup_outcome": NULLABLE_STRING,
            "server_trace_id": NULLABLE_STRING,
            "indexing_state": NULLABLE_STRING,
            "error_code": NULLABLE_STRING,
            "error_message": NULLABLE_STRING,
            "retryable": {"type": ["boolean", "null"]},
            "submissions": {"type": "integer", "minimum": 1},
            "recorded_at": {"type": "string"},
        },
        ["receipt_id", "run_id", "idempotency_key", "status", "submissions"],
    ),
    "benchmark": schema(
        "benchmark",
        "One frozen question. The file carries no run stamp: two runs over the same corpus "
        "must produce byte-identical questions, or 'the same question in two runs' is not the "
        "same question and there is no trend line to draw.",
        {
            "question_id": {"type": "string", "pattern": "^q-[0-9a-f]{16}$"},
            "topic_id": {"type": "string"},
            "text": {"type": "string", "minLength": 1},
            "type": {
                "enum": [
                    "atomic_fact",
                    "synthesis",
                    "mechanism",
                    "contradiction",
                    "temporal",
                    "source_id",
                    "abstention",
                ]
            },
            "difficulty": {"enum": ["direct", "multi_hop", "adversarial"]},
            "required_fact_ids": {"type": "array", "items": {"type": "string"}},
            "acceptable_evidence_source_ids": {"type": "array", "items": {"type": "string"}},
            "known_negative_source_ids": {"type": "array", "items": {"type": "string"}},
            "as_of": NULLABLE_STRING,
            "cohorts": {
                "type": "array",
                "items": {"enum": ["anchor", "learned", "temporal_holdout", "control", "invariant"]},
                "minItems": 1,
            },
            "created_from_claim_ids": {"type": "array", "items": {"type": "string"}},
            "facet_ids": {"type": "array", "items": {"type": "string"}},
        },
        ["question_id", "topic_id", "text", "type", "difficulty", "cohorts"],
    ),
    "answer": schema(
        "answer",
        "One arm's answer in one session. `read_passages` is what the session actually read; "
        "a citation naming anything else is invalid and is counted.",
        {
            "answer_id": {"type": "string"},
            "run_id": {"type": "string"},
            "question_id": {"type": "string"},
            "arm_id": {"enum": ["S0", "C0", "P0", "P1", "P2"]},
            "repetition": {"type": "integer", "minimum": 1},
            "session_id": {"type": "string"},
            "trace_id": {"type": "string"},
            "answer_text": NULLABLE_STRING,
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "citations": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["text", "citations"],
                },
            },
            "abstained": {"type": "boolean"},
            "abstention_reason": NULLABLE_STRING,
            "queries_used": {"type": "array", "items": {"type": "string"}},
            "search_calls": {"type": "array"},
            "read_passages": {"type": "array"},
            "retrieved_artifact_ids": {"type": "array", "items": {"type": "string"}},
            "latency_ms": {"type": "number", "minimum": 0},
            "snapshot_id": NULLABLE_STRING,
            "graph_generation": NULLABLE_STRING,
            "status": {"type": "string"},
            "model": {"type": "string"},
            "provider": {"type": "string"},
            "cost_usd": {"type": "number", "minimum": 0},
        },
        ["answer_id", "run_id", "question_id", "arm_id", "repetition", "session_id", "claims", "abstained"],
    ),
    "metric-result": schema(
        "metric-result",
        "One measured value with everything needed to defend it. A metric that could not be "
        "computed reports `insufficient_evidence` with `value: null` - never 0.0 - and every "
        "row states what it does not support.",
        {
            "metric_result_id": {"type": "string"},
            "metric": {"type": "string"},
            "version": {"type": "string"},
            "classification": {"enum": ["primary", "diagnostic", "gate", "descriptive"]},
            "scope": {"type": "object"},
            "formula": {"type": "string", "minLength": 1},
            "substituted": {"type": "string"},
            "value": NULLABLE_NUMBER,
            "numerator": NULLABLE_NUMBER,
            "denominator": NULLABLE_NUMBER,
            "unit": {"type": "string"},
            "excluded": {"type": "integer", "minimum": 0},
            "exclusion_reasons": {"type": "object"},
            "operand_ids": {"type": "array", "items": {"type": "string"}},
            "evidence_refs": {"type": "array", "items": {"type": "string"}},
            "baseline": NULLABLE_NUMBER,
            "treatment": NULLABLE_NUMBER,
            "absolute_delta": NULLABLE_NUMBER,
            "relative_delta": NULLABLE_NUMBER,
            "interval": {"type": ["array", "null"]},
            "status": {"enum": ["ok", "insufficient_evidence", "not_available", "not_comparable"]},
            "threshold": NULLABLE_NUMBER,
            "claim_supported": {"type": "string", "minLength": 1},
            "claim_not_supported": {"type": "string", "minLength": 1},
            "limitation": {"type": "string", "minLength": 1},
        },
        [
            "metric_result_id",
            "metric",
            "version",
            "classification",
            "formula",
            "status",
            "claim_supported",
            "claim_not_supported",
            "limitation",
        ],
        allOf=[
            {
                "if": {"properties": {"status": {"const": "ok"}}},
                "then": {"required": ["value", "denominator"], "properties": {"value": {"type": "number"}}},
            },
            {
                "if": {"properties": {"status": {"enum": ["insufficient_evidence", "not_available"]}}},
                "then": {"properties": {"value": {"type": "null"}}},
            },
        ],
    ),
    "refinement-candidate": schema(
        "refinement-candidate",
        "A proposal with its evidence attached. Nothing here has been tested, and nothing here "
        "enters ordinary knowledge retrieval by default.",
        {
            "candidate_id": {"type": "string"},
            "target": {
                "enum": [
                    "ingestion",
                    "chunking",
                    "metadata",
                    "graph",
                    "retrieval",
                    "memory",
                    "mcp_adapter",
                    "orchestration",
                ]
            },
            "trigger_type": {
                "enum": [
                    "repeated_error",
                    "retrieval_miss",
                    "unsupported_claim",
                    "negative_exposure",
                    "latency",
                    "duplication",
                    "drift",
                ]
            },
            "evidence_refs": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "frequency": {"type": "integer", "minimum": 1},
            "affected_query_ids": {"type": "array", "items": {"type": "string"}},
            "affected_source_ids": {"type": "array", "items": {"type": "string"}},
            "suspected_cause": {"type": "string", "minLength": 1},
            "proposed_change": {"type": "string", "minLength": 1},
            "expected_metric": {"type": "string"},
            "risk_class": {"enum": ["low", "medium", "high"]},
            "status": {
                "enum": ["proposed", "shadow_tested", "accepted", "rejected", "insufficient_evidence"]
            },
        },
        [
            "candidate_id",
            "target",
            "trigger_type",
            "evidence_refs",
            "frequency",
            "suspected_cause",
            "proposed_change",
            "risk_class",
            "status",
        ],
    ),
}


def render(name: str) -> str:
    return json.dumps(SCHEMAS[name], indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-clean", action="store_true", help="fail if a file would change")
    args = parser.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    stale: list[str] = []
    for name in sorted(SCHEMAS):
        path = OUT / f"{name}.schema.json"
        body = render(name)
        if args.check_clean:
            if not path.is_file() or path.read_text(encoding="utf-8") != body:
                stale.append(path.name)
            continue
        path.write_text(body, encoding="utf-8")
    if args.check_clean and stale:
        print("schemas are stale; run `python scripts/export_schemas.py`:", file=sys.stderr)
        for name in stale:
            print(f"  - {name}", file=sys.stderr)
        return 1
    print(f"{len(SCHEMAS)} schemas {'checked' if args.check_clean else 'written'} in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
