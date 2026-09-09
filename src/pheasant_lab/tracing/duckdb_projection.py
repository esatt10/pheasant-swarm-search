"""The queryable projection.

DuckDB here is **read-side only**: raw JSONL is authoritative, this file is a
disposable projection rebuilt from it, and a projection error never mutates a
raw trace. The projector verifies schema version, sequence continuity, payload
digests and foreign-key resolvability, and reports what it found rather than
repairing it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..hashing import digest
from ..lifecycle import RunPaths
from .events import read_jsonl

TABLES = (
    "runs",
    "agents",
    "spans",
    "events",
    "source_candidates",
    "sources",
    "acquisition_attempts",
    "claims",
    "mcp_sessions",
    "mcp_calls",
    "ingest_receipts",
    "search_results",
    "errors",
    "retry_attempts",
    "refinement_candidates",
    "questions",
    "cohorts",
    "answers",
    "answer_claims",
    "citations",
    "proof_events",
    "metric_results",
    "paired_deltas",
    "gate_results",
    "token_usage",
    "cost_events",
    "latency_events",
)


class ProjectionUnavailable(RuntimeError):
    """DuckDB is not installed. The raw trace is unaffected and complete."""


@dataclass
class ProjectionReport:
    tables: dict[str, int] = field(default_factory=dict)
    findings: list[str] = field(default_factory=list)
    database: str | None = None

    @property
    def clean(self) -> bool:
        return not self.findings

    def as_dict(self) -> dict[str, Any]:
        return {
            "database": self.database,
            "clean": self.clean,
            "tables": dict(sorted(self.tables.items())),
            "findings": list(self.findings),
        }


def _duckdb():
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - depends on the extra
        raise ProjectionUnavailable(
            "duckdb is not installed. Install the `duckdb` extra to build the queryable "
            "projection; the raw JSONL trace is authoritative and is unaffected."
        ) from exc
    return duckdb


def verify_raw(paths: RunPaths, *, check_digests: bool = True) -> list[str]:
    """Structural checks over the raw stream, before anything is projected."""

    findings: list[str] = []
    expected = 0
    seen_ids: set[str] = set()
    for record in read_jsonl(paths.raw / "events.jsonl"):
        expected += 1
        sequence = int(record.get("sequence", -1))
        if sequence != expected:
            findings.append(f"sequence discontinuity: expected {expected}, found {sequence}")
            expected = sequence
        if record.get("schema_version") != 1:
            findings.append(
                f"event {record.get('event_id')} has schema_version {record.get('schema_version')}"
            )
        if record["event_id"] in seen_ids:
            findings.append(f"duplicate event_id {record['event_id']}")
        seen_ids.add(record["event_id"])
        if check_digests:
            recomputed = digest(record.get("payload", {}))
            if recomputed != record.get("payload_digest"):
                findings.append(
                    f"payload digest mismatch on {record['event_id']}: "
                    f"recorded {record.get('payload_digest')}, recomputed {recomputed}"
                )
    return findings


def _rows(paths: RunPaths) -> dict[str, list[dict[str, Any]]]:
    """Flatten the raw files into the projection's tables."""

    raw = paths.raw
    events = list(read_jsonl(raw / "events.jsonl"))
    answers = list(read_jsonl(raw / "answers.jsonl"))
    questions = list(read_jsonl(raw / "questions.jsonl"))

    tables: dict[str, list[dict[str, Any]]] = {name: [] for name in TABLES}

    manifest = (
        json.loads(paths.manifest.read_text(encoding="utf-8")) if paths.manifest.is_file() else {}
    )
    if manifest:
        tables["runs"].append(
            {
                "run_id": manifest.get("run_id"),
                "experiment_name": manifest.get("experiment_name"),
                "created_at": manifest.get("created_at"),
                "config_digest": manifest.get("config_digest"),
                "status": manifest.get("status"),
                "cost_budget_usd": manifest.get("cost_budget_usd"),
            }
        )

    for event in events:
        tables["events"].append(
            {
                "event_id": event["event_id"],
                "run_id": event["run_id"],
                "sequence": event["sequence"],
                "occurred_at": event["occurred_at"],
                "trace_id": event["trace_id"],
                "span_id": event["span_id"],
                "parent_span_id": event.get("parent_span_id"),
                "event_type": event["event_type"],
                "status": event["status"],
                "agent_id": event.get("agent_id"),
                "agent_role": event.get("agent_role"),
                "topic_id": event.get("topic_id"),
                "question_id": event.get("question_id"),
                "arm_id": event.get("arm_id"),
                "payload_digest": event["payload_digest"],
                "payload": json.dumps(event.get("payload", {}), sort_keys=True),
            }
        )
        if event.get("agent_id"):
            tables["agents"].append(
                {
                    "agent_id": event["agent_id"],
                    "run_id": event["run_id"],
                    "agent_role": event.get("agent_role"),
                    "topic_id": event.get("topic_id"),
                }
            )
        payload = event.get("payload", {})
        if event["event_type"].startswith("cost."):
            tables["cost_events"].append(
                {"event_id": event["event_id"], "run_id": event["run_id"], **payload}
            )
            tables["token_usage"].append(
                {
                    "event_id": event["event_id"],
                    "run_id": event["run_id"],
                    "role": payload.get("role"),
                    "model": payload.get("model"),
                    "input_tokens": payload.get("input_tokens"),
                    "output_tokens": payload.get("output_tokens"),
                    "actual_usd": payload.get("actual_usd"),
                }
            )
        if event["event_type"].startswith("mcp."):
            tables["mcp_calls"].append(
                {
                    "event_id": event["event_id"],
                    "run_id": event["run_id"],
                    "method": payload.get("method"),
                    "tool": payload.get("tool"),
                    "status": event["status"],
                    "duration_ms": payload.get("duration_ms"),
                    "attempt": payload.get("attempt"),
                    "arm_id": event.get("arm_id"),
                    "question_id": event.get("question_id"),
                }
            )
            if payload.get("duration_ms") is not None:
                tables["latency_events"].append(
                    {
                        "event_id": event["event_id"],
                        "run_id": event["run_id"],
                        "stage": "mcp",
                        "operation": payload.get("tool") or payload.get("method"),
                        "duration_ms": payload["duration_ms"],
                    }
                )
        if event["event_type"] == "mcp.session.initialized":
            tables["mcp_sessions"].append(
                {
                    "event_id": event["event_id"],
                    "run_id": event["run_id"],
                    "session_id": payload.get("session_id"),
                    "protocol_version": payload.get("protocol_version"),
                    "server_name": payload.get("server_name"),
                    "tool_count": payload.get("tool_count"),
                }
            )
        if event["event_type"].startswith("error.retry"):
            tables["retry_attempts"].append(
                {"event_id": event["event_id"], "run_id": event["run_id"], **payload}
            )

    for span in read_jsonl(raw / "spans.jsonl"):
        tables["spans"].append(
            {
                "span_id": span["span_id"],
                "trace_id": span["trace_id"],
                "parent_span_id": span.get("parent_span_id"),
                "run_id": span["run_id"],
                "name": span["name"],
                "started_at": span["started_at"],
                "ended_at": span.get("ended_at"),
                "duration_ms": span.get("duration_ms"),
                "status": span.get("status"),
                "attributes": json.dumps(span.get("attributes", {}), sort_keys=True),
            }
        )
        if span.get("duration_ms") is not None:
            tables["latency_events"].append(
                {
                    "event_id": None,
                    "run_id": span["run_id"],
                    "stage": str(
                        span.get("attributes", {}).get("stage") or span["name"].split(".")[0]
                    ),
                    "operation": span["name"],
                    "duration_ms": span["duration_ms"],
                }
            )

    for error in read_jsonl(raw / "errors.jsonl"):
        flat = dict(error)
        impact = flat.pop("impact", {}) or {}
        flat["affected_queries"] = json.dumps(impact.get("affected_queries", []))
        flat["excluded_from_metrics"] = impact.get("excluded_from_metrics")
        flat["comparability"] = impact.get("comparability")
        flat["error_class"] = flat.pop("class", None)
        tables["errors"].append(flat)

    for source in read_jsonl(raw / "sources.jsonl"):
        tables["sources"].append(
            {key: value for key, value in source.items() if not isinstance(value, dict | list)}
            | {"authors": json.dumps(source.get("authors", []))}
        )
        tables["source_candidates"].append(
            {
                "candidate_id": source.get("candidate_id") or source.get("source_id"),
                "source_id": source.get("source_id"),
                "run_id": source.get("run_id"),
                "state": source.get("state"),
                "reason_code": source.get("reason_code"),
                "provider": source.get("provider"),
                "subtopic_id": source.get("subtopic_id"),
            }
        )
        for attempt in source.get("acquisition_attempts", []) or []:
            tables["acquisition_attempts"].append({"source_id": source["source_id"], **attempt})

    for claim in read_jsonl(raw / "claims.jsonl"):
        tables["claims"].append({k: v for k, v in claim.items() if not isinstance(v, dict | list)})

    for receipt in read_jsonl(raw / "ingest-receipts.jsonl"):
        tables["ingest_receipts"].append(
            {k: v for k, v in receipt.items() if not isinstance(v, dict | list)}
        )

    for question in questions:
        tables["questions"].append(
            {
                "question_id": question["question_id"],
                "run_id": question.get("run_id"),
                "topic_id": question.get("topic_id"),
                "text": question.get("text"),
                "type": question.get("type"),
                "difficulty": question.get("difficulty"),
                "as_of": question.get("as_of"),
                "required_fact_count": len(question.get("required_fact_ids", [])),
            }
        )
        for cohort in question.get("cohorts", []):
            tables["cohorts"].append({"question_id": question["question_id"], "cohort": cohort})

    for answer in answers:
        tables["answers"].append(
            {
                "answer_id": answer["answer_id"],
                "run_id": answer.get("run_id"),
                "question_id": answer["question_id"],
                "arm_id": answer["arm_id"],
                "repetition": answer.get("repetition"),
                "session_id": answer.get("session_id"),
                "abstained": answer.get("abstained"),
                "status": answer.get("status"),
                "latency_ms": answer.get("latency_ms"),
                "snapshot_id": answer.get("snapshot_id"),
                "graph_generation": answer.get("graph_generation"),
                "answer_text": answer.get("answer_text"),
            }
        )
        for index, claim in enumerate(answer.get("claims", [])):
            tables["answer_claims"].append(
                {
                    "answer_claim_id": f"{answer['answer_id']}#{index}",
                    "answer_id": answer["answer_id"],
                    "question_id": answer["question_id"],
                    "arm_id": answer["arm_id"],
                    "text": claim.get("text"),
                }
            )
            for citation in claim.get("citations", []) or []:
                tables["citations"].append(
                    {
                        "answer_claim_id": f"{answer['answer_id']}#{index}",
                        "answer_id": answer["answer_id"],
                        "citation": citation,
                    }
                )
        for call in answer.get("search_calls", []) or []:
            for result in call.get("results", []) or []:
                tables["search_results"].append(
                    {
                        "search_request_id": call.get("search_request_id"),
                        "answer_id": answer["answer_id"],
                        "question_id": answer["question_id"],
                        "arm_id": answer["arm_id"],
                        "query": call.get("query"),
                        "round": call.get("round"),
                        "rank": result.get("rank"),
                        "score": result.get("score"),
                        "artifact_id": result.get("artifact_id"),
                        "source_id": result.get("source_id"),
                        "content_digest": result.get("content_digest"),
                        "snapshot_id": call.get("snapshot_id"),
                        "memory_ids": json.dumps(result.get("memory_ids", [])),
                    }
                )

    for proof in read_jsonl(raw / "proof-events.jsonl"):
        tables["proof_events"].append(
            {k: v for k, v in proof.items() if not isinstance(v, dict | list)}
        )

    metrics_dir = paths.metrics
    if (metrics_dir / "per-query.jsonl").is_file():
        for result in read_jsonl(metrics_dir / "per-query.jsonl"):
            tables["metric_results"].append(
                {
                    "metric_result_id": result.get("metric_result_id"),
                    "metric": result.get("metric"),
                    "version": result.get("version"),
                    "arm_id": result.get("arm_id"),
                    "question_id": result.get("question_id"),
                    "cohort": result.get("cohort"),
                    "value": result.get("value"),
                    "numerator": result.get("numerator"),
                    "denominator": result.get("denominator"),
                    "status": result.get("status"),
                }
            )
    deltas = metrics_dir / "paired-deltas.jsonl"
    if deltas.is_file():
        for row in read_jsonl(deltas):
            tables["paired_deltas"].append(
                {k: v for k, v in row.items() if not isinstance(v, dict | list)}
            )
    gates = metrics_dir / "gates.json"
    if gates.is_file():
        payload = json.loads(gates.read_text(encoding="utf-8"))
        for gate_set, body in payload.get("gate_sets", {}).items():
            for gate in body.get("gates", []):
                tables["gate_results"].append({"gate_set": gate_set, **gate})

    candidates = paths.metrics / "refinement-candidates.jsonl"
    if candidates.is_file():
        for row in read_jsonl(candidates):
            tables["refinement_candidates"].append(
                {k: v for k, v in row.items() if not isinstance(v, dict | list)}
            )

    return tables


def project(
    paths: RunPaths, *, verify_foreign_keys: bool = True, check_digests: bool = True
) -> ProjectionReport:
    """Rebuild ``projections/run.duckdb`` from the raw stream."""

    duckdb = _duckdb()
    report = ProjectionReport()
    report.findings.extend(verify_raw(paths, check_digests=check_digests))

    tables = _rows(paths)
    target = paths.projections / "run.duckdb"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    connection = duckdb.connect(str(target))
    try:
        for name, rows in tables.items():
            report.tables[name] = len(rows)
            if not rows:
                connection.execute(f'CREATE TABLE "{name}" (empty_placeholder VARCHAR)')
                continue
            columns = sorted({key for row in rows for key in row})
            normalised = [[_scalar(row.get(column)) for column in columns] for row in rows]
            connection.execute(
                f'CREATE TABLE "{name}" (' + ", ".join(f'"{c}" VARCHAR' for c in columns) + ")"
            )
            placeholders = ", ".join("?" for _ in columns)
            connection.executemany(f'INSERT INTO "{name}" VALUES ({placeholders})', normalised)
        if verify_foreign_keys:
            report.findings.extend(_foreign_keys(tables))
    finally:
        connection.close()
    report.database = str(target)
    return report


def _scalar(value: Any) -> Any:
    if isinstance(value, dict | list):
        return json.dumps(value, sort_keys=True)
    if value is None:
        return None
    return str(value)


def _foreign_keys(tables: dict[str, list[dict[str, Any]]]) -> list[str]:
    findings: list[str] = []
    question_ids = {row["question_id"] for row in tables["questions"]}
    answer_ids = {row["answer_id"] for row in tables["answers"]}
    source_ids = {row["source_id"] for row in tables["sources"] if row.get("source_id")}

    for row in tables["answers"]:
        if question_ids and row["question_id"] not in question_ids:
            findings.append(
                f"answer {row['answer_id']} names unknown question {row['question_id']}"
            )
    for row in tables["answer_claims"]:
        if row["answer_id"] not in answer_ids:
            findings.append(
                f"answer_claim {row['answer_claim_id']} names unknown answer {row['answer_id']}"
            )
    for row in tables["claims"]:
        if source_ids and row.get("source_id") not in source_ids:
            findings.append(
                f"claim {row.get('claim_id')} names unknown source {row.get('source_id')}"
            )
    for row in tables["search_results"]:
        if row["answer_id"] not in answer_ids:
            findings.append(f"search_result names unknown answer {row['answer_id']}")
    return findings


def query(database: str | Path, sql: str) -> list[tuple[Any, ...]]:
    """Run one read-only query against a built projection."""

    duckdb = _duckdb()
    connection = duckdb.connect(str(database), read_only=True)
    try:
        return connection.execute(sql).fetchall()
    finally:
        connection.close()
