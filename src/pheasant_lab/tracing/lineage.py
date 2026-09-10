"""The lineage chain.

Every reported number must resolve through:

    run -> agent span -> source discovery -> acquisition -> source artifact
        -> extracted claim -> Pheasant ingest request -> ingest receipt
        -> Pheasant snapshot/index state -> benchmark question
        -> arm/session -> MCP search call -> returned context
        -> answer -> answer claim/citation -> proof event
        -> metric operands -> aggregate delta -> report statement

This module walks it in either direction over the raw files. It is what makes
the acceptance test possible: start at one sentence in ``summary.md``, resolve
it to an aggregate, to per-question values, to the Pheasant result and ingest
receipt, to the source and locator, and to every exclusion that touched the
denominator.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..lifecycle import RunPaths
from .events import read_jsonl

LINK_ORDER = (
    "run",
    "span",
    "source_candidate",
    "acquisition",
    "source",
    "claim",
    "ingest_request",
    "ingest_receipt",
    "snapshot",
    "question",
    "session",
    "search_call",
    "search_result",
    "answer",
    "answer_claim",
    "proof_event",
    "metric_operand",
    "metric_result",
    "paired_delta",
    "report_statement",
)


@dataclass
class LineageNode:
    kind: str
    id: str
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "id": self.id, "detail": self.detail}


@dataclass
class LineageChain:
    target: str
    nodes: list[LineageNode] = field(default_factory=list)
    broken_links: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.broken_links

    def as_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "complete": self.complete,
            "broken_links": list(self.broken_links),
            "nodes": [node.as_dict() for node in self.nodes],
        }


class LineageIndex:
    """An in-memory index over one run's raw files.

    Built by reading, never by writing: a lineage question must never be able
    to change the trace it is asked about.
    """

    def __init__(self, paths: RunPaths) -> None:
        self.paths = paths
        self.sources: dict[str, dict[str, Any]] = {}
        self.claims: dict[str, dict[str, Any]] = {}
        self.receipts: dict[str, dict[str, Any]] = {}
        self.questions: dict[str, dict[str, Any]] = {}
        self.answers: dict[str, dict[str, Any]] = {}
        self.proof_events: dict[str, list[dict[str, Any]]] = {}
        self.search_calls: dict[str, list[dict[str, Any]]] = {}
        self.events_by_type: dict[str, list[dict[str, Any]]] = {}
        self.metric_results: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        raw = self.paths.raw
        for record in read_jsonl(raw / "sources.jsonl"):
            self.sources[record["source_id"]] = record
        for record in read_jsonl(raw / "claims.jsonl"):
            self.claims[record["claim_id"]] = record
        # Last row wins: crossing the index barrier appends a new row for a
        # key rather than editing the old one.
        for record in read_jsonl(raw / "ingest-receipts.jsonl"):
            self.receipts[str(record.get("source_id") or record.get("idempotency_key", ""))] = (
                record
            )
        for record in read_jsonl(raw / "questions.jsonl"):
            self.questions[record["question_id"]] = record
        for record in read_jsonl(raw / "answers.jsonl"):
            self.answers[record["answer_id"]] = record
            key = (record["question_id"], record["arm_id"])
            self.search_calls.setdefault(f"{key[0]}::{key[1]}", []).extend(
                record.get("search_calls", [])
            )
        for record in read_jsonl(raw / "proof-events.jsonl"):
            self.proof_events.setdefault(record.get("question_id", ""), []).append(record)
        for record in read_jsonl(raw / "events.jsonl"):
            self.events_by_type.setdefault(record["event_type"], []).append(record)
        metrics = self.paths.metrics / "per-query.jsonl"
        if metrics.is_file():
            for record in read_jsonl(metrics):
                self.metric_results[record.get("metric_result_id", "")] = record

    # -- forward walks -----------------------------------------------------
    def chain_for_answer(self, answer_id: str) -> LineageChain:
        chain = LineageChain(target=answer_id)
        answer = self.answers.get(answer_id)
        if answer is None:
            chain.broken_links.append(f"answer {answer_id} not found")
            return chain
        chain.nodes.append(LineageNode("run", answer.get("run_id", ""), {}))
        chain.nodes.append(
            LineageNode(
                "session",
                answer.get("session_id", ""),
                {"arm": answer.get("arm_id"), "trace_id": answer.get("trace_id")},
            )
        )
        question = self.questions.get(answer["question_id"])
        if question is None:
            chain.broken_links.append(f"question {answer['question_id']} not found")
        else:
            chain.nodes.append(
                LineageNode(
                    "question",
                    question["question_id"],
                    {"type": question.get("type"), "cohorts": question.get("cohorts")},
                )
            )
        for call in answer.get("search_calls", []):
            chain.nodes.append(
                LineageNode(
                    "search_call",
                    call.get("search_request_id", ""),
                    {"query": call.get("query"), "results": len(call.get("results", []))},
                )
            )
            for result in call.get("results", []):
                chain.nodes.append(
                    LineageNode(
                        "search_result",
                        result.get("artifact_id", ""),
                        {
                            "rank": result.get("rank"),
                            "score": result.get("score"),
                            "source_id": result.get("source_id"),
                        },
                    )
                )
                self._extend_to_source(chain, result.get("source_id"))
        chain.nodes.append(LineageNode("answer", answer_id, {"abstained": answer.get("abstained")}))
        for index, claim in enumerate(answer.get("claims", [])):
            chain.nodes.append(
                LineageNode(
                    "answer_claim",
                    f"{answer_id}#{index}",
                    {"citations": claim.get("citations", [])},
                )
            )
        for proof in self.proof_events.get(answer.get("question_id", ""), []):
            if proof.get("arm_id") in (None, answer.get("arm_id")):
                chain.nodes.append(
                    LineageNode(
                        "proof_event",
                        proof.get("proof_id", ""),
                        {"event_type": proof.get("event_type")},
                    )
                )
        return chain

    def _extend_to_source(self, chain: LineageChain, source_id: str | None) -> None:
        if not source_id:
            chain.broken_links.append("search result carries no source_id")
            return
        source = self.sources.get(source_id)
        if source is None:
            chain.broken_links.append(f"source {source_id} not in sources.jsonl")
            return
        chain.nodes.append(
            LineageNode(
                "source",
                source_id,
                {
                    "stable_identifier": source.get("stable_identifier"),
                    "canonical_url": source.get("canonical_url"),
                    "state": source.get("state"),
                },
            )
        )
        receipt = self.receipts.get(source_id)
        if receipt is None:
            chain.broken_links.append(f"no ingest receipt for source {source_id}")
        else:
            chain.nodes.append(
                LineageNode(
                    "ingest_receipt",
                    receipt.get("receipt_id") or receipt.get("idempotency_key", ""),
                    {"status": receipt.get("status"), "artifact_id": receipt.get("artifact_id")},
                )
            )

    def chain_for_question(self, question_id: str, arm_id: str | None = None) -> list[LineageChain]:
        return [
            self.chain_for_answer(answer_id)
            for answer_id, answer in sorted(self.answers.items())
            if answer["question_id"] == question_id
            and (arm_id is None or answer["arm_id"] == arm_id)
        ]

    def chain_for_metric(self, metric_result_id: str) -> LineageChain:
        chain = LineageChain(target=metric_result_id)
        result = self.metric_results.get(metric_result_id)
        if result is None:
            chain.broken_links.append(f"metric result {metric_result_id} not found")
            return chain
        chain.nodes.append(
            LineageNode("metric_result", metric_result_id, {"metric": result.get("metric")})
        )
        for operand in result.get("operand_ids", []):
            chain.nodes.append(LineageNode("metric_operand", operand, {}))
            if operand in self.answers:
                chain.nodes.extend(self.chain_for_answer(operand).nodes)
        return chain

    # -- audits ------------------------------------------------------------
    def orphans(self) -> dict[str, list[str]]:
        """Referenced ids that resolve nowhere.

        A broken link is a finding: it is precisely the case where a report
        sentence cannot be resolved to its evidence.
        """

        findings: dict[str, list[str]] = {}
        claim_sources = [
            c["claim_id"] for c in self.claims.values() if c.get("source_id") not in self.sources
        ]
        if claim_sources:
            findings["claims_with_unknown_source"] = sorted(claim_sources)
        answer_questions = [
            a["answer_id"]
            for a in self.answers.values()
            if a.get("question_id") not in self.questions
        ]
        if answer_questions:
            findings["answers_with_unknown_question"] = sorted(answer_questions)
        uncited = [
            f"{a['answer_id']}#{i}"
            for a in self.answers.values()
            for i, claim in enumerate(a.get("claims", []))
            if not claim.get("citations")
        ]
        if uncited:
            findings["answer_claims_without_citation"] = sorted(uncited)
        return findings

    def sources_without_receipt(self) -> list[str]:
        return sorted(
            source_id
            for source_id, source in self.sources.items()
            if source.get("state") in {"submitted", "indexed", "verified"}
            and source_id not in self.receipts
        )


def load_records(paths: RunPaths, filename: str) -> list[dict[str, Any]]:
    return list(read_jsonl(paths.raw / filename))


def iter_files(paths: RunPaths, names: Iterable[str]) -> dict[str, Path]:
    return {name: paths.raw / name for name in names}
