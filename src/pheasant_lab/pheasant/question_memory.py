"""Publish frozen benchmark questions as durable Pheasant memories.

Publication happens only after evaluation. A benchmark prompt present during
evaluation would change the system being measured, especially the memory arms.
Expected facts, matchers, evidence IDs, and answers never cross this boundary.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..benchmark.question_types import Question
from .capabilities import CapabilityMap, MissingCapability
from .client import PheasantClient
from .protocol import ProtocolError


@dataclass(frozen=True)
class QuestionMemoryResult:
    question_id: str
    record_id: str
    outcome: str | None
    created: bool | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "record_id": self.record_id,
            "outcome": self.outcome,
            "created": self.created,
            "published_after_evaluation": True,
        }


class BenchmarkQuestionPublisher:
    """Write question text and provenance, never benchmark answer material."""

    def __init__(
        self,
        client: PheasantClient,
        capabilities: CapabilityMap,
        config: Any,
        *,
        run_id: str,
        benchmark_version: str,
        tracer: Any = None,
        principal: str = "pheasant-swarm-lab",
    ) -> None:
        if not capabilities.has("write_memory"):
            raise MissingCapability(
                "benchmark.persist_questions_to_pheasant is enabled, but this region "
                "does not offer the configured write_memory capability"
            )
        self.client = client
        self.capabilities = capabilities
        self.config = config
        self.run_id = run_id
        self.benchmark_version = benchmark_version
        self.tracer = tracer
        self.principal = principal
        self._kb_field = str(config.argument_map.get("knowledge_base_field", "knowledge_base"))

    def publish(self, questions: Sequence[Question]) -> list[QuestionMemoryResult]:
        results: list[QuestionMemoryResult] = []
        for question in questions:
            outcome = self.client.call(
                self.capabilities.tool("write_memory"),
                {
                    self._kb_field: self.config.knowledge_base,
                    "text": _memory_text(question, self.benchmark_version),
                    "scope": "org",
                    "kind": "fact",
                    "subject": f"research-question:{question.topic_id}",
                    "principal": self.principal,
                    "sync": True,
                    "tags": [
                        "pheasant-swarm-lab",
                        "benchmark-question",
                        f"question:{question.question_id}",
                        f"topic:{question.topic_id}",
                        f"benchmark:{self.benchmark_version}",
                        f"run:{self.run_id}",
                    ],
                },
                idempotent=False,
                stage="benchmark-publication",
                question_id=question.question_id,
            )
            payload = outcome.result.payload() if outcome.result else {}
            body = payload if isinstance(payload, Mapping) else {}
            record = body.get("record") if isinstance(body.get("record"), Mapping) else {}
            record_id = str(record.get("record_id") or body.get("record_id") or "") or None
            if record_id is None:
                raise ProtocolError(
                    f"write_memory returned no record_id for question {question.question_id}"
                )
            result = QuestionMemoryResult(
                question_id=question.question_id,
                record_id=record_id,
                outcome=str(body.get("outcome") or "") or None,
                created=body.get("created") if isinstance(body.get("created"), bool) else None,
            )
            results.append(result)
            if self.tracer is not None:
                self.tracer.append("question-memories.jsonl", result.as_dict())
        if self.tracer is not None:
            self.tracer.emit(
                "benchmark.questions_published",
                payload={
                    "benchmark_version": self.benchmark_version,
                    "questions": len(results),
                    "record_ids": len(results),
                    "published_after_evaluation": True,
                },
            )
        return results


def _memory_text(question: Question, benchmark_version: str) -> str:
    lines = [
        "Research evaluation question (not an answer):",
        question.text,
        "",
        f"Question ID: {question.question_id}",
        f"Topic ID: {question.topic_id}",
        f"Question type: {question.type}",
        f"Difficulty: {question.difficulty}",
        f"Benchmark version: {benchmark_version}",
        "Purpose: verify whether this knowledge base serves the intended information.",
    ]
    if question.as_of:
        lines.append(f"Answer as of: {question.as_of}")
    return "\n".join(lines)
