"""``S0`` - the source-aware specialist.

Answers from the research package: the sources this run collected and the
trace that produced them. It is the *attainable specialist reference*, not
ground truth - it is scored by the same deterministic fact and citation rules
as every other arm, and it can be wrong.

It is also the only arm that sees the package, which is why it is constructed
with one and the Pheasant arms raise if they are.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

from ..textkit import content_terms, tokens
from .base import Answer, Arm


class SpecialistArm(Arm):
    arm_id = "S0"
    role = "specialist"

    def answer(self, question: Any, *, repetition: int) -> Answer:
        answer = self._new_answer(question, repetition)
        started = time.monotonic()
        with self.tracer.span(
            "arm.answer", arm_id=self.arm_id, question_id=question.question_id, stage="answer"
        ):
            passages = self._package_passages(question)
            answer.retrieved_artifact_ids = [
                str(p["artifact_id"]) for p in passages if p.get("artifact_id")
            ]
            answer.queries_used = [question.text]
            self._compose(question, passages, answer)
        answer.latency_ms = (time.monotonic() - started) * 1000.0
        return self._record(answer)

    def _package_passages(self, question: Any) -> list[dict[str, Any]]:
        """Select from the package the way a specialist reads their own notes.

        Ranked by term overlap and capped at the same budget the Pheasant arms
        get, so the comparison is between *what each arm can see*, not between
        how much text each was handed.
        """

        package: Sequence[Mapping[str, Any]] = self.context.research_package or []
        terms = set(content_terms(question.text))
        if question.as_of:
            package = [
                item
                for item in package
                if not item.get("published_at") or str(item["published_at"]) <= str(question.as_of)
            ]
        scored: list[tuple[float, Mapping[str, Any]]] = []
        for item in package:
            haystack = set(tokens(f"{item.get('title', '')} {item.get('text', '')}"))
            hits = terms & haystack
            if not hits:
                continue
            title_hits = terms & set(tokens(str(item.get("title", ""))))
            scored.append((len(hits) + 2.0 * len(title_hits), item))
        scored.sort(key=lambda pair: (-pair[0], str(pair[1].get("source_id"))))
        limit = self.config.replay.max_results_per_search
        return [
            {
                "artifact_id": item.get("artifact_id") or item.get("source_id"),
                "source_id": item.get("source_id"),
                "title": item.get("title"),
                "text": item.get("text"),
                "rank": index,
                "score": score,
                "locator": item.get("locator") or "package",
            }
            for index, (score, item) in enumerate(scored[:limit], start=1)
        ]
