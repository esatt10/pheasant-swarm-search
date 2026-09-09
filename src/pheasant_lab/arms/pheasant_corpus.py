"""``P0`` - the Pheasant corpus baseline.

Search only, memory and steering **off**. Measures what the indexed corpus is
worth to a stateless agent that has never seen the research trace.

Memory is spelled ``off`` rather than omitted: omitting the argument selects
the region's default, and the difference between the region's default and no
memory at all is exactly what ``P1 - P0`` is supposed to measure.
"""

from __future__ import annotations

import time
from typing import Any

from ..models import ModelRequest
from ..pheasant.retrieval import MemoryOptions, SearchRequest, SearchResponse
from .base import Answer, Arm, passages_from


class PheasantCorpusArm(Arm):
    arm_id = "P0"
    role = "test_agent"
    memory_enabled = False

    def answer(self, question: Any, *, repetition: int) -> Answer:
        answer = self._new_answer(question, repetition)
        started = time.monotonic()
        with self.tracer.span(
            "arm.answer", arm_id=self.arm_id, question_id=question.question_id, stage="retrieval"
        ):
            responses = self._retrieve(question, answer)
            passages = passages_from(responses, limit=self.config.replay.max_results_per_search)
            answer.retrieved_artifact_ids = [
                str(p["artifact_id"]) for p in passages if p.get("artifact_id")
            ]
            self._compose(question, passages, answer)
        answer.latency_ms = (time.monotonic() - started) * 1000.0
        return self._record(answer)

    # -- retrieval ---------------------------------------------------------
    def _retrieve(self, question: Any, answer: Answer) -> list[SearchResponse]:
        assert self.context.retriever is not None
        responses: list[SearchResponse] = []
        rounds = max(1, self.config.replay.max_search_rounds_per_answer)
        for round_number in range(1, rounds + 1):
            queries = self._queries(question, responses, round_number)
            if not queries:
                break
            for query in queries:
                request = SearchRequest(
                    run_id=self.context.run_id,
                    arm_id=self.arm_id,
                    question_id=question.question_id,
                    query=query,
                    namespace=self.config.pheasant.knowledge_base,
                    top_k=self.config.replay.max_results_per_search,
                    snapshot_id=self.context.snapshot_id,
                    as_of=question.as_of,
                    memory=self.memory_options(),
                    round=round_number,
                    session=answer.session_id,
                    principal=self.context.principal,
                )
                try:
                    response = self.context.retriever.search(request)
                except Exception as exc:
                    self.tracer.errors.record(
                        exc,
                        stage="retrieval",
                        component=f"arms.{self.arm_id}",
                        operation="search",
                        resolution="terminal",
                        trace_id=answer.trace_id,
                    )
                    answer.status = "failed"
                    answer.error = str(exc)
                    return responses
                responses.append(response)
                answer.queries_used.append(query)
                answer.search_calls.append(
                    response.as_record(store_text=self.config.privacy.store_response_text)
                )
                answer.snapshot_id = answer.snapshot_id or response.snapshot_id
                answer.graph_generation = answer.graph_generation or response.graph_generation
                if response.partial:
                    # Partial is partial. An arm that treats a truncated
                    # result set as complete reports a corpus gap that is
                    # really a transport event.
                    answer.status = "partial"
        return responses

    def memory_options(self) -> MemoryOptions:
        return MemoryOptions(enabled=False)

    def _queries(
        self, question: Any, responses: list[SearchResponse], round_number: int
    ) -> list[str]:
        spec = self.config.role(self.role)
        seen = [
            {"text": result.matched_text, "title": result.title}
            for response in responses
            for result in response.results[:5]
        ]
        request = ModelRequest(
            role=self.role,
            schema="queries",
            system=self.prompt,
            user=(
                f"Question: {question.text}\nRound: {round_number}\n"
                'Return {"queries": [...]} and nothing else.'
            ),
            context={
                "question": question.text,
                "round": round_number,
                "results": seen,
                "strategy": dict(self.context.query_strategy or {}),
            },
            max_output_tokens=512,
            temperature=spec.temperature,
        )
        with self.ledger.spend(
            bucket=self.bucket,
            role=f"{self.arm_id}:queries",
            model=spec.model,
            prompt=request.prompt_text,
            max_output_tokens=512,
        ) as cost:
            response = self.model.complete(request)
            cost.input_tokens = response.input_tokens
            cost.output_tokens = response.output_tokens
        self.tracer.emit(
            "cost.model_call",
            payload=cost.as_payload(),
            arm_id=self.arm_id,
            question_id=question.question_id,
        )
        return [str(query) for query in response.data.get("queries", []) if str(query).strip()]
