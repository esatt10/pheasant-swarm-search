"""The arm contract, and the isolation it enforces.

An arm receives a question and produces an answer. What it may *see* to do
that is the experiment's independent variable, so the permitted inputs are a
declared property of each arm and are checked here rather than trusted to each
implementation - a leak added by a later edit would otherwise be invisible
until somebody read the diff carefully.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .. import ids
from ..budget import CostLedger
from ..models import ModelProvider, ModelRequest
from ..pheasant.retrieval import Retriever, SearchResponse
from ..promptlib import load as load_prompt
from ..settings import LabConfig
from ..textkit import truncate

# What each arm is permitted to receive. Anything else is an isolation error.
PERMITTED_INPUTS: dict[str, frozenset[str]] = {
    "S0": frozenset({"question", "schema", "research_package"}),
    "C0": frozenset({"question", "schema"}),
    "P0": frozenset({"question", "schema", "capability_map"}),
    "P1": frozenset({"question", "schema", "capability_map"}),
    "P2": frozenset({"question", "schema", "capability_map", "query_strategy"}),
}

FORBIDDEN_FOR_PHEASANT_ARMS = frozenset(
    {
        "source_urls",
        "topic_plan",
        "specialist_notes",
        "expected_facts",
        "expected_evidence",
        "other_arm_answers",
        "other_arm_traces",
        "research_package",
        "claims",
    }
)


class IsolationError(RuntimeError):
    """An arm was handed something its design says it must not see."""


@dataclass
class AnswerClaim:
    text: str
    citations: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"text": self.text, "citations": list(self.citations)}


@dataclass
class Answer:
    answer_id: str
    run_id: str
    question_id: str
    arm_id: str
    repetition: int
    session_id: str
    trace_id: str
    answer_text: str = ""
    claims: list[AnswerClaim] = field(default_factory=list)
    abstained: bool = False
    abstention_reason: str | None = None
    uncertainty: str | None = None
    queries_used: list[str] = field(default_factory=list)
    search_calls: list[dict[str, Any]] = field(default_factory=list)
    # What the arm actually read before answering. Recorded for *every* arm,
    # including the ones that do not search: "a citation names something this
    # session read" is the question, and defining it as "something a Pheasant
    # search returned" would score the specialist's every citation invalid
    # because it does not use Pheasant.
    read_passages: list[dict[str, Any]] = field(default_factory=list)
    retrieved_artifact_ids: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    snapshot_id: str | None = None
    graph_generation: str | None = None
    status: str = "succeeded"
    error: str | None = None
    model: str = ""
    provider: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    as_of: str | None = None

    @property
    def complete(self) -> bool:
        """Did this arm produce a usable answer for a paired comparison?"""

        return self.status in {"succeeded", "abstained"}

    def as_record(self, *, store_text: bool = True) -> dict[str, Any]:
        return {
            "answer_id": self.answer_id,
            "run_id": self.run_id,
            "question_id": self.question_id,
            "arm_id": self.arm_id,
            "repetition": self.repetition,
            "session_id": self.session_id,
            "trace_id": self.trace_id,
            "answer_text": self.answer_text if store_text else None,
            "claims": [claim.as_dict() for claim in self.claims],
            "abstained": self.abstained,
            "abstention_reason": self.abstention_reason,
            "uncertainty": self.uncertainty,
            "queries_used": list(self.queries_used),
            "search_calls": list(self.search_calls),
            "read_passages": [
                {**passage, **({} if store_text else {"text": None})}
                for passage in self.read_passages
            ],
            "retrieved_artifact_ids": list(self.retrieved_artifact_ids),
            "latency_ms": self.latency_ms,
            "snapshot_id": self.snapshot_id,
            "graph_generation": self.graph_generation,
            "status": self.status,
            "error": self.error,
            "model": self.model,
            "provider": self.provider,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "as_of": self.as_of,
        }


@dataclass
class ArmContext:
    """Everything an arm may be constructed with.

    Note what is absent: the frozen benchmark's expected facts, the other
    arms, and the collection state. The engine holds those; an arm cannot
    reach them because it was never given them.
    """

    config: LabConfig
    run_id: str
    tracer: Any
    ledger: CostLedger
    model: ModelProvider
    retriever: Retriever | None = None
    snapshot_id: str | None = None
    research_package: Sequence[Mapping[str, Any]] | None = None
    query_strategy: Mapping[str, Any] | None = None
    memory_enabled: bool = False
    principal: str | None = None


class Arm(ABC):
    arm_id = "??"
    role = "test_agent"
    bucket = "evaluation"

    def __init__(self, context: ArmContext) -> None:
        self.context = context
        self.config = context.config
        self.tracer = context.tracer
        self.ledger = context.ledger
        self.model = context.model
        self.prompt = load_prompt(self.role)
        self._check_isolation()

    # -- isolation ---------------------------------------------------------
    def _check_isolation(self) -> None:
        permitted = PERMITTED_INPUTS[self.arm_id]
        if self.context.research_package is not None and "research_package" not in permitted:
            raise IsolationError(
                f"arm {self.arm_id} was given the research package. It answers a different "
                "question when it can see the trace that built the corpus."
            )
        if self.context.query_strategy is not None and "query_strategy" not in permitted:
            raise IsolationError(f"arm {self.arm_id} was given a tuned query strategy")
        if self.arm_id in {"P0", "P1", "P2"} and self.context.retriever is None:
            raise IsolationError(f"arm {self.arm_id} needs a Pheasant retriever and was given none")
        if self.arm_id == "C0" and self.context.retriever is not None:
            raise IsolationError(
                "the prior-only control was given a retriever; it measures the model prior and "
                "nothing else"
            )

    @staticmethod
    def guard_payload(arm_id: str, payload: Mapping[str, Any]) -> None:
        """Refuse a context payload carrying something an arm must not see."""

        if arm_id not in {"P0", "P1", "P2"}:
            return
        leaked = sorted(set(payload) & FORBIDDEN_FOR_PHEASANT_ARMS)
        if leaked:
            raise IsolationError(f"arm {arm_id} would be handed {leaked}")

    # -- the contract ------------------------------------------------------
    @abstractmethod
    def answer(self, question: Any, *, repetition: int) -> Answer:
        """Answer one question in a fresh session."""

    # -- shared plumbing ---------------------------------------------------
    def _new_answer(self, question: Any, repetition: int) -> Answer:
        session = f"{self.arm_id}-{question.question_id}-{repetition}-{ids.new_nonce(4)}"
        trace = (
            self.tracer.new_trace()
            if self.config.replay.fresh_session_per_question
            else self.tracer.trace_id
        )
        return Answer(
            answer_id=ids.answer_id(
                self.context.run_id, self.arm_id, question.question_id, repetition
            ),
            run_id=self.context.run_id,
            question_id=question.question_id,
            arm_id=self.arm_id,
            repetition=repetition,
            session_id=session,
            trace_id=trace,
            as_of=question.as_of,
        )

    def _compose(
        self, question: Any, passages: Sequence[Mapping[str, Any]], answer: Answer
    ) -> None:
        """Turn retrieved passages into an answer, and bill it."""

        spec = self.config.role(self.role)
        context = {
            "question": question.text,
            "question_type": question.type,
            "as_of": question.as_of,
            "passages": [dict(passage) for passage in passages],
            "queries_used": list(answer.queries_used),
        }
        self.guard_payload(self.arm_id, context)
        answer.read_passages = [
            {
                "artifact_id": str(passage.get("artifact_id") or ""),
                "source_id": passage.get("source_id"),
                "text": str(passage.get("text") or ""),
                "rank": passage.get("rank"),
            }
            for passage in passages
        ]
        request = ModelRequest(
            role=self.role,
            schema="answer",
            system=self.prompt,
            user=_render_question(question, passages),
            context=context,
            max_output_tokens=spec.max_output_tokens,
            temperature=spec.temperature,
            seed=self.config.experiment.seed,
        )
        with self.ledger.spend(
            bucket=self.bucket,
            role=f"{self.arm_id}:{self.role}",
            model=spec.model,
            prompt=request.prompt_text,
            max_output_tokens=spec.max_output_tokens,
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

        answer.answer_text = str(response.data.get("answer_text") or "")
        answer.claims = [
            AnswerClaim(
                text=str(row.get("text") or ""),
                citations=[str(c) for c in (row.get("citations") or []) if c],
            )
            for row in response.data.get("claims", [])
            if str(row.get("text") or "").strip()
        ]
        answer.abstained = bool(response.data.get("abstained"))
        answer.abstention_reason = response.data.get("abstention_reason")
        answer.uncertainty = response.data.get("uncertainty")
        answer.model = spec.model
        answer.provider = response.provider
        answer.input_tokens = response.input_tokens
        answer.output_tokens = response.output_tokens
        answer.cost_usd = cost.actual_usd or 0.0
        if answer.abstained:
            answer.status = "abstained"

    def _record(self, answer: Answer) -> Answer:
        self.tracer.append(
            "answers.jsonl", answer.as_record(store_text=self.config.privacy.store_response_text)
        )
        self.tracer.emit(
            "arm.answered",
            status="succeeded" if answer.complete else "failed",
            payload={
                "answer_id": answer.answer_id,
                "arm_id": answer.arm_id,
                "question_id": answer.question_id,
                "repetition": answer.repetition,
                "abstained": answer.abstained,
                "claims": len(answer.claims),
                "retrieved": len(answer.retrieved_artifact_ids),
                "latency_ms": answer.latency_ms,
                "cost_usd": answer.cost_usd,
                "snapshot_id": answer.snapshot_id,
            },
            arm_id=answer.arm_id,
            question_id=answer.question_id,
        )
        return answer


def passages_from(responses: Sequence[SearchResponse], *, limit: int) -> list[dict[str, Any]]:
    """Flatten search responses into the passages an answerer reads.

    Deduplicated by artifact and kept in fused rank order, because an
    answerer handed the same passage three times will cite it three times and
    the precision denominator would count that as three claims.
    """

    seen: set[str] = set()
    passages: list[dict[str, Any]] = []
    for response in responses:
        for result in response.results:
            key = result.artifact_id or result.matched_text_digest or ""
            if not key or key in seen:
                continue
            seen.add(key)
            passages.append(
                {
                    "artifact_id": result.artifact_id,
                    "source_id": result.source_id,
                    "title": result.title,
                    "text": result.matched_text or "",
                    "rank": result.rank,
                    "score": result.score,
                    "locator": result.locator,
                }
            )
            if len(passages) >= limit:
                return passages
    return passages


def _render_question(question: Any, passages: Sequence[Mapping[str, Any]]) -> str:
    body = "\n\n".join(
        f"[{passage.get('artifact_id')}] {truncate(str(passage.get('text') or ''), 1200)}"
        for passage in passages
    )
    as_of = f"\nAnswer as of: {question.as_of}" if question.as_of else ""
    return (
        f"Question ({question.type}): {question.text}{as_of}\n\n"
        f"Retrieved passages:\n{body or '(nothing was retrieved)'}\n\n"
        "Return the JSON object described above and nothing else."
    )
