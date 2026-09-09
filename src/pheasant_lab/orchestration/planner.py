"""The topic planner.

Its whole value is terminology expansion. A field's literature is split across
the names it used in each decade, so a plan written in one vocabulary
retrieves one decade and the coverage numbers that follow describe the
vocabulary rather than the field.

The planner never answers anything. Everything it emits is a search
instruction a research agent will test against the literature.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .. import ids
from ..budget import CostLedger
from ..models import ModelProvider, ModelRequest
from ..promptlib import load as load_prompt
from ..settings import LabConfig, Topic


@dataclass
class Subtopic:
    subtopic_id: str
    facet_ids: list[str]
    question: str
    terminology: list[str]
    providers: list[str] = field(default_factory=list)
    date_from: str | None = None
    date_to: str | None = None
    expect_disagreement: bool = False
    rationale: str = ""

    def queries(self, *, rounds: int) -> list[str]:
        """The query ladder for this subtopic.

        Round one is the question's own wording plus the planned terminology;
        later rounds are single-term probes, which is what surfaces the papers
        that use one of the field's other names for the same thing.
        """

        base = " ".join([self.question, *self.terminology[:3]])
        ladder = [base]
        for term in self.terminology[: max(0, rounds - 1)]:
            ladder.append(term)
        return ladder[:rounds]

    def as_dict(self) -> dict[str, Any]:
        return {
            "subtopic_id": self.subtopic_id,
            "facet_ids": list(self.facet_ids),
            "question": self.question,
            "terminology": list(self.terminology),
            "providers": list(self.providers),
            "date_from": self.date_from,
            "date_to": self.date_to,
            "expect_disagreement": self.expect_disagreement,
            "rationale": self.rationale,
        }


class Planner:
    def __init__(
        self,
        config: LabConfig,
        provider: ModelProvider,
        ledger: CostLedger,
        *,
        tracer: Any = None,
    ) -> None:
        self.config = config
        self.provider = provider
        self.ledger = ledger
        self.tracer = tracer
        self.prompt = load_prompt("planner")

    def plan(self, topic: Topic, *, max_subtopics: int, run_id: str) -> list[Subtopic]:
        spec = self.config.role("planner")
        context = {
            "topic": topic.model_dump(mode="json", by_alias=True),
            "max_subtopics": max_subtopics,
            "providers": list(self.config.collection.providers),
        }
        request = ModelRequest(
            role="planner",
            schema="plan",
            system=self.prompt,
            user=_render(topic, max_subtopics),
            context=context,
            max_output_tokens=spec.max_output_tokens,
            temperature=spec.temperature,
            seed=self.config.experiment.seed,
        )
        with self.ledger.spend(
            bucket="planning",
            role="planner",
            model=spec.model,
            prompt=request.prompt_text,
            max_output_tokens=spec.max_output_tokens,
            event_kind="model_call",
        ) as cost:
            response = self.provider.complete(request)
            cost.input_tokens = response.input_tokens
            cost.output_tokens = response.output_tokens
        if self.tracer is not None:
            self.tracer.emit("cost.model_call", payload=cost.as_payload())

        subtopics: list[Subtopic] = []
        for row in response.data.get("subtopics", [])[:max_subtopics]:
            question = str(row.get("question") or "").strip()
            if not question:
                continue
            subtopics.append(
                Subtopic(
                    subtopic_id=str(row.get("subtopic_id") or ids.subtopic_id(topic.id, question)),
                    facet_ids=[str(f) for f in row.get("facet_ids", []) if f],
                    question=question,
                    terminology=[str(t) for t in row.get("terminology", []) if t],
                    providers=[str(p) for p in row.get("providers", []) if p]
                    or list(self.config.collection.providers),
                    date_from=row.get("date_from"),
                    date_to=row.get("date_to"),
                    expect_disagreement=bool(row.get("expect_disagreement")),
                    rationale=str(row.get("rationale") or ""),
                )
            )
        if self.tracer is not None:
            self.tracer.emit(
                "collection.plan",
                payload={
                    "topic_id": topic.id,
                    "run_id": run_id,
                    "subtopics": [s.as_dict() for s in subtopics],
                    "requested": max_subtopics,
                },
                topic_id=topic.id,
            )
        return subtopics

    def replan(
        self, topic: Topic, *, gaps: list[str], existing: list[Subtopic], run_id: str
    ) -> list[Subtopic]:
        """Extend a plan to cover facets the audit found short.

        Widening beats deepening: a facet with no sources needs a different
        vocabulary, not another round of the one that already missed it.
        """

        covered = {facet for subtopic in existing for facet in subtopic.facet_ids}
        wanted = [
            facet for facet in topic.facets if facet.id in set(gaps) or facet.id not in covered
        ]
        if not wanted:
            return []
        fresh = self.plan(topic, max_subtopics=len(wanted), run_id=run_id)
        known = {s.subtopic_id for s in existing}
        return [s for s in fresh if s.subtopic_id not in known] or [
            Subtopic(
                subtopic_id=ids.subtopic_id(topic.id, f"{facet.label} (widened)"),
                facet_ids=[facet.id],
                question=f"What further independent evidence addresses {facet.label.lower()}?",
                terminology=[facet.label, *topic.seed_terms[:2]],
                providers=list(self.config.collection.providers),
                rationale="widened after the audit found this facet short",
            )
            for facet in wanted
        ]


def _render(topic: Topic, max_subtopics: int) -> str:
    facets = "\n".join(f"- {f.id} (weight {f.weight}): {f.label}" for f in topic.facets)
    seeds = ", ".join(topic.seed_terms) or "(none)"
    window = topic.date_range
    return (
        f"Topic: {topic.title}\n"
        f"Seed terms: {seeds}\n"
        f"Date window: {window.from_ or 'open'} to {window.to or 'open'}\n"
        f"Facets:\n{facets}\n\n"
        f"Produce at most {max_subtopics} subtopics. Return the JSON object described above and nothing else."
    )
