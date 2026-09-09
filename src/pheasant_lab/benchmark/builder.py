"""Building the question set from a frozen evidence ledger.

Every question is grounded before it is written: the builder must be able to
name its required facts, the sources that support them, and (where it has
them) the sources that look relevant and are not. A question it cannot ground
that way does not go in the set - it is recorded as an exclusion, with the
reason, so the composition shortfall is visible rather than silently filled
with something weaker.

The composition is filled exactly. The types are not interchangeable: a
contradiction question that quietly became an atomic-fact question measures
something the report will still call contradiction handling.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .. import ids
from ..budget import CostLedger
from ..hashing import digest
from ..models import ModelProvider, ModelRequest
from ..orchestration.state import ClaimRecord, CollectionState
from ..promptlib import load as load_prompt
from ..settings import LabConfig, Topic
from ..textkit import content_terms, normalise, numbers, salient_terms, truncate
from .question_types import (
    COHORTS,
    COMPOSITION_ALIASES,
    ExpectedEvidence,
    ExpectedFact,
    FactMatcher,
    Question,
)

MECHANISM_MARKERS = (
    "via",
    "through",
    "pathway",
    "mechanism",
    "binds",
    "mediat",
    "catalys",
    "because",
    "due to",
)

# A question is built from a claim's *subject* terms and matched on the rest.
# Without that split the question's own wording satisfies its expected fact -
# which the leakage checker calls `answer_in_question` and refuses, correctly:
# a question that contains its answer measures nothing.
SUBJECT_TERMS = 3
MINIMUM_ANSWER_TERMS = 2


def _split_terms(
    text: str, *, subject: int = SUBJECT_TERMS, limit: int = 10
) -> tuple[list[str], list[str]]:
    """Return ``(subject terms, answer terms)`` for one claim.

    The subject terms are what the question names; the answer terms are what
    an answer has to supply. They are disjoint by construction, so a matcher
    built from the second can never be satisfied by wording built from the
    first.
    """

    terms = salient_terms(text, limit=limit)
    return terms[:subject], terms[subject:]


@dataclass
class Exclusion:
    kind: str
    reason: str
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "reason": self.reason, "detail": self.detail}


@dataclass
class BuiltBenchmark:
    version: str
    topic_id: str
    questions: list[Question] = field(default_factory=list)
    facts: dict[str, ExpectedFact] = field(default_factory=dict)
    evidence: dict[str, ExpectedEvidence] = field(default_factory=dict)
    exclusions: list[Exclusion] = field(default_factory=list)
    composition_requested: dict[str, int] = field(default_factory=dict)
    composition_built: dict[str, int] = field(default_factory=dict)

    @property
    def abstention_questions(self) -> list[Question]:
        return [q for q in self.questions if q.is_abstention]

    def cohort_membership(self) -> list[dict[str, Any]]:
        return [
            {"question_id": question.question_id, "cohort": cohort}
            for question in self.questions
            for cohort in question.cohorts
        ]

    def facts_for(self, question: Question) -> list[ExpectedFact]:
        return [self.facts[f] for f in question.required_fact_ids if f in self.facts]

    def as_summary(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "topic_id": self.topic_id,
            "questions": len(self.questions),
            "expected_facts": len(self.facts),
            "eligible_facts": sum(1 for f in self.facts.values() if f.eligible),
            "composition_requested": dict(self.composition_requested),
            "composition_built": dict(self.composition_built),
            "exclusions": len(self.exclusions),
            "cohorts": {
                cohort: sum(1 for q in self.questions if cohort in q.cohorts) for cohort in COHORTS
            },
        }


class BenchmarkBuilder:
    """The last role that may see the research trace."""

    def __init__(
        self,
        config: LabConfig,
        *,
        model: ModelProvider | None = None,
        ledger: CostLedger | None = None,
        tracer: Any = None,
        run_id: str = "",
    ) -> None:
        self.config = config
        self.model = model
        self.ledger = ledger
        self.tracer = tracer
        self.run_id = run_id

    def build(self, topic: Topic, state: CollectionState) -> BuiltBenchmark:
        ledger_digest = digest(sorted(claim.claim_id for claim in state.claims.values()))
        requested = {
            COMPOSITION_ALIASES.get(key, key): value
            for key, value in self.config.benchmark.composition.items()
        }
        version = ids.benchmark_version(ledger_digest, requested)
        built = BuiltBenchmark(version=version, topic_id=topic.id)
        built.composition_requested = dict(requested)

        eligible = sorted(
            (claim for claim in state.eligible_claims() if len(claim.claim_text) >= 60),
            key=lambda claim: claim.claim_id,
        )
        by_facet: dict[str, list[ClaimRecord]] = defaultdict(list)
        for claim in eligible:
            for facet in claim.facet_ids or ["unassigned"]:
                by_facet[facet].append(claim)

        builders = {
            "atomic_fact": self._atomic,
            "synthesis": self._synthesis,
            "mechanism": self._mechanism,
            "contradiction": self._contradiction,
            "temporal": self._temporal,
            "source_id": self._source_identification,
            "abstention": self._abstention,
        }
        used_claims: set[str] = set()
        for kind, wanted in requested.items():
            builder = builders.get(kind)
            if builder is None:
                built.exclusions.append(Exclusion("composition", f"unknown question type '{kind}'"))
                continue
            produced = builder(topic, state, by_facet, used_claims, wanted, version, built)
            built.composition_built[kind] = len(produced)
            if len(produced) < wanted:
                built.exclusions.append(
                    Exclusion(
                        "composition_shortfall",
                        f"{kind}: wanted {wanted}, grounded {len(produced)}",
                        {"type": kind, "wanted": wanted, "built": len(produced)},
                    )
                )
            built.questions.extend(produced)

        self._assign_cohorts(built)
        self._record(built)
        return built

    # -- generators --------------------------------------------------------
    def _atomic(
        self,
        topic: Topic,
        state: CollectionState,
        by_facet: dict[str, list[ClaimRecord]],
        used: set[str],
        wanted: int,
        version: str,
        built: BuiltBenchmark,
    ) -> list[Question]:
        out: list[Question] = []
        for claim in self._ordered(by_facet, used):
            if len(out) >= wanted:
                break
            subject, answer = _split_terms(claim.claim_text)
            if len(subject) < 2 or len(answer) < MINIMUM_ANSWER_TERMS:
                built.exclusions.append(
                    Exclusion(
                        "claim_not_gradable",
                        "the claim has too little content beyond its own subject to ask about",
                        {"claim_id": claim.claim_id, "type": "atomic_fact"},
                    )
                )
                continue
            text = (
                f"What does the collected literature establish about "
                f"{', '.join(subject)} in the context of {topic.title.lower()}?"
            )
            question = self._question(text, "atomic_fact", "direct", topic, claim, version)
            fact = self._fact_from_claim(
                claim, question.question_id, kind="all_of", answer_terms=answer
            )
            self._attach(built, question, [fact], state, claim)
            used.add(claim.claim_id)
            out.append(question)
        return out

    def _synthesis(
        self,
        topic: Topic,
        state: CollectionState,
        by_facet: dict[str, list[ClaimRecord]],
        used: set[str],
        wanted: int,
        version: str,
        built: BuiltBenchmark,
    ) -> list[Question]:
        out: list[Question] = []
        for facet, claims in sorted(by_facet.items()):
            if len(out) >= wanted:
                break
            by_source: dict[str, list[ClaimRecord]] = defaultdict(list)
            for claim in claims:
                if claim.claim_id not in used:
                    by_source[claim.source_id].append(claim)
            sources = sorted(by_source)
            for index in range(0, len(sources) - 1, 2):
                if len(out) >= wanted:
                    break
                left = by_source[sources[index]][0]
                right = by_source[sources[index + 1]][0]
                left_subject, left_answer = _split_terms(left.claim_text, subject=2)
                right_subject, right_answer = _split_terms(right.claim_text, subject=2)
                subject = sorted(set(left_subject) | set(right_subject))
                left_answer = [term for term in left_answer if term not in subject]
                right_answer = [term for term in right_answer if term not in subject]
                if (
                    len(subject) < 3
                    or len(left_answer) < MINIMUM_ANSWER_TERMS
                    or len(right_answer) < MINIMUM_ANSWER_TERMS
                ):
                    continue
                text = (
                    f"Combining independent sources, what can be said about {', '.join(subject[:4])}? "
                    "Name what each source contributes."
                )
                question = self._question(
                    text, "synthesis", "multi_hop", topic, left, version, extra_claim=right
                )
                facts = [
                    self._fact_from_claim(
                        left,
                        question.question_id,
                        kind="all_of",
                        suffix="a",
                        answer_terms=left_answer,
                    ),
                    self._fact_from_claim(
                        right,
                        question.question_id,
                        kind="all_of",
                        suffix="b",
                        answer_terms=right_answer,
                    ),
                ]
                self._attach(built, question, facts, state, left, right)
                question.facet_ids = [facet]
                used.update({left.claim_id, right.claim_id})
                out.append(question)
        return out

    def _mechanism(
        self,
        topic: Topic,
        state: CollectionState,
        by_facet: dict[str, list[ClaimRecord]],
        used: set[str],
        wanted: int,
        version: str,
        built: BuiltBenchmark,
    ) -> list[Question]:
        out: list[Question] = []
        for claim in self._ordered(by_facet, used):
            if len(out) >= wanted:
                break
            lowered = claim.claim_text.lower()
            if not any(marker in lowered for marker in MECHANISM_MARKERS):
                continue
            subject, answer = _split_terms(claim.claim_text)
            if len(subject) < 2 or len(answer) < MINIMUM_ANSWER_TERMS:
                continue
            text = f"By what mechanism does the literature explain {', '.join(subject)}?"
            question = self._question(text, "mechanism", "multi_hop", topic, claim, version)
            fact = self._fact_from_claim(
                claim, question.question_id, kind="all_of", answer_terms=answer
            )
            self._attach(built, question, [fact], state, claim)
            used.add(claim.claim_id)
            out.append(question)
        return out

    def _contradiction(
        self,
        topic: Topic,
        state: CollectionState,
        by_facet: dict[str, list[ClaimRecord]],
        used: set[str],
        wanted: int,
        version: str,
        built: BuiltBenchmark,
    ) -> list[Question]:
        out: list[Question] = []
        for contradiction in sorted(
            state.contradictions.values(), key=lambda c: c.contradiction_id
        ):
            if len(out) >= wanted:
                break
            claims = [state.claims[c] for c in contradiction.claim_ids if c in state.claims]
            if len(claims) < 2:
                claims = [
                    c for c in state.claims.values() if c.source_id in contradiction.source_ids
                ][:2]
            if len(claims) < 2 or len({c.source_id for c in claims}) < 2:
                built.exclusions.append(
                    Exclusion(
                        "contradiction_ungrounded",
                        "fewer than two independent sources carry this disagreement",
                        {"contradiction_id": contradiction.contradiction_id},
                    )
                )
                continue
            left, right = claims[0], claims[1]
            text = (
                f"Do the collected sources agree about {contradiction.about or 'this question'}? "
                "State the positions and say which sources hold them."
            )
            question = self._question(
                text, "contradiction", "adversarial", topic, left, version, extra_claim=right
            )
            about_terms = set(content_terms(contradiction.about or ""))
            matcher = FactMatcher(
                kind="disagreement",
                groups=[
                    _terms_group(left.claim_text, exclude=about_terms),
                    _terms_group(right.claim_text, exclude=about_terms),
                ],
            )
            fact = ExpectedFact(
                fact_id=ids.fact_id(contradiction.contradiction_id, question.question_id),
                text=f"sources disagree about {contradiction.about}",
                matcher=matcher,
                claim_ids=[left.claim_id, right.claim_id],
                source_ids=sorted({left.source_id, right.source_id}),
                confidence_basis="direct_text",
            )
            self._attach(built, question, [fact], state, left, right)
            contradiction.converted_to_benchmark_case = True
            used.update({left.claim_id, right.claim_id})
            out.append(question)
        return out

    def _temporal(
        self,
        topic: Topic,
        state: CollectionState,
        by_facet: dict[str, list[ClaimRecord]],
        used: set[str],
        wanted: int,
        version: str,
        built: BuiltBenchmark,
    ) -> list[Question]:
        out: list[Question] = []
        for facet, claims in sorted(by_facet.items()):
            if len(out) >= wanted:
                break
            dated = [
                (state.sources[c.source_id].candidate.published_at, c)
                for c in claims
                if c.claim_id not in used
                and c.source_id in state.sources
                and state.sources[c.source_id].candidate.published_at
            ]
            dated.sort(key=lambda pair: (str(pair[0]), pair[1].claim_id))
            if len(dated) < 2 or dated[0][0] == dated[-1][0]:
                continue
            earlier_date, earlier = dated[0]
            _later_date, later = dated[-1]
            as_of = str(earlier_date)
            subject, answer = _split_terms(earlier.claim_text)
            if len(subject) < 2 or len(answer) < MINIMUM_ANSWER_TERMS:
                continue
            text = f"As of {as_of}, what did the literature hold about {', '.join(subject)}?"
            question = self._question(text, "temporal", "multi_hop", topic, earlier, version)
            question.as_of = as_of
            fact = self._fact_from_claim(
                earlier, question.question_id, kind="all_of", answer_terms=answer
            )
            self._attach(built, question, [fact], state, earlier)
            question.facet_ids = [facet]
            # The later claim is a known negative *for this as_of*: an answer
            # that reaches for it has answered a different question.
            question.known_negative_source_ids = sorted({later.source_id} - {earlier.source_id})
            used.add(earlier.claim_id)
            out.append(question)
        return out

    def _source_identification(
        self,
        topic: Topic,
        state: CollectionState,
        by_facet: dict[str, list[ClaimRecord]],
        used: set[str],
        wanted: int,
        version: str,
        built: BuiltBenchmark,
    ) -> list[Question]:
        out: list[Question] = []
        for claim in self._ordered(by_facet, used):
            if len(out) >= wanted:
                break
            record = state.sources.get(claim.source_id)
            if record is None:
                continue
            identifiers = [
                value
                for value in (
                    record.candidate.stable_identifier,
                    record.candidate.title,
                    record.source_id,
                )
                if value
            ]
            if not identifiers:
                continue
            text = (
                "Which source in the knowledge base establishes that "
                f"{truncate(claim.claim_text, 160)}? Name it."
            )
            question = self._question(text, "source_id", "direct", topic, claim, version)
            fact = ExpectedFact(
                fact_id=ids.fact_id(claim.claim_id + ":source", question.question_id),
                text=f"the source is {identifiers[0]}",
                matcher=FactMatcher(kind="source", identifiers=identifiers),
                claim_ids=[claim.claim_id],
                source_ids=[claim.source_id],
                confidence_basis="structured_metadata",
            )
            self._attach(built, question, [fact], state, claim)
            used.add(claim.claim_id)
            out.append(question)
        return out

    def _abstention(
        self,
        topic: Topic,
        state: CollectionState,
        by_facet: dict[str, list[ClaimRecord]],
        used: set[str],
        wanted: int,
        version: str,
        built: BuiltBenchmark,
    ) -> list[Question]:
        """Cases the corpus deterministically does not contain.

        Built from facets the collection **explicitly did not cover**, and
        verified against the corpus: an abstention case with retrievable
        supporting evidence is a broken abstention case, not a hard question.
        """

        corpus = " ".join(
            (record.candidate.title or "") + " " + (record.candidate.abstract or "")
            for record in state.retained_sources()
        ).lower()
        out: list[Question] = []
        uncovered = [facet for facet in topic.facets if not state.sources_for_facet(facet.id)]
        for facet in uncovered:
            if len(out) >= wanted:
                break
            probe = f"quantitative dose-response tables for {facet.label.lower()}"
            if all(term in corpus for term in content_terms(facet.label)[:2]):
                built.exclusions.append(
                    Exclusion(
                        "abstention_unsafe",
                        f"facet {facet.id} has no sources but its terms appear in the corpus; "
                        "an abstention case here could be answered",
                        {"facet_id": facet.id},
                    )
                )
                continue
            text = f"What does the knowledge base contain about {probe}?"
            question = Question(
                question_id=ids.question_id(text, version),
                topic_id=topic.id,
                text=text,
                type="abstention",
                difficulty="adversarial",
                facet_ids=[facet.id],
            )
            fact = ExpectedFact(
                fact_id=ids.fact_id(f"abstain:{facet.id}", question.question_id),
                text="the region does not contain this",
                matcher=FactMatcher(kind="abstain"),
                confidence_basis="structured_metadata",
            )
            question.required_fact_ids = [fact.fact_id]
            built.facts[fact.fact_id] = fact
            built.evidence[question.question_id] = ExpectedEvidence(
                question_id=question.question_id
            )
            out.append(question)

        while len(out) < wanted:
            index = len(out)
            nonce = f"{topic.id}-absent-{index}"
            token = "zzq" + digest(nonce).removeprefix("sha256:")[:10]
            if token in corpus:  # pragma: no cover - a 40-bit collision
                continue
            text = (
                f"What does the knowledge base say about the {token} protocol as applied to "
                f"{topic.title.lower()}?"
            )
            question = Question(
                question_id=ids.question_id(text, version),
                topic_id=topic.id,
                text=text,
                type="abstention",
                difficulty="adversarial",
            )
            fact = ExpectedFact(
                fact_id=ids.fact_id(nonce, question.question_id),
                text="the region does not contain this",
                matcher=FactMatcher(kind="abstain"),
                confidence_basis="structured_metadata",
            )
            question.required_fact_ids = [fact.fact_id]
            built.facts[fact.fact_id] = fact
            built.evidence[question.question_id] = ExpectedEvidence(
                question_id=question.question_id
            )
            out.append(question)
        return out

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _ordered(by_facet: dict[str, list[ClaimRecord]], used: set[str]) -> list[ClaimRecord]:
        """Claims in a stable order, one facet at a time.

        Round-robin across facets rather than facet-by-facet, so a
        composition that runs out of room does not spend every atomic-fact
        slot on the first facet.
        """

        buckets = [
            sorted(claims, key=lambda c: c.claim_id) for _facet, claims in sorted(by_facet.items())
        ]
        ordered: list[ClaimRecord] = []
        index = 0
        while any(index < len(bucket) for bucket in buckets):
            for bucket in buckets:
                if index < len(bucket) and bucket[index].claim_id not in used:
                    ordered.append(bucket[index])
            index += 1
        return ordered

    def _question(
        self,
        text: str,
        kind: str,
        difficulty: str,
        topic: Topic,
        claim: ClaimRecord,
        version: str,
        *,
        extra_claim: ClaimRecord | None = None,
    ) -> Question:
        created = [claim.claim_id] + ([extra_claim.claim_id] if extra_claim else [])
        return Question(
            question_id=ids.question_id(text, version),
            topic_id=topic.id,
            text=text,
            type=kind,
            difficulty=difficulty,
            created_from_claim_ids=created,
            facet_ids=list(claim.facet_ids),
            subtopic_id=claim.subtopic_id,
        )

    def _fact_from_claim(
        self,
        claim: ClaimRecord,
        question_id: str,
        *,
        kind: str,
        suffix: str = "",
        answer_terms: Sequence[str] | None = None,
    ) -> ExpectedFact:
        terms = (
            list(answer_terms)
            if answer_terms is not None
            else salient_terms(claim.claim_text, limit=6)
        )
        groups = [[term] for term in terms[:3]]
        found = numbers(claim.claim_text)
        matcher = (
            FactMatcher(kind="numeric", value=found[0], groups=groups)
            if found and kind == "all_of"
            else FactMatcher(kind=kind, groups=groups)  # type: ignore[arg-type]
        )
        return ExpectedFact(
            fact_id=ids.fact_id(claim.claim_id + suffix, question_id),
            text=truncate(claim.claim_text, 300),
            matcher=matcher,
            claim_ids=[claim.claim_id],
            source_ids=[claim.source_id],
            facet_id=claim.facet_ids[0] if claim.facet_ids else None,
            confidence_basis=claim.confidence_basis,
        )

    def _attach(
        self,
        built: BuiltBenchmark,
        question: Question,
        facts: list[ExpectedFact],
        state: CollectionState,
        *claims: ClaimRecord,
    ) -> None:
        # Guard at the point of construction, not only at the door. The
        # leakage checker refuses a question whose wording satisfies its own
        # expected fact; building one and then refusing it wastes the slot and
        # shrinks the composition. This rebuilds the matcher from what is left
        # once the question's wording is taken out of it.
        hardened: list[ExpectedFact] = []
        for fact in facts:
            if fact.matcher.kind == "abstain" or not fact.matcher.matches(question.text):
                hardened.append(fact)
                continue
            rebuilt = _harden(fact, question.text)
            if rebuilt is None:
                built.exclusions.append(
                    Exclusion(
                        "answer_in_question",
                        "no matcher survives removing the question's own wording",
                        {"fact_id": fact.fact_id, "question_id": question.question_id},
                    )
                )
                continue
            hardened.append(rebuilt)
        facts = hardened

        eligible = [fact for fact in facts if fact.eligible]
        for fact in facts:
            if not fact.eligible:
                built.exclusions.append(
                    Exclusion(
                        "fact_ineligible",
                        "matcher is not deterministic, or the claim is researcher inference",
                        {"fact_id": fact.fact_id, "question_id": question.question_id},
                    )
                )
        question.required_fact_ids = [fact.fact_id for fact in eligible]
        for fact in eligible:
            built.facts[fact.fact_id] = fact
        source_ids = sorted({claim.source_id for claim in claims})
        question.acceptable_evidence_source_ids = source_ids
        artifacts = [
            state.sources[source_id].artifact_id
            for source_id in source_ids
            if source_id in state.sources and state.sources[source_id].artifact_id
        ]
        negatives = sorted(
            {
                record.source_id
                for record in state.retained_sources()
                if record.source_id not in source_ids
                and not (set(record.facet_ids) & set(question.facet_ids))
            }
        )[:5]
        question.known_negative_source_ids = negatives
        built.evidence[question.question_id] = ExpectedEvidence(
            question_id=question.question_id,
            acceptable_source_ids=source_ids,
            acceptable_artifact_ids=[a for a in artifacts if a],
            known_negative_source_ids=negatives,
        )

    def _assign_cohorts(self, built: BuiltBenchmark) -> None:
        """Deterministic cohort assignment.

        Three rules that are not negotiable, and are enforced here rather than
        remembered later:

        * an **abstention** case is never ``learned`` - there is nothing for a
          memory rule to learn from a question whose answer is "not here";
        * a **control** question must be one no steering rule can fire on, so
          the control cohort is drawn from abstention and invariant material
          plus questions the learned cohort never touched;
        * ``temporal_holdout`` questions never appear in ``learned``, which is
          what makes ``learned - holdout`` a memorisation detector rather than
          a restatement.
        """

        split = self.config.benchmark.cohorts
        rng = random.Random(self.config.experiment.seed)
        pool = sorted(built.questions, key=lambda q: q.question_id)
        rng.shuffle(pool)

        total = len(pool)
        counts = {
            "anchor": round(total * split.anchor_fraction),
            "learned": round(total * split.learned_fraction),
            "temporal_holdout": round(total * split.temporal_holdout_fraction),
            "control": round(total * split.control_fraction),
        }
        counts["invariant"] = max(0, total - sum(counts.values()))

        learnable = [q for q in pool if not q.is_abstention]
        rest = [q for q in pool if q.is_abstention]
        assigned: dict[str, str] = {}

        def take(cohort: str, source: list[Question], number: int) -> None:
            taken = 0
            for question in list(source):
                if taken >= number:
                    break
                if question.question_id in assigned:
                    continue
                assigned[question.question_id] = cohort
                taken += 1

        take("learned", learnable, counts["learned"])
        take("temporal_holdout", learnable, counts["temporal_holdout"])
        take("anchor", learnable, counts["anchor"])
        take("control", rest + learnable, counts["control"])
        take("invariant", rest + pool, counts["invariant"])
        for question in pool:
            assigned.setdefault(question.question_id, "anchor")

        for question in built.questions:
            question.cohorts = [assigned[question.question_id]]

    def _record(self, built: BuiltBenchmark) -> None:
        if self.tracer is None:
            return
        self.tracer.emit(
            "benchmark.built",
            payload=built.as_summary(),
            topic_id=built.topic_id,
        )
        if self.model is not None and self.ledger is not None:
            spec = self.config.role("benchmark_builder")
            request = ModelRequest(
                role="benchmark_builder",
                schema="benchmark",
                system=load_prompt("benchmark_builder"),
                user="Return the question set exactly as constructed.",
                context={"questions": {"questions": [q.as_dict() for q in built.questions]}},
                max_output_tokens=spec.max_output_tokens,
                temperature=spec.temperature,
            )
            with self.ledger.spend(
                bucket="benchmark",
                role="benchmark_builder",
                model=spec.model,
                prompt=request.prompt_text,
                max_output_tokens=spec.max_output_tokens,
            ) as cost:
                response = self.model.complete(request)
                cost.input_tokens = response.input_tokens
                cost.output_tokens = response.output_tokens
            self.tracer.emit("cost.model_call", payload=cost.as_payload())


def _harden(fact: ExpectedFact, question_text: str) -> ExpectedFact | None:
    """Rebuild a matcher the question's own wording would satisfy.

    A numeric matcher whose value is quoted in the question falls back to its
    term groups; a term matcher drops the groups the question already
    contains. Fewer than two groups left means the claim has nothing to ask
    about that the question does not already say, and the fact is dropped.
    """

    matcher = fact.matcher
    surviving = [
        group
        for group in matcher.groups
        if not any(normalise(alternative) in normalise(question_text) for alternative in group)
    ]
    if len(surviving) < 2:
        return None
    rebuilt = FactMatcher(kind="all_of", groups=surviving)
    if rebuilt.matches(question_text):
        return None
    return ExpectedFact(
        fact_id=fact.fact_id,
        text=fact.text,
        matcher=rebuilt,
        claim_ids=list(fact.claim_ids),
        source_ids=list(fact.source_ids),
        facet_id=fact.facet_id,
        required=fact.required,
        confidence_basis=fact.confidence_basis,
    )


def _terms_group(text: str, *, exclude: set[str] | None = None) -> list[str]:
    blocked = exclude or set()
    terms = [term for term in salient_terms(text, limit=6) if term not in blocked]
    return terms[:3] or [term for term in content_terms(text) if term not in blocked][:2]
