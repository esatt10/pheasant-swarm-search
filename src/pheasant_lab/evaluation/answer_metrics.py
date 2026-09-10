"""Answer metrics.

Deterministic wherever an expected fact set exists. Where it does not, the
question is *excluded* from the denominator and the exclusion is reported -
never scored as a miss.

The one place judgement is unavoidable is precision: deciding whether a
returned claim the benchmark did not ask for is *correct*. This module answers
that with a support test - the claim's content must be present in a passage
the arm actually cited - and every precision result says in its ``limitation``
that token containment against a cited passage is not entailment. That is
weaker than a judge and stronger than nothing, and it is stated rather than
smoothed over.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..benchmark.question_types import ExpectedEvidence, ExpectedFact, Question
from ..settings import AnswerMatchingSection
from ..textkit import (
    ABSTENTION_MARKERS,
    DISAGREEMENT_MARKERS,
    containment,
    normalise,
    tokens,
)
from .metric import MetricResult, MetricScope, insufficient

VERSION = "answer-1"

# Framing sentences that assert nothing and must not enter the precision
# denominator. An arm is not penalised for saying "the sources disagree".
_NON_ASSERTIONS = ("sources disagree", "the retrieved evidence", "the available knowledge")


def eligible_claims(answer: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The returned claims that are factual assertions."""

    out: list[dict[str, Any]] = []
    for claim in answer.get("claims") or []:
        text = str(claim.get("text") or "").strip()
        if len(text) < 20:
            continue
        folded = normalise(text)
        if any(marker in folded for marker in _NON_ASSERTIONS) and len(folded) < 80:
            continue
        out.append(dict(claim))
    return out


def retrieved_passages(answer: Mapping[str, Any]) -> dict[str, str]:
    """What this session read, keyed by the id an answer would cite.

    ``read_passages`` first, because it is what the answerer was handed and it
    exists for every arm. The search calls are a fallback for a trace written
    before arms recorded their reading, and they carry the same ids.
    """

    passages: dict[str, str] = {}
    for passage in answer.get("read_passages") or []:
        artifact = str(passage.get("artifact_id") or "")
        if artifact:
            passages.setdefault(artifact, str(passage.get("text") or ""))
    for call in answer.get("search_calls") or []:
        for result in call.get("results") or []:
            artifact = str(result.get("artifact_id") or "")
            if artifact:
                passages.setdefault(
                    artifact, str(result.get("matched_text") or result.get("text") or "")
                )
    return passages


def compute(
    question: Question,
    facts: Sequence[ExpectedFact],
    evidence: ExpectedEvidence,
    answer: Mapping[str, Any],
    matching: AnswerMatchingSection,
    *,
    run_id: str,
    cohort: str | None = None,
) -> list[MetricResult]:
    scope = MetricScope(
        run_id=run_id,
        arm_id=str(answer.get("arm_id")),
        question_id=question.question_id,
        cohort=cohort,
        snapshot_id=answer.get("snapshot_id"),
        topic_id=question.topic_id,
    )
    text = str(answer.get("answer_text") or "")
    abstained = bool(answer.get("abstained"))
    claims = eligible_claims(answer)
    passages = retrieved_passages(answer)
    citations = [c for claim in claims for c in (claim.get("citations") or [])]
    eligible_facts = [fact for fact in facts if fact.eligible]
    ineligible = len(facts) - len(eligible_facts)

    out: list[MetricResult] = []

    # -- recall -----------------------------------------------------------
    if not eligible_facts:
        out.append(
            insufficient(
                "fact_recall",
                VERSION,
                scope,
                formula="required facts returned / known required facts",
                reason="this question carries no deterministically matchable required fact; it is "
                "excluded from the recall denominator rather than counted as a miss",
            )
        )
    else:
        matched = [
            fact
            for fact in eligible_facts
            if fact.matcher.matches(
                text,
                abstained=abstained,
                citations=citations,
                tolerance=matching.numeric_relative_tolerance,
            )
        ]
        out.append(
            MetricResult(
                metric="fact_recall",
                version=VERSION,
                classification="primary",
                scope=scope,
                formula="required facts returned / known required facts",
                numerator=float(len(matched)),
                denominator=float(len(eligible_facts)),
                operand_ids=[fact.fact_id for fact in eligible_facts],
                evidence_refs=[fact.fact_id for fact in matched],
                excluded=ineligible,
                exclusion_reasons={"fact_not_deterministically_matchable": ineligible}
                if ineligible
                else {},
                claim_supported="the share of the facts this benchmark requires that the answer contains",
                claim_not_supported="that the answer is complete, or that unmatched facts are absent "
                "from the region",
                limitation="matching is lexical over a normalised form; a correct answer phrased "
                "entirely in vocabulary the matcher does not list will be scored as a miss",
            ).validate()
        )

    matched_count = len(
        [
            fact
            for fact in eligible_facts
            if fact.matcher.matches(
                text,
                abstained=abstained,
                citations=citations,
                tolerance=matching.numeric_relative_tolerance,
            )
        ]
    )
    correct_claims = 0

    # -- precision --------------------------------------------------------
    if not claims:
        out.append(
            insufficient(
                "fact_precision",
                VERSION,
                scope,
                formula="correct eligible returned facts / eligible returned facts",
                reason="the answer returned no eligible factual claim"
                + (" (it abstained)" if abstained else ""),
            )
        )
        supported = 0
    else:
        correct = 0
        supported = 0
        for claim in claims:
            claim_text = str(claim.get("text") or "")
            cited = [str(c) for c in (claim.get("citations") or [])]
            is_supported = _supported(claim_text, cited, passages, matching)
            if is_supported:
                supported += 1
            matches_expected = any(
                fact.matcher.matches(
                    claim_text,
                    abstained=False,
                    citations=cited,
                    tolerance=matching.numeric_relative_tolerance,
                )
                for fact in eligible_facts
            )
            if matches_expected or is_supported:
                correct += 1
        correct_claims = correct
        out.append(
            MetricResult(
                metric="fact_precision",
                version=VERSION,
                classification="primary",
                scope=scope,
                formula="correct eligible returned facts / eligible returned facts",
                numerator=float(correct),
                denominator=float(len(claims)),
                operand_ids=[str(answer.get("answer_id"))],
                claim_supported="the share of the answer's assertions that either match a required "
                "fact or are supported by a passage the answer cited",
                claim_not_supported="that the remaining assertions are false - only that this "
                "benchmark cannot verify them",
                limitation="'supported' is token containment between the claim and the passage it "
                "cites. That is not entailment: a claim can contain its passage's words and still "
                "misstate it.",
            ).validate()
        )

    # -- F1 ---------------------------------------------------------------
    #
    # Computed from the counts rather than from the two ratios. That matters
    # for exactly one case, and it is the case an abstaining arm produces:
    # with no returned claims, precision has no denominator and reports
    # `insufficient_evidence` - but F1 does have one, because zero true
    # positives is `2TP / (2TP + FP + FN) = 0` whatever precision was. Driving
    # F1 off the ratios would drop every abstention out of the comparison,
    # which flatters the arm that abstained by removing it from its own
    # denominator.
    true_positive = matched_count
    false_negative = len(eligible_facts) - matched_count
    false_positive = len(claims) - correct_claims
    f1_denominator = 2 * true_positive + false_positive + false_negative
    if not eligible_facts and not claims:
        out.append(
            insufficient(
                "fact_f1",
                VERSION,
                scope,
                formula="2TP / (2TP + FP + FN)",
                reason="the question carries no matchable required fact and the answer returned no claim",
            )
        )
    else:
        out.append(
            MetricResult(
                metric="fact_f1",
                version=VERSION,
                classification="primary",
                scope=scope,
                formula="2TP / (2TP + FP + FN)",
                numerator=float(2 * true_positive),
                denominator=float(f1_denominator) if f1_denominator else None,
                value=(2 * true_positive / f1_denominator) if f1_denominator else 0.0,
                substituted=f"2*{true_positive} / (2*{true_positive} + {false_positive} + {false_negative})",
                operand_ids=[fact.fact_id for fact in eligible_facts]
                + [str(answer.get("answer_id"))],
                detail={
                    "true_positive": true_positive,
                    "false_positive": false_positive,
                    "false_negative": false_negative,
                },
                claim_supported="the balance of what the answer got right against what it missed and "
                "what it asserted without support",
                claim_not_supported="anything precision and recall do not each support on their own",
                limitation="an F1 of zero can mean an abstention, which is a different thing from a "
                "wrong answer; `abstention_accuracy` is what separates them",
            ).validate(minimum_denominator=1e-9)
        )

    # -- evidence support and citation validity ---------------------------
    out.append(
        MetricResult(
            metric="evidence_support_rate",
            version=VERSION,
            classification="primary",
            scope=scope,
            formula="claims supported by a cited retrieved passage / eligible claims",
            numerator=float(supported),
            denominator=float(len(claims)),
            operand_ids=[str(answer.get("answer_id"))],
            claim_supported="how much of the answer rests on something the arm actually retrieved",
            claim_not_supported="that the supported claims are true",
            limitation="support is containment against the cited passage, not entailment",
        ).validate()
    )

    valid = (
        [c for c in citations if c in passages]
        if matching.citation_must_be_retrieved
        else citations
    )
    out.append(
        MetricResult(
            metric="citation_validity",
            version=VERSION,
            classification="primary",
            scope=scope,
            formula="citations naming an artifact retrieved in this session / citations",
            numerator=float(len(valid)),
            denominator=float(len(citations)),
            operand_ids=sorted(set(citations)),
            claim_supported="whether the answer's citations point at things it was actually served",
            claim_not_supported="whether the cited passage supports the claim it is attached to",
            limitation="an arm that cites nothing scores insufficient_evidence here rather than "
            "zero; abstention is not a citation failure",
        ).validate()
    )

    out.append(
        MetricResult(
            metric="unsupported_claim_rate",
            version=VERSION,
            classification="primary",
            scope=scope,
            formula="1 - (supported claims / eligible claims)",
            numerator=float(len(claims) - supported),
            denominator=float(len(claims)),
            operand_ids=[str(answer.get("answer_id"))],
            claim_supported="how much of the answer the retrieved evidence does not carry",
            claim_not_supported="that the unsupported part is false",
            limitation="lower is better; an arm that answers from its own prior scores high here, "
            "which is exactly what the Pheasant-only arms exist to detect",
        ).validate()
    )

    # -- type-specific -----------------------------------------------------
    out.extend(_type_specific(question, facts, answer, matching, scope, text, abstained, citations))
    return out


def _type_specific(
    question: Question,
    facts: Sequence[ExpectedFact],
    answer: Mapping[str, Any],
    matching: AnswerMatchingSection,
    scope: MetricScope,
    text: str,
    abstained: bool,
    citations: Sequence[str],
) -> list[MetricResult]:
    out: list[MetricResult] = []

    # Abstention accuracy is scored on every question, not only abstention
    # cases: abstaining on an answerable question is as wrong as answering an
    # unanswerable one, and a metric that only looks at the latter rewards an
    # arm for abstaining everywhere.
    said_no = abstained or any(marker in normalise(text) for marker in ABSTENTION_MARKERS)
    correct = said_no if question.is_abstention else not said_no
    out.append(
        MetricResult(
            metric="abstention_accuracy",
            version=VERSION,
            classification="primary",
            scope=scope,
            formula="1 when the arm abstained exactly on the abstention cases, else 0",
            numerator=1.0 if correct else 0.0,
            denominator=1.0,
            value=1.0 if correct else 0.0,
            operand_ids=[str(answer.get("answer_id"))],
            claim_supported="whether the arm knows when the region cannot answer",
            claim_not_supported="whether a non-abstaining answer was correct",
            limitation="an answer that hedges without abstaining is scored as not abstaining",
        ).validate()
    )

    if question.type == "contradiction":
        marked = bool(set(tokens(text)) & DISAGREEMENT_MARKERS)
        both = all(
            fact.matcher.matches(text, abstained=abstained, citations=list(citations))
            for fact in facts
            if fact.eligible
        )
        value = 1.0 if (marked and both) else 0.0
        out.append(
            MetricResult(
                metric="contradiction_handling_accuracy",
                version=VERSION,
                classification="primary",
                scope=scope,
                formula="1 when the answer names the disagreement and both positions, else 0",
                numerator=value,
                denominator=1.0,
                value=value,
                operand_ids=[fact.fact_id for fact in facts],
                claim_supported="whether the arm reported a disagreement as a disagreement",
                claim_not_supported="whether it characterised either position accurately",
                limitation="'named the disagreement' is a lexical test over a marker list; an "
                "answer that conveys disagreement without any of those words scores zero",
            ).validate()
        )

    if question.type == "temporal":
        as_of_respected = all(
            fact.matcher.matches(text, abstained=abstained, citations=list(citations))
            for fact in facts
            if fact.eligible
        )
        reached_forward = any(
            str(citation) in set(question.known_negative_source_ids) for citation in citations
        )
        value = 1.0 if (as_of_respected and not reached_forward) else 0.0
        out.append(
            MetricResult(
                metric="temporal_validity",
                version=VERSION,
                classification="primary",
                scope=scope,
                formula="1 when the answer holds as of the question's instant and cites nothing later",
                numerator=value,
                denominator=1.0,
                value=value,
                operand_ids=[fact.fact_id for fact in facts],
                claim_supported="whether the arm answered as of the stated instant",
                claim_not_supported="whether the region could have answered it correctly",
                limitation="the forward-reach test uses the benchmark's known-negative set, which "
                "is partial",
            ).validate()
        )

    if question.facet_ids:
        covered = sum(
            1
            for fact in facts
            if fact.eligible
            and fact.facet_id in question.facet_ids
            and fact.matcher.matches(text, abstained=abstained, citations=list(citations))
        )
        wanted = sum(1 for fact in facts if fact.eligible and fact.facet_id in question.facet_ids)
        out.append(
            MetricResult(
                metric="answer_completeness_by_facet",
                version=VERSION,
                classification="diagnostic",
                scope=scope,
                formula="required facts matched within the question's facets / required facts in those facets",
                numerator=float(covered),
                denominator=float(wanted),
                operand_ids=[fact.fact_id for fact in facts if fact.facet_id in question.facet_ids],
                claim_supported="whether the answer covered the facets the question spans",
                claim_not_supported="that a facet with no required fact was covered or missed",
                limitation="facets carrying no deterministically matchable fact are not in the "
                "denominator",
            ).validate()
        )
    return out


def _supported(
    claim_text: str,
    citations: Sequence[str],
    passages: Mapping[str, str],
    matching: AnswerMatchingSection,
) -> bool:
    if matching.require_citation_for_support and not citations:
        return False
    claim_tokens = tokens(claim_text)
    if not claim_tokens:
        return False
    for citation in citations:
        passage = passages.get(str(citation))
        if passage is None:
            continue
        if containment(claim_tokens, tokens(passage)) >= matching.support_overlap_threshold:
            return True
    return False
