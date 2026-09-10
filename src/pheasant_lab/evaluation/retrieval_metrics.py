"""Retrieval and evidence metrics, per question.

The rule that shapes every function here: **an unjudged result is unknown and
is never silently counted as a negative.** A question with no known-positive
set is excluded from the denominator rather than scored as a failure, and a
metric whose evidence minimum is unmet reports ``insufficient_evidence`` with
``value: None``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from ..benchmark.question_types import ExpectedEvidence, Question
from .metric import MetricResult, MetricScope, insufficient

VERSION = "retrieval-1"


def _positive(result: Mapping[str, Any], evidence: ExpectedEvidence) -> bool:
    return str(result.get("source_id") or "") in set(evidence.acceptable_source_ids) or str(
        result.get("artifact_id") or ""
    ) in set(evidence.acceptable_artifact_ids)


def _negative(result: Mapping[str, Any], evidence: ExpectedEvidence) -> bool:
    return str(result.get("source_id") or "") in set(evidence.known_negative_source_ids)


def ranked_results(answer: Mapping[str, Any], *, k: int) -> list[dict[str, Any]]:
    """The fused, deduplicated ranking this answer actually saw.

    Rank is assigned across search rounds in the order results arrived, and a
    repeated artifact keeps its best rank - which is what a reader
    experiences, and therefore what a rank-sensitive metric must score.
    """

    seen: dict[str, dict[str, Any]] = {}
    position = 0
    for call in answer.get("search_calls") or []:
        for result in call.get("results") or []:
            key = str(result.get("artifact_id") or "")
            if not key:
                continue
            position += 1
            if key in seen:
                continue
            row = dict(result)
            row["fused_rank"] = position
            seen[key] = row
    ordered = sorted(seen.values(), key=lambda row: row["fused_rank"])
    return ordered[:k]


def compute(
    question: Question,
    evidence: ExpectedEvidence,
    answer: Mapping[str, Any],
    *,
    k: int,
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
    results = ranked_results(answer, k=k)
    positives = set(evidence.acceptable_source_ids) | set(evidence.acceptable_artifact_ids)
    out: list[MetricResult] = []

    # -- query evidence coverage ------------------------------------------
    out.append(
        MetricResult(
            metric="query_evidence_coverage",
            version=VERSION,
            classification="descriptive",
            scope=scope,
            formula="1 if the question has a known-positive set, else 0",
            numerator=1.0 if positives else 0.0,
            denominator=1.0,
            value=1.0 if positives else 0.0,
            operand_ids=[str(answer.get("answer_id"))],
            claim_supported="whether this question can be scored on retrieval at all",
            claim_not_supported="anything about how well retrieval performed",
            limitation="a question with no known positives is excluded from retrieval denominators, "
            "not counted as a retrieval failure",
        ).validate()
    )

    if not positives:
        for name in (
            "known_positive_hit_at_k",
            "known_positive_recall_at_k",
            "known_positive_reciprocal_rank",
            "evidence_weighted_dcg",
        ):
            out.append(
                insufficient(
                    name,
                    VERSION,
                    scope,
                    formula="requires a known-positive set",
                    reason="this question has no known-positive evidence set; scoring it would "
                    "manufacture a failure the benchmark cannot support",
                )
            )
        return out

    hits = [row for row in results if _positive(row, evidence)]
    negatives = [row for row in results if _negative(row, evidence)]
    judged = [row for row in results if _positive(row, evidence) or _negative(row, evidence)]
    first_hit_rank = min((row["fused_rank"] for row in hits), default=None)

    out.append(
        MetricResult(
            metric="result_evidence_coverage_at_k",
            version=VERSION,
            classification="descriptive",
            scope=scope,
            formula="judged results in the top k / results returned in the top k",
            numerator=float(len(judged)),
            denominator=float(len(results)),
            operand_ids=[str(row.get("artifact_id")) for row in results],
            claim_supported="how much of what this arm saw the benchmark can judge",
            claim_not_supported="that the unjudged remainder was wrong",
            limitation="unjudged results are unknown; they are neither positives nor negatives",
        ).validate(minimum_denominator=1.0)
    )

    out.append(
        MetricResult(
            metric="known_positive_hit_at_k",
            version=VERSION,
            classification="primary",
            scope=scope,
            formula=f"1 if any known positive appears in the top {k}, else 0",
            numerator=1.0 if hits else 0.0,
            denominator=1.0,
            value=1.0 if hits else 0.0,
            operand_ids=[str(row.get("artifact_id")) for row in hits],
            threshold=None,
            claim_supported=f"whether the arm surfaced any acceptable evidence within {k}",
            claim_not_supported="that the answer used it, or that it was ranked usefully",
            limitation="a hit at rank k and a hit at rank 1 score identically here; see the "
            "reciprocal rank for position",
        ).validate()
    )

    retrieved_positive_sources = {
        str(row.get("source_id") or row.get("artifact_id") or "") for row in hits
    }
    out.append(
        MetricResult(
            metric="known_positive_recall_at_k",
            version=VERSION,
            classification="primary",
            scope=scope,
            formula=f"distinct known positives retrieved in the top {k} / known positives",
            numerator=float(len(retrieved_positive_sources & positives)),
            denominator=float(len(set(evidence.acceptable_source_ids)) or len(positives)),
            operand_ids=sorted(retrieved_positive_sources & positives),
            claim_supported="the share of acceptable evidence this arm reached",
            claim_not_supported="that unreached evidence does not exist in the region",
            limitation="the denominator is the benchmark's known-positive set, not everything in "
            "the corpus that would have supported an answer",
        ).validate()
    )

    reciprocal = 1.0 / first_hit_rank if first_hit_rank else 0.0
    out.append(
        MetricResult(
            metric="known_positive_reciprocal_rank",
            version=VERSION,
            classification="primary",
            scope=scope,
            formula="1 / rank of the first known positive, 0 when none appears",
            numerator=reciprocal,
            denominator=1.0,
            value=reciprocal,
            operand_ids=[str(row.get("artifact_id")) for row in hits[:1]],
            claim_supported="how early acceptable evidence appeared",
            claim_not_supported="how much acceptable evidence there was",
            limitation="zero here means 'no known positive in the top k', which is a retrieval "
            "miss and not evidence that the document is absent from the region",
        ).validate()
    )

    out.append(
        MetricResult(
            metric="negative_exposure_at_k",
            version=VERSION,
            classification="primary",
            scope=scope,
            formula=f"known negatives in the top {k} / results returned",
            numerator=float(len(negatives)),
            denominator=float(len(results)),
            operand_ids=[str(row.get("artifact_id")) for row in negatives],
            claim_supported="how much material the benchmark marked as wrong-for-this-question "
            "the arm was shown",
            claim_not_supported="that the remaining results were relevant",
            limitation="lower is better; the known-negative set is partial, so this is a floor on "
            "exposure rather than a measurement of it",
        ).validate(minimum_denominator=1.0)
    )

    gain = 0.0
    ideal = 0.0
    for index, row in enumerate(results, start=1):
        relevance = 1.0 if _positive(row, evidence) else (-1.0 if _negative(row, evidence) else 0.0)
        gain += relevance / math.log2(index + 1)
    for index in range(1, min(len(positives), len(results)) + 1):
        ideal += 1.0 / math.log2(index + 1)
    out.append(
        MetricResult(
            metric="evidence_weighted_dcg",
            version=VERSION,
            classification="diagnostic",
            scope=scope,
            formula="sum(rel_i / log2(i+1)) / ideal, rel = +1 known positive, -1 known negative, 0 unjudged",
            numerator=gain,
            denominator=ideal if ideal else None,
            operand_ids=[str(row.get("artifact_id")) for row in results],
            claim_supported="the ranking's discounted gain against the benchmark's judgements",
            claim_not_supported="a comparison with any other system's nDCG: the judgements here "
            "are this run's own, and unjudged results contribute zero rather than a graded score",
            limitation="unjudged results contribute nothing, so this understates a ranking that "
            "surfaced good material the benchmark did not judge",
        ).validate()
    )
    return out


def pairwise_proof_accuracy(
    proof_events: Sequence[Mapping[str, Any]],
    results_by_question: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    run_id: str,
    arm_id: str,
) -> MetricResult:
    """Ranked-above accuracy over pairs the proof ledger actually judged.

    Only pairs where one target has positive proof and the other negative are
    counted. Pairs involving an unjudged target are not counted as either
    correct or incorrect, because nobody said anything about them.
    """

    scope = MetricScope(run_id=run_id, arm_id=arm_id)
    polarity: dict[tuple[str, str], str] = {}
    for event in proof_events:
        if event.get("polarity") in {"positive", "negative"} and event.get("arm_id") == arm_id:
            polarity[(str(event["question_id"]), str(event["target_id"]))] = str(event["polarity"])

    correct = 0
    total = 0
    for question_id, results in results_by_question.items():
        ranks = {
            str(row.get("artifact_id")): int(row.get("fused_rank") or row.get("rank") or 0)
            for row in results
        }
        positives = [
            t
            for (q, t), p in polarity.items()
            if q == question_id and p == "positive" and t in ranks
        ]
        negatives = [
            t
            for (q, t), p in polarity.items()
            if q == question_id and p == "negative" and t in ranks
        ]
        for positive in positives:
            for negative in negatives:
                total += 1
                if ranks[positive] < ranks[negative]:
                    correct += 1

    return MetricResult(
        metric="pairwise_proof_accuracy",
        version=VERSION,
        classification="primary",
        scope=scope,
        formula="judged pairs ranked correctly / judged pairs",
        numerator=float(correct),
        denominator=float(total),
        claim_supported="whether the ranking put judged-good material above judged-bad material",
        claim_not_supported="anything about pairs where either side was unjudged",
        limitation="only pairs with explicit positive and negative proof are counted; a sparse "
        "proof ledger makes this a measurement of a small corner of the ranking",
    ).validate()
