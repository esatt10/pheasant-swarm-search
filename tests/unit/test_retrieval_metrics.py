"""Retrieval scoring, and what it refuses to score."""

from __future__ import annotations

import pytest

from pheasant_lab.benchmark.question_types import ExpectedEvidence, Question
from pheasant_lab.evaluation import retrieval_metrics

QUESTION = Question(question_id="q-1", topic_id="t", text="q", type="atomic_fact")


def answer(results, arm="P0"):
    return {
        "answer_id": "a-1",
        "arm_id": arm,
        "question_id": "q-1",
        "search_calls": [{"search_request_id": "s-1", "query": "q", "results": results}],
    }


def row(rank, artifact, source=None):
    return {"rank": rank, "artifact_id": artifact, "source_id": source or artifact}


def score(evidence, results, k=10):
    return {
        m.metric: m
        for m in retrieval_metrics.compute(QUESTION, evidence, answer(results), k=k, run_id="run-1")
    }


def test_without_a_known_positive_set_nothing_is_scored():
    rows = score(ExpectedEvidence(question_id="q-1"), [row(1, "a")])
    for name in (
        "known_positive_hit_at_k",
        "known_positive_recall_at_k",
        "known_positive_reciprocal_rank",
    ):
        assert rows[name].status == "insufficient_evidence"
        assert rows[name].value is None
    assert rows["query_evidence_coverage"].value == 0.0


def test_hit_recall_and_reciprocal_rank():
    evidence = ExpectedEvidence(question_id="q-1", acceptable_source_ids=["s-good", "s-also"])
    rows = score(evidence, [row(1, "a", "s-bad"), row(2, "b", "s-good")])
    assert rows["known_positive_hit_at_k"].value == 1.0
    assert rows["known_positive_recall_at_k"].value == pytest.approx(0.5)
    assert rows["known_positive_reciprocal_rank"].value == pytest.approx(0.5)


def test_a_miss_is_a_zero_reciprocal_rank_and_says_what_that_means():
    evidence = ExpectedEvidence(question_id="q-1", acceptable_source_ids=["s-good"])
    rows = score(evidence, [row(1, "a", "s-bad")])
    assert rows["known_positive_reciprocal_rank"].value == 0.0
    assert (
        "not evidence that the document is absent"
        in rows["known_positive_reciprocal_rank"].limitation
    )


def test_negative_exposure_counts_only_declared_negatives():
    evidence = ExpectedEvidence(
        question_id="q-1", acceptable_source_ids=["s-good"], known_negative_source_ids=["s-bad"]
    )
    rows = score(evidence, [row(1, "a", "s-bad"), row(2, "b", "s-good"), row(3, "c", "s-unjudged")])
    assert rows["negative_exposure_at_k"].value == pytest.approx(1 / 3)
    assert rows["result_evidence_coverage_at_k"].value == pytest.approx(2 / 3)


def test_a_repeated_artifact_keeps_its_best_rank():
    record = {
        "answer_id": "a",
        "arm_id": "P0",
        "question_id": "q-1",
        "search_calls": [
            {"results": [row(1, "a"), row(2, "b")]},
            {"results": [row(1, "b"), row(2, "c")]},
        ],
    }
    ranked = retrieval_metrics.ranked_results(record, k=10)
    assert [r["artifact_id"] for r in ranked] == ["a", "b", "c"]
    assert [r["fused_rank"] for r in ranked] == [1, 2, 4]


def test_the_cut_at_k_is_applied_before_scoring():
    evidence = ExpectedEvidence(question_id="q-1", acceptable_source_ids=["s-good"])
    results = [row(i, f"a{i}", "s-bad") for i in range(1, 5)] + [row(5, "good", "s-good")]
    rows = score(evidence, results, k=3)
    assert rows["known_positive_hit_at_k"].value == 0.0


def test_pairwise_proof_accuracy_ignores_unjudged_pairs():
    proof = [
        {"question_id": "q-1", "arm_id": "P0", "target_id": "good", "polarity": "positive"},
        {"question_id": "q-1", "arm_id": "P0", "target_id": "bad", "polarity": "negative"},
    ]
    results = {"q-1": [row(1, "good"), row(2, "bad"), row(3, "unjudged")]}
    result = retrieval_metrics.pairwise_proof_accuracy(proof, results, run_id="run-1", arm_id="P0")
    assert result.numerator == 1.0
    assert result.denominator == 1.0
