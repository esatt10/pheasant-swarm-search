"""Deterministic answer scoring."""

from __future__ import annotations

import pytest

from pheasant_lab.benchmark.question_types import (
    ExpectedEvidence,
    ExpectedFact,
    FactMatcher,
    Question,
)
from pheasant_lab.evaluation import answer_metrics
from pheasant_lab.settings import AnswerMatchingSection

MATCHING = AnswerMatchingSection()
PASSAGE = "Dsup binds nucleosomes and reduced hydroxyl radical damage by 42 percent in vitro."


def answer(**kwargs):
    base = {
        "answer_id": "answer-1",
        "arm_id": "P0",
        "question_id": "q-test",
        "answer_text": "",
        "claims": [],
        "abstained": False,
        "read_passages": [{"artifact_id": "art-1", "text": PASSAGE}],
        "search_calls": [],
    }
    return {**base, **kwargs}


def score(question, facts, record, evidence=None):
    results = answer_metrics.compute(
        question,
        facts,
        evidence or ExpectedEvidence(question_id=question.question_id),
        record,
        MATCHING,
        run_id="run-1",
    )
    return {row.metric: row for row in results}


def test_a_matching_answer_scores_recall_and_f1(question):
    q, fact = question
    record = answer(
        answer_text="Dsup coats chromatin and cuts hydroxyl radical damage.",
        claims=[
            {
                "text": "Dsup coats chromatin and cuts hydroxyl radical damage.",
                "citations": ["art-1"],
            }
        ],
    )
    rows = score(q, [fact], record)
    assert rows["fact_recall"].value == 1.0
    assert rows["fact_f1"].value == pytest.approx(1.0)
    assert rows["citation_validity"].value == 1.0


def test_an_abstention_scores_f1_zero_rather_than_vanishing(question):
    q, fact = question
    record = answer(answer_text="The region does not contain this.", abstained=True)
    rows = score(q, [fact], record)
    assert rows["fact_precision"].status == "insufficient_evidence"
    assert rows["fact_f1"].status == "ok"
    assert rows["fact_f1"].value == 0.0, (
        "an abstention has zero true positives; dropping it would remove the arm "
        "that abstained from its own denominator"
    )


def test_abstention_accuracy_scores_both_directions(question):
    q, fact = question
    abstention_question = Question(
        question_id="q-abstain",
        topic_id="topic-test",
        text="What does the region say about the zzq protocol?",
        type="abstention",
        required_fact_ids=["fact-abstain"],
    )
    abstain_fact = ExpectedFact(
        fact_id="fact-abstain", text="absent", matcher=FactMatcher(kind="abstain")
    )
    correct = score(
        abstention_question,
        [abstain_fact],
        answer(answer_text="No evidence in the knowledge base.", abstained=True),
    )
    assert correct["abstention_accuracy"].value == 1.0

    wrong = score(
        abstention_question,
        [abstain_fact],
        answer(
            answer_text="The zzq protocol reduces damage.",
            claims=[{"text": "x" * 30, "citations": []}],
        ),
    )
    assert wrong["abstention_accuracy"].value == 0.0

    over_abstained = score(q, [fact], answer(answer_text="No information.", abstained=True))
    assert over_abstained["abstention_accuracy"].value == 0.0


def test_a_citation_to_something_never_read_is_invalid(question):
    q, fact = question
    record = answer(
        answer_text="Dsup reduces hydroxyl radical damage.",
        claims=[
            {"text": "Dsup reduces hydroxyl radical damage in vitro.", "citations": ["art-unseen"]}
        ],
    )
    rows = score(q, [fact], record)
    assert rows["citation_validity"].value == 0.0
    assert rows["evidence_support_rate"].value == 0.0
    assert rows["unsupported_claim_rate"].value == 1.0


def test_a_fact_the_benchmark_cannot_match_is_excluded_not_missed(question):
    q, _fact = question
    unmatchable = ExpectedFact(
        fact_id="fact-x",
        text="something",
        matcher=FactMatcher(kind="all_of", groups=[]),
        confidence_basis="researcher_inference",
    )
    rows = score(q, [unmatchable], answer(answer_text="anything"))
    assert rows["fact_recall"].status == "insufficient_evidence"
    assert rows["fact_recall"].value is None


def test_contradiction_handling_needs_a_marker_and_both_positions():
    q = Question(
        question_id="q-c",
        topic_id="t",
        text="Do the sources agree about Dsup protection?",
        type="contradiction",
        required_fact_ids=["fact-c"],
    )
    fact = ExpectedFact(
        fact_id="fact-c",
        text="sources disagree",
        matcher=FactMatcher(kind="disagreement", groups=[["reduced"], ["no significant"]]),
    )
    one_sided = score(q, [fact], answer(answer_text="Dsup reduced breaks by 38 percent."))
    assert one_sided["contradiction_handling_accuracy"].value == 0.0

    both = score(
        q,
        [fact],
        answer(
            answer_text="Sources disagree: one reports breaks reduced by 38 percent, "
            "another reports no significant reduction."
        ),
    )
    assert both["contradiction_handling_accuracy"].value == 1.0


def test_temporal_validity_fails_when_the_answer_reaches_past_its_as_of():
    q = Question(
        question_id="q-t",
        topic_id="t",
        text="As of 2016-01-01, what did the literature hold?",
        type="temporal",
        as_of="2016-01-01",
        required_fact_ids=["fact-t"],
        known_negative_source_ids=["source-later"],
    )
    fact = ExpectedFact(
        fact_id="fact-t",
        text="early view",
        matcher=FactMatcher(kind="all_of", groups=[["survival"]]),
    )
    reached_forward = score(
        q,
        [fact],
        answer(
            answer_text="Survival was reported at 5000 Gy.",
            claims=[
                {"text": "Survival was reported at 5000 Gy." * 2, "citations": ["source-later"]}
            ],
        ),
    )
    assert reached_forward["temporal_validity"].value == 0.0


def test_numeric_matching_respects_relative_tolerance():
    matcher = FactMatcher(kind="numeric", value=42.0, relative_tolerance=0.05)
    assert matcher.matches("a reduction of 43 percent")
    assert not matcher.matches("a reduction of 60 percent")
