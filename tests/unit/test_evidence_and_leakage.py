"""Typed proof, and the leakage checks."""

from __future__ import annotations

import pytest

from pheasant_lab.benchmark.leakage import check_leakage
from pheasant_lab.benchmark.question_types import ExpectedFact, FactMatcher, Question
from pheasant_lab.evaluation.evidence import ProofLedger
from pheasant_lab.settings import ProofPolicyFile

POLICY = ProofPolicyFile.model_validate(
    {
        "proof": {
            "event_types": {
                "served": {"polarity": "unknown", "base_weight": 0.0},
                "cited": {"polarity": "positive", "base_weight": 0.4},
                "explicit_reject": {"polarity": "negative", "base_weight": 1.0},
                "deterministic_validation_pass": {"polarity": "positive", "base_weight": 1.0},
            },
            "multipliers": {
                "directness": {"direct_text": 1.0, "researcher_inference": 0.0},
                "independence": {"independent_judge": 1.0, "self_reported": 0.3},
                "specificity": {"exact_locator": 1.0, "document_level": 0.7},
                "recency": {"current": 1.0, "superseded": 0.0},
            },
        }
    }
)


def ledger() -> ProofLedger:
    return ProofLedger(POLICY, run_id="run-1")


def test_being_served_is_worth_nothing():
    proof = ledger()
    event = proof.record(
        question_id="q", target_id="a", event_type="served", directness="direct_text"
    )
    assert event.polarity == "unknown"
    assert event.weight == 0.0
    assert proof.summarise().positive_weight == 0.0


def test_an_unknown_event_type_is_refused_rather_than_defaulted():
    with pytest.raises(KeyError, match="not in the proof policy"):
        ledger().record(question_id="q", target_id="a", event_type="invented")


def test_weight_is_the_product_of_four_reported_multipliers():
    proof = ledger()
    event = proof.record(
        question_id="q",
        target_id="a",
        event_type="cited",
        directness="direct_text",
        independence="self_reported",
        specificity="document_level",
        recency="current",
    )
    assert event.weight == pytest.approx(0.4 * 1.0 * 0.3 * 0.7 * 1.0)
    assert set(event.multipliers) == {"directness", "independence", "specificity", "recency"}


def test_an_unclassified_multiplier_level_is_worth_nothing_not_everything():
    proof = ledger()
    event = proof.record(
        question_id="q", target_id="a", event_type="cited", independence="not-in-the-table"
    )
    assert event.weight == 0.0


def test_positive_and_negative_never_cancel_and_conflicts_are_published():
    proof = ledger()
    proof.record(
        question_id="q",
        target_id="a",
        event_type="cited",
        independence="independent_judge",
        specificity="exact_locator",
    )
    proof.record(
        question_id="q",
        target_id="a",
        event_type="explicit_reject",
        independence="independent_judge",
        specificity="exact_locator",
    )
    summary = proof.summarise()
    assert summary.positive_weight > 0
    assert summary.negative_weight > 0
    assert summary.conflict_rate == 1.0
    assert summary.net_weight == pytest.approx(summary.positive_weight - summary.negative_weight)


def test_researcher_inference_can_never_carry_weight():
    proof = ledger()
    event = proof.record(
        question_id="q",
        target_id="a",
        event_type="deterministic_validation_pass",
        directness="researcher_inference",
        independence="independent_judge",
        specificity="exact_locator",
    )
    assert event.weight == 0.0


# -- leakage ---------------------------------------------------------------


def question(text: str, *, cohorts=("anchor",), kind="atomic_fact") -> Question:
    return Question(
        question_id="q-1",
        topic_id="t",
        text=text,
        type=kind,
        required_fact_ids=["f-1"],
        cohorts=list(cohorts),
    )


def test_a_question_containing_its_own_answer_is_invalidating():
    q = question("What did Dsup reduce hydroxyl radical damage by?")
    fact = ExpectedFact(
        fact_id="f-1",
        text="",
        matcher=FactMatcher(kind="all_of", groups=[["hydroxyl"], ["radical"]]),
    )
    report = check_leakage([q], {"f-1": fact})
    assert not report.clean
    assert report.findings[0].check == "answer_in_question"


def test_a_holdout_question_that_created_its_treatment_is_invalidating():
    q = question("What is the mechanism?", cohorts=("temporal_holdout",))
    fact = ExpectedFact(
        fact_id="f-1", text="", matcher=FactMatcher(kind="all_of", groups=[["dsup"]])
    )
    report = check_leakage(
        [q],
        {"f-1": fact},
        memory_records=[
            {"record_id": "m-1", "originating_question_id": "q-1", "text": "unrelated"}
        ],
    )
    assert not report.clean
    assert any(f.check == "holdout_created_treatment" for f in report.findings)


def test_an_abstention_case_the_corpus_can_answer_is_invalidating():
    q = question(
        "What does the corpus say about trehalose accumulation kinetics?", kind="abstention"
    )
    fact = ExpectedFact(fact_id="f-1", text="", matcher=FactMatcher(kind="abstain"))
    report = check_leakage(
        [q],
        {"f-1": fact},
        corpus_texts={"art-1": "trehalose accumulation kinetics were measured across species"},
    )
    assert not report.clean
    assert any(f.check == "abstention_answerable" for f in report.findings)


def test_a_benchmark_file_in_the_namespace_is_invalidating():
    q = question("anything")
    fact = ExpectedFact(
        fact_id="f-1", text="", matcher=FactMatcher(kind="all_of", groups=[["zzz"]])
    )
    report = check_leakage([q], {"f-1": fact}, namespace_paths=["topic/expected-facts.jsonl"])
    assert not report.clean
    assert any(f.check == "benchmark_in_namespace" for f in report.findings)


def test_a_clean_set_reports_clean():
    q = question("What does the literature establish about chromatin?")
    fact = ExpectedFact(
        fact_id="f-1", text="", matcher=FactMatcher(kind="all_of", groups=[["forty"], ["percent"]])
    )
    report = check_leakage([q], {"f-1": fact}, corpus_texts={"a": "unrelated text"})
    assert report.clean
