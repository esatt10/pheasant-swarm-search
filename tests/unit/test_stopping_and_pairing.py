"""The stop decision and the pairing policy."""

from __future__ import annotations

import pytest

from pheasant_lab.orchestration.state import (
    ClaimRecord,
    CollectionState,
    ContradictionRecord,
    RoundRecord,
    SourceRecord,
    SourceState,
)
from pheasant_lab.orchestration.stopping import StoppingCalculus
from pheasant_lab.providers.base import SourceCandidate
from pheasant_lab.settings import Facet, SourceAuthority, StoppingSection, Topic

TOPIC = Topic(
    id="topic-1",
    title="t",
    facets=[Facet(id="f1", label="one", weight=1.0)],
    source_authority=SourceAuthority(),
)
CONFIG = StoppingSection(
    minimum_sources_per_subtopic=2,
    minimum_independent_source_families=2,
    minimum_review_or_primary_sources=1,
    marginal_unique_claim_window=2,
    marginal_unique_claim_threshold=0.10,
    consecutive_saturated_rounds=1,
    maximum_duplicate_rate=0.5,
)


def state_with(sources: int, *, families: int = 2) -> CollectionState:
    state = CollectionState("run-1", TOPIC.id)
    for index in range(sources):
        candidate = SourceCandidate(
            provider="fixtures",
            title=f"paper {index}",
            stable_identifier=f"10.1/{index}",
            abstract="body",
            family_key=f"lab-{index % families}",
            source_type="journal_article",
        )
        record = SourceRecord(
            source_id=candidate.source_id,
            candidate=candidate,
            subtopic_id="sub-1",
            facet_ids=["f1"],
            state=SourceState.indexed,
            researcher_agent_id="agent-1",
            discovery_event_id="event-1",
        )
        state.add_source(record)
        state.add_claim(
            ClaimRecord(
                claim_id=f"claim-{index}",
                claim_text="a claim",
                claim_type="observation",
                source_id=record.source_id,
                locator="abstract",
                support="supports",
                confidence_basis="direct_text",
                researcher_agent_id="agent-1",
                subtopic_id="sub-1",
                facet_ids=["f1"],
            )
        )
    state.rounds.append(
        RoundRecord(
            number=1,
            subtopics=["sub-1"],
            new_eligible_claims=0,
            eligible_claims_before=sources,
            acquired=sources,
            duplicates=0,
            cost_usd=0.0,
        )
    )
    return state


def test_sufficient_requires_every_condition():
    calculus = StoppingCalculus(CONFIG, TOPIC)
    decision = calculus.evaluate(state_with(4), receipt_rate=(4, 4), evaluation_reserve_intact=True)
    assert decision.outcome == "sufficient"
    assert decision.unmet == []


def test_a_short_facet_blocks_sufficiency():
    calculus = StoppingCalculus(CONFIG, TOPIC)
    decision = calculus.evaluate(state_with(1), receipt_rate=(1, 1), evaluation_reserve_intact=True)
    assert decision.outcome == "continue"
    assert "facet_minimums" in decision.unmet


def test_budget_exhaustion_is_never_renamed_sufficient():
    calculus = StoppingCalculus(CONFIG, TOPIC)
    decision = calculus.evaluate(
        state_with(4), receipt_rate=(4, 4), evaluation_reserve_intact=False, budget_exhausted=True
    )
    assert decision.outcome == "stopped_budget_incomplete"
    assert decision.sufficient is False


def test_time_exhaustion_reports_itself():
    calculus = StoppingCalculus(CONFIG, TOPIC)
    decision = calculus.evaluate(
        state_with(4), receipt_rate=(4, 4), evaluation_reserve_intact=True, time_exhausted=True
    )
    assert decision.outcome == "stopped_time_incomplete"


def test_an_unresolved_critical_contradiction_blocks_sufficiency():
    state = state_with(4)
    state.contradictions["c-1"] = ContradictionRecord(
        contradiction_id="c-1",
        about="x",
        claim_ids=[],
        source_ids=[],
        severity="critical",
        subtopic_id="sub-1",
    )
    decision = StoppingCalculus(CONFIG, TOPIC).evaluate(
        state, receipt_rate=(4, 4), evaluation_reserve_intact=True
    )
    assert "critical_contradictions" in decision.unmet


def test_a_contradiction_converted_to_a_benchmark_case_is_closed():
    state = state_with(4)
    state.contradictions["c-1"] = ContradictionRecord(
        contradiction_id="c-1",
        about="x",
        claim_ids=[],
        source_ids=[],
        severity="critical",
        subtopic_id="sub-1",
        converted_to_benchmark_case=True,
    )
    decision = StoppingCalculus(CONFIG, TOPIC).evaluate(
        state, receipt_rate=(4, 4), evaluation_reserve_intact=True
    )
    assert "critical_contradictions" not in decision.unmet


def test_one_family_is_not_coverage_however_many_sources():
    calculus = StoppingCalculus(CONFIG, TOPIC)
    coverage, rows = calculus.facet_coverage(state_with(6, families=1))
    assert coverage == 0.0
    assert any("families" in reason for reason in rows[0]["unmet"])


def test_an_unassigned_critical_gap_blocks_sufficiency():
    decision = StoppingCalculus(CONFIG, TOPIC).evaluate(
        state_with(4),
        receipt_rate=(4, 4),
        evaluation_reserve_intact=True,
        unassigned_critical_gaps=["f1"],
    )
    assert "no_unassigned_critical_gap" in decision.unmet


# -- pairing ---------------------------------------------------------------


def _row(question, arm, value, repetition=1, metric="fact_f1", status="ok"):
    return {
        "metric": metric,
        "arm_id": arm,
        "question_id": question,
        "value": value,
        "status": status,
        "repetition": repetition,
        "answer_id": f"a-{arm}-{question}-{repetition}",
    }


def test_a_question_missing_in_one_arm_is_excluded_from_both():
    from pheasant_lab.evaluation.pairing import pair_arms

    rows = [_row("q1", "P0", 0.5), _row("q1", "P1", 0.7), _row("q2", "P0", 0.4)]
    questions = {"q1": {"cohorts": ["anchor"]}, "q2": {"cohorts": ["anchor"]}}
    pairing = pair_arms(
        rows, metric="fact_f1", baseline_arm="P0", treatment_arm="P1", questions=questions
    )
    assert pairing.n == 1
    assert pairing.excluded["missing_in_treatment"] == ["q2"]
    assert pairing.coverage == pytest.approx(0.5)


def test_repetitions_collapse_to_one_value_per_question():
    from pheasant_lab.evaluation.pairing import collapse_repetitions

    rows = [_row("q1", "P0", 0.2, 1), _row("q1", "P0", 0.8, 2), _row("q1", "P0", 0.6, 3)]
    collapsed = collapse_repetitions(rows, "fact_f1", "P0")
    assert collapsed["q1"][0] == pytest.approx(0.6)


def test_a_withheld_value_never_enters_a_pair():
    from pheasant_lab.evaluation.pairing import pair_arms

    rows = [
        _row("q1", "P0", None, status="insufficient_evidence"),
        _row("q1", "P1", 0.7),
    ]
    pairing = pair_arms(
        rows, metric="fact_f1", baseline_arm="P0", treatment_arm="P1", questions={"q1": {}}
    )
    assert pairing.n == 0
