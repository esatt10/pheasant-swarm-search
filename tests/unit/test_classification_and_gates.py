"""Turning deltas into statuses, and gates that refuse to be empty."""

from __future__ import annotations

import pytest

from pheasant_lab.evaluation.classification import Status, classify, non_inferior
from pheasant_lab.evaluation.gates import GateOutcome, GateSet, Verdict, evaluate_gates
from pheasant_lab.evaluation.pairing import PairedSample, Pairing
from pheasant_lab.settings import ClassificationSection, GateSpec, StatisticsSection


def pairing(deltas, *, cohorts=None, metric="fact_f1", types=None):
    samples = [
        PairedSample(
            question_id=f"q-{i}",
            cohort=(cohorts or ["anchor"] * len(deltas))[i],
            baseline=0.5,
            treatment=0.5 + delta,
            question_type=(types or ["atomic_fact"] * len(deltas))[i],
        )
        for i, delta in enumerate(deltas)
    ]
    result = Pairing(metric=metric, baseline_arm="P0", treatment_arm="P1", samples=samples)
    result.eligible_questions = len(deltas)
    return result


CLASSIFICATION = ClassificationSection(
    practical_threshold={"default": 0.05},
    minimum_paired_questions=6,
    minimum_pairing_coverage=0.5,
    protected_subgroups=["question_type", "cohort"],
)
STATS = StatisticsSection(bootstrap_resamples=200, minimum_n_for_tests=6, bootstrap_seed=1)


def test_too_few_pairs_is_insufficient_evidence_not_unchanged():
    result = classify(pairing([0.2, 0.2]), classification=CLASSIFICATION, statistics_config=STATS)
    assert result.status is Status.insufficient_evidence
    assert "below the configured minimum" in result.reasons[0]


def test_a_clear_gain_is_improved_and_names_its_threshold():
    result = classify(pairing([0.3] * 8), classification=CLASSIFICATION, statistics_config=STATS)
    assert result.status is Status.improved
    assert result.threshold == 0.05
    assert "+0.05" in result.reasons[0]


def test_a_gain_whose_interval_includes_zero_is_unchanged():
    result = classify(
        pairing([0.6, -0.5, 0.6, -0.5, 0.6, -0.4, 0.5, -0.3]),
        classification=CLASSIFICATION,
        statistics_config=STATS,
    )
    assert result.status in {Status.unchanged, Status.mixed}
    assert any(
        "includes zero" in reason or "protected subgroup" in reason for reason in result.reasons
    )


def test_a_loss_is_regressed():
    result = classify(pairing([-0.3] * 8), classification=CLASSIFICATION, statistics_config=STATS)
    assert result.status is Status.regressed


def test_an_aggregate_gain_with_a_regressed_subgroup_is_mixed():
    deltas = [0.4] * 6 + [-0.4] * 3
    types = ["atomic_fact"] * 6 + ["contradiction"] * 3
    result = classify(
        pairing(deltas, types=types, cohorts=["anchor"] * 9),
        classification=CLASSIFICATION,
        statistics_config=STATS,
    )
    assert result.status is Status.mixed
    assert any("regressed" in reason for reason in result.reasons)


def test_lower_is_better_metrics_flip_the_sign():
    lower = ClassificationSection(
        practical_threshold={"default": 0.05},
        lower_is_better=["negative_exposure_at_k"],
        minimum_paired_questions=6,
        minimum_pairing_coverage=0.5,
    )
    result = classify(
        pairing([-0.3] * 8, metric="negative_exposure_at_k"),
        classification=lower,
        statistics_config=STATS,
    )
    assert result.status is Status.improved


def test_a_gate_failure_overrides_a_numerical_gain():
    result = classify(
        pairing([0.4] * 8),
        classification=CLASSIFICATION,
        statistics_config=STATS,
        gate_failures=["acl_leak"],
    )
    assert result.status is Status.regressed


def test_uncontrolled_differences_make_the_comparison_not_comparable():
    result = classify(
        pairing([0.4] * 8),
        classification=CLASSIFICATION,
        statistics_config=STATS,
        comparability_findings=["models.researcher differs"],
    )
    assert result.status is Status.not_comparable


def test_non_inferiority_does_not_require_a_numeric_win():
    result = classify(pairing([-0.01] * 12), classification=CLASSIFICATION, statistics_config=STATS)
    answer, reason = non_inferior(result, margin=0.05)
    assert answer is True
    assert "no worse than the margin" in reason


def test_non_inferiority_without_an_interval_is_unknown_not_false():
    result = classify(pairing([0.1, 0.1]), classification=CLASSIFICATION, statistics_config=STATS)
    answer, reason = non_inferior(result, margin=0.05)
    assert answer is None
    assert "no confidence interval" in reason


# -- gates -----------------------------------------------------------------


def test_an_empty_gate_set_cannot_be_constructed():
    with pytest.raises(ValueError, match="cannot be empty"):
        GateSet("core", [])


def test_a_skipped_gate_makes_the_set_incomplete_not_passing():
    gate_set = GateSet(
        "core",
        [
            GateOutcome("a", True, True, 1, 1, "ok"),
            GateOutcome("b", False, None, None, 0, "", skip_reason="no observation"),
        ],
    )
    assert gate_set.verdict() is Verdict.incomplete
    assert gate_set.heading() == "INCOMPLETE (1 of 2 gates evaluated)"


def test_a_failure_beats_an_incomplete():
    gate_set = GateSet(
        "core",
        [
            GateOutcome("a", True, False, 5, 1, "over"),
            GateOutcome("b", False, None, None, 0, "", skip_reason="skipped"),
        ],
    )
    assert gate_set.verdict() is Verdict.failed


def test_evaluate_gates_skips_a_gate_with_no_observation_and_keeps_its_reason():
    specs = {"acl_leak": GateSpec(enabled=True, max_violations=0)}
    result = evaluate_gates(specs, {"acl_leak": (None, "no ACL is configured")})
    outcome = result.sets["isolation"].outcomes[0]
    assert outcome.evaluated is False
    assert outcome.skip_reason == "no ACL is configured"
    assert result.verdict is Verdict.incomplete


def test_a_disabled_gate_is_skipped_rather_than_passed():
    specs = {"budget": GateSpec(enabled=False, max_overrun_usd=0.0)}
    result = evaluate_gates(specs, {"budget": (0.0, "no overrun")})
    assert result.sets["budget"].verdict() is Verdict.incomplete
