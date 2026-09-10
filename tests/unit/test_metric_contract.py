"""The metric result contract."""

from __future__ import annotations

import pytest

from pheasant_lab.evaluation.metric import MetricResult, MetricScope, insufficient, not_available


def build(**kwargs) -> MetricResult:
    defaults = {
        "metric": "m",
        "version": "1",
        "classification": "primary",
        "scope": MetricScope(run_id="run-1"),
        "formula": "n / d",
        "limitation": "a limitation",
    }
    return MetricResult(**{**defaults, **kwargs})


def test_a_metric_without_a_denominator_is_withheld_not_zeroed():
    result = build(numerator=0.0, denominator=0.0).validate()
    assert result.status == "insufficient_evidence"
    assert result.value is None, "0.0 would read as a measured failure"


def test_a_metric_must_state_what_it_does_not_support():
    with pytest.raises(ValueError, match="published with no limitation"):
        build(numerator=1.0, denominator=2.0, limitation="").validate()


def test_substitution_is_rendered_from_the_operands():
    result = build(numerator=3.0, denominator=4.0).validate()
    assert result.substituted == "3 / 4 = 0.75"
    assert result.value == pytest.approx(0.75)


def test_the_id_is_a_function_of_scope_and_operands():
    first = build(numerator=1.0, denominator=2.0, operand_ids=["a", "b"]).validate()
    second = build(numerator=1.0, denominator=2.0, operand_ids=["b", "a"]).validate()
    assert first.metric_result_id == second.metric_result_id
    third = build(numerator=1.0, denominator=3.0, operand_ids=["a", "b"]).validate()
    assert third.metric_result_id != first.metric_result_id


def test_insufficient_and_not_available_are_different_answers():
    scope = MetricScope(run_id="run-1")
    low = insufficient("m", "1", scope, formula="f", reason="too few")
    broken = not_available("m", "1", scope, formula="f", reason="the diagnostic did not run")
    assert low.status == "insufficient_evidence"
    assert broken.status == "not_available"
    assert low.value is None and broken.value is None
