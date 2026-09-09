"""The paired diagnostics."""

from __future__ import annotations

import pytest

from pheasant_lab.evaluation.statistics import (
    benjamini_hochberg,
    bootstrap_interval,
    effect_size,
    mcnemar,
    wilcoxon_signed_rank,
)


def test_bootstrap_is_reproducible_for_a_seed():
    deltas = [0.1, -0.05, 0.2, 0.0, 0.15, -0.1, 0.05, 0.3]
    first = bootstrap_interval(deltas, resamples=200, seed=7)
    second = bootstrap_interval(deltas, resamples=200, seed=7)
    assert first is not None and (first.lower, first.upper) == (second.lower, second.upper)
    assert bootstrap_interval(deltas, resamples=200, seed=8) != first


def test_bootstrap_needs_more_than_one_observation():
    assert bootstrap_interval([0.1], resamples=100, seed=1) is None


def test_an_interval_that_straddles_zero_says_so():
    interval = bootstrap_interval([0.5, -0.5, 0.4, -0.4, 0.1, -0.1], resamples=400, seed=3)
    assert interval is not None and not interval.excludes_zero


def test_mcnemar_matches_the_exact_binomial():
    # 5 discordant pairs all in one direction: 2 * (1/2)^5 = 0.0625
    baseline = [True] * 5 + [True, False]
    treatment = [False] * 5 + [True, False]
    result = mcnemar(baseline, treatment)
    assert result.p_value == pytest.approx(0.0625)
    assert result.detail["discordant"] == 5


def test_mcnemar_reports_no_discordant_pairs_rather_than_a_p_value():
    result = mcnemar([True, False], [True, False])
    assert result.p_value is None
    assert "no discordant" in result.detail["note"]


def test_wilcoxon_refuses_too_few_nonzero_differences():
    result = wilcoxon_signed_rank([0.1, 0.0, -0.2])
    assert result.p_value is None
    assert "too few" in result.detail["note"]


def test_wilcoxon_detects_a_consistent_shift():
    result = wilcoxon_signed_rank([0.3, 0.4, 0.2, 0.5, 0.35, 0.45, 0.25, 0.6])
    assert result.p_value is not None and result.p_value < 0.05


def test_effect_size_publishes_raw_counts_beside_the_standardised_number():
    effect = effect_size([1.0, -1.0, 0.0, 1.0])
    assert (effect["wins"], effect["ties"], effect["losses"]) == (2, 1, 1)
    assert effect["rank_biserial"] == pytest.approx(0.25)


def test_benjamini_hochberg_never_promotes_a_missing_p_value():
    survives = benjamini_hochberg([0.001, None, 0.9], fdr=0.10)
    assert survives == [True, False, False]


def test_benjamini_hochberg_uses_the_step_up_threshold():
    assert benjamini_hochberg([0.01, 0.02, 0.03], fdr=0.05) == [True, True, True]
    assert benjamini_hochberg([0.04, 0.05, 0.9], fdr=0.05) == [False, False, False]
