"""The budget guard."""

from __future__ import annotations

import threading

import pytest

from pheasant_lab.budget import BudgetExceeded, CostLedger, PricingUnavailable, estimate_tokens
from pheasant_lab.settings import ModelPrice, PricingFile


def ledger(total: float = 1.0, *, price: float = 1.0) -> CostLedger:
    return CostLedger(
        total_budget_usd=total,
        allocation={
            "planning": 0.2,
            "collection": 0.2,
            "benchmark": 0.2,
            "evaluation": 0.2,
            "reserve": 0.2,
        },
        pricing=PricingFile(models={"m": ModelPrice(input=price, output=price)}),
    )


def test_estimate_reserves_the_output_cap_not_the_average():
    guard = ledger(total=100.0)
    estimate = guard.estimate(model="m", prompt="x" * 350, max_output_tokens=1000)
    assert estimate.max_output_tokens == 1000
    assert estimate.total_usd == pytest.approx((estimate.input_tokens + 1000) / 1e6)


def test_a_call_that_would_exceed_the_budget_is_refused_before_it_runs():
    guard = ledger(total=0.000001)
    with (
        pytest.raises(BudgetExceeded) as caught,
        guard.spend(
            bucket="collection", role="r", model="m", prompt="x" * 4000, max_output_tokens=100000
        ),
    ):
        pytest.fail("the body must not run")
    assert caught.value.bucket == "collection"


def test_actual_usage_is_committed_and_the_reservation_released():
    guard = ledger(total=100.0)
    with guard.spend(
        bucket="collection", role="r", model="m", prompt="x" * 350, max_output_tokens=1000
    ) as event:
        event.input_tokens = 10
        event.output_tokens = 20
    assert guard.reserved_usd == 0.0
    assert guard.committed_usd == pytest.approx(30 / 1e6)
    assert guard.events[0].reconciled is True


def test_a_failing_call_still_commits_what_it_spent():
    guard = ledger(total=100.0)
    with (
        pytest.raises(RuntimeError),
        guard.spend(
            bucket="collection", role="r", model="m", prompt="p", max_output_tokens=10
        ) as event,
    ):
        event.input_tokens = 5
        event.output_tokens = 5
        raise RuntimeError("provider failed after billing")
    assert guard.committed_usd == pytest.approx(10 / 1e6)


def test_a_model_with_no_price_fails_rather_than_costing_nothing():
    guard = CostLedger(
        total_budget_usd=1.0,
        allocation={
            "planning": 1.0,
            "collection": 0,
            "benchmark": 0,
            "evaluation": 0,
            "reserve": 0,
        },
        pricing=PricingFile(models={}),
    )
    with pytest.raises(PricingUnavailable, match="not free"):
        guard.price("unlisted")


def test_concurrent_reservations_cannot_both_win_the_last_dollar():
    guard = ledger(total=0.000002)
    refusals: list[BudgetExceeded] = []
    successes: list[int] = []

    def attempt() -> None:
        try:
            with guard.spend(
                bucket="collection", role="r", model="m", prompt="x", max_output_tokens=1000
            ) as event:
                event.input_tokens = 1
                event.output_tokens = 1000
            successes.append(1)
        except BudgetExceeded as exc:
            refusals.append(exc)

    threads = [threading.Thread(target=attempt) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert refusals, "every thread was admitted; the guard is not arbitrating"
    assert guard.committed_usd <= guard.total_budget_usd + 1e-9


def test_evaluation_reserve_is_reported_before_it_is_spent():
    guard = ledger(total=1.0)
    assert guard.evaluation_reserve_intact(0.4) is True
    guard.commit("collection", 0.8)
    assert guard.evaluation_reserve_intact(0.4) is False


def test_token_estimate_never_returns_zero():
    assert estimate_tokens("") == 1
    assert estimate_tokens("a") == 1
