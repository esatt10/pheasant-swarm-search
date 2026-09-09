"""Cost reservation, commitment and reconciliation.

Before every model or paid provider call:

1. reserve the estimated **maximum** cost from input size, the output cap, the
   tool allowance and the pricing profile;
2. refuse the call when committed plus reserved would exceed the hard budget;
3. reconcile against actual usage once the response is back;
4. persist estimated, reserved, actual and currency.

Reserving the worst case rather than the expected case is the whole point. A
guard that reserves an average lets a run overshoot on the calls that were
above it, which is exactly the population of calls a budget exists to stop.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from .settings import BudgetSection, LabConfig, PricingFile

BUCKETS = ("planning", "collection", "benchmark", "evaluation", "reserve")


class BudgetExceeded(RuntimeError):
    """A call that would take the run past its hard budget. Never retryable."""

    def __init__(self, message: str, *, bucket: str, requested: float, available: float) -> None:
        super().__init__(message)
        self.bucket = bucket
        self.requested = requested
        self.available = available


class PricingUnavailable(RuntimeError):
    """A model with no price, where the config says that is fatal."""


@dataclass(frozen=True)
class CostEstimate:
    model: str
    input_tokens: int
    max_output_tokens: int
    input_usd: float
    output_usd: float

    @property
    def total_usd(self) -> float:
        return self.input_usd + self.output_usd


@dataclass
class CostEvent:
    """One priced call, recorded whether or not it succeeded."""

    event_kind: str
    bucket: str
    role: str
    model: str
    estimated_usd: float
    reserved_usd: float
    actual_usd: float | None
    input_tokens: int
    output_tokens: int
    currency: str
    reconciled: bool
    detail: dict[str, Any] = field(default_factory=dict)

    def as_payload(self) -> dict[str, Any]:
        return {
            "event_kind": self.event_kind,
            "bucket": self.bucket,
            "role": self.role,
            "model": self.model,
            "estimated_usd": round(self.estimated_usd, 8),
            "reserved_usd": round(self.reserved_usd, 8),
            "actual_usd": None if self.actual_usd is None else round(self.actual_usd, 8),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "currency": self.currency,
            "reconciled": self.reconciled,
            "detail": self.detail,
        }


def estimate_tokens(text: str) -> int:
    """A deliberately conservative token estimate.

    Four characters per token underestimates for code and identifier-dense
    text, so the estimate is rounded up and floored at one. It is used only to
    *reserve*; the reconciliation uses what the provider reported.
    """

    if not text:
        return 1
    return max(1, math.ceil(len(text) / 3.5))


class CostLedger:
    """The run's one budget authority.

    Thread-safe because collection runs bounded concurrent branches, and two
    branches reserving against one remaining balance is precisely the race a
    budget guard exists to arbitrate.
    """

    def __init__(
        self,
        *,
        total_budget_usd: float,
        allocation: dict[str, float],
        pricing: PricingFile,
        fail_when_price_missing: bool = True,
        budget_section: BudgetSection | None = None,
        currency: str = "USD",
    ) -> None:
        self.total_budget_usd = float(total_budget_usd)
        self.allocation = dict(allocation)
        self.pricing = pricing
        self.fail_when_price_missing = fail_when_price_missing
        self.section = budget_section or BudgetSection()
        self.currency = currency
        self._lock = threading.RLock()
        self._committed: dict[str, float] = dict.fromkeys(BUCKETS, 0.0)
        self._reserved: dict[str, float] = dict.fromkeys(BUCKETS, 0.0)
        self.events: list[CostEvent] = []

    # -- construction -----------------------------------------------------
    @classmethod
    def from_config(cls, config: LabConfig, *, budget_override: float | None = None) -> CostLedger:
        return cls(
            total_budget_usd=budget_override
            if budget_override is not None
            else config.experiment.cost_budget_usd,
            allocation=config.budget.allocation.as_dict(),
            pricing=config.pricing,
            fail_when_price_missing=config.pricing_ref.fail_when_model_price_missing,
            budget_section=config.budget,
            currency=config.pricing.currency,
        )

    # -- pricing ----------------------------------------------------------
    def price(self, model: str) -> tuple[float, float]:
        """Return ``(input, output)`` price per token for ``model``."""

        entry = self.pricing.models.get(model)
        if entry is None:
            if self.fail_when_price_missing:
                raise PricingUnavailable(
                    f"no price for model '{model}' in the pricing file. A model with no price "
                    "is not free; it is a budget guard that does not exist."
                )
            return (0.0, 0.0)
        divisor = 1_000_000.0 if self.pricing.unit == "per_million_tokens" else 1_000.0
        return (entry.input / divisor, entry.output / divisor)

    def estimate(
        self, *, model: str, prompt: str, max_output_tokens: int, tool_calls: int = 0
    ) -> CostEstimate:
        input_price, output_price = self.price(model)
        input_tokens = estimate_tokens(prompt)
        output_tokens = (
            max_output_tokens
            if self.section.reserve_output_at_max_tokens
            else max_output_tokens // 2
        )
        if tool_calls:
            # Each tool round trip feeds its result back in and produces more
            # output; both halves are reserved.
            extra = tool_calls * self.section.assumed_tool_call_output_tokens
            input_tokens += extra
            output_tokens += extra
        return CostEstimate(
            model=model,
            input_tokens=input_tokens,
            max_output_tokens=output_tokens,
            input_usd=input_tokens * input_price,
            output_usd=output_tokens * output_price,
        )

    # -- accounting -------------------------------------------------------
    @property
    def committed_usd(self) -> float:
        with self._lock:
            return sum(self._committed.values())

    @property
    def reserved_usd(self) -> float:
        with self._lock:
            return sum(self._reserved.values())

    @property
    def remaining_usd(self) -> float:
        with self._lock:
            return (
                self.total_budget_usd - sum(self._committed.values()) - sum(self._reserved.values())
            )

    def bucket_budget(self, bucket: str) -> float:
        return self.total_budget_usd * self.allocation.get(bucket, 0.0)

    def bucket_remaining(self, bucket: str) -> float:
        with self._lock:
            return self.bucket_budget(bucket) - self._committed[bucket] - self._reserved[bucket]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total_budget_usd": self.total_budget_usd,
                "committed_usd": round(sum(self._committed.values()), 8),
                "reserved_usd": round(sum(self._reserved.values()), 8),
                "remaining_usd": round(self.remaining_usd, 8),
                "currency": self.currency,
                "by_bucket": {
                    bucket: {
                        "budget_usd": round(self.bucket_budget(bucket), 8),
                        "committed_usd": round(self._committed[bucket], 8),
                        "reserved_usd": round(self._reserved[bucket], 8),
                    }
                    for bucket in BUCKETS
                },
                "calls": len(self.events),
            }

    # -- the guard --------------------------------------------------------
    def reserve(self, bucket: str, amount: float) -> None:
        if bucket not in self._committed:
            raise KeyError(f"unknown budget bucket '{bucket}'")
        with self._lock:
            projected = sum(self._committed.values()) + sum(self._reserved.values()) + amount
            if projected > self.total_budget_usd + 1e-12:
                raise BudgetExceeded(
                    f"reserving {amount:.4f} {self.currency} for '{bucket}' would take the run to "
                    f"{projected:.4f} against a hard budget of {self.total_budget_usd:.4f}",
                    bucket=bucket,
                    requested=amount,
                    available=self.total_budget_usd
                    - sum(self._committed.values())
                    - sum(self._reserved.values()),
                )
            self._reserved[bucket] += amount

    def release(self, bucket: str, amount: float) -> None:
        with self._lock:
            self._reserved[bucket] = max(0.0, self._reserved[bucket] - amount)

    def commit(self, bucket: str, amount: float) -> None:
        with self._lock:
            self._committed[bucket] += amount

    @contextmanager
    def spend(
        self,
        *,
        bucket: str,
        role: str,
        model: str,
        prompt: str,
        max_output_tokens: int,
        tool_calls: int = 0,
        event_kind: str = "model_call",
    ) -> Iterator[CostEvent]:
        """Reserve, run, reconcile.

        The caller sets ``event.input_tokens`` / ``event.output_tokens`` from
        what the provider reported; on exit the reservation is released and the
        actual cost committed. A call that raises still commits what it spent,
        because a failed call that consumed tokens consumed them.
        """

        estimate = self.estimate(
            model=model, prompt=prompt, max_output_tokens=max_output_tokens, tool_calls=tool_calls
        )
        self.reserve(bucket, estimate.total_usd)
        event = CostEvent(
            event_kind=event_kind,
            bucket=bucket,
            role=role,
            model=model,
            estimated_usd=estimate.total_usd,
            reserved_usd=estimate.total_usd,
            actual_usd=None,
            input_tokens=estimate.input_tokens,
            output_tokens=0,
            currency=self.currency,
            reconciled=False,
        )
        try:
            yield event
        finally:
            input_price, output_price = self.price(model)
            actual = event.input_tokens * input_price + event.output_tokens * output_price
            event.actual_usd = actual
            event.reconciled = True
            self.release(bucket, estimate.total_usd)
            self.commit(bucket, actual)
            with self._lock:
                self.events.append(event)

    def record_external(
        self, *, bucket: str, role: str, label: str, amount_usd: float
    ) -> CostEvent:
        """A paid non-model call (a provider with a per-request price)."""

        self.reserve(bucket, amount_usd)
        self.release(bucket, amount_usd)
        self.commit(bucket, amount_usd)
        event = CostEvent(
            event_kind="external_call",
            bucket=bucket,
            role=role,
            model=label,
            estimated_usd=amount_usd,
            reserved_usd=amount_usd,
            actual_usd=amount_usd,
            input_tokens=0,
            output_tokens=0,
            currency=self.currency,
            reconciled=True,
        )
        with self._lock:
            self.events.append(event)
        return event

    # -- collection / evaluation split ------------------------------------
    def evaluation_reserve_intact(self, reserve_fraction: float) -> bool:
        """Is enough left to run every arm and the reports?"""

        return self.remaining_usd >= self.total_budget_usd * reserve_fraction - 1e-12

    def usage_by(self, key: str) -> dict[str, float]:
        """Committed cost grouped by ``role``, ``model`` or ``bucket``."""

        totals: dict[str, float] = {}
        with self._lock:
            for event in self.events:
                name = getattr(event, key)
                totals[name] = totals.get(name, 0.0) + (event.actual_usd or 0.0)
        return dict(sorted(totals.items()))
