"""Operational metrics: latency, reliability, cost.

Cost per *proven successful answer* is the number worth reading and the one
that needs the most care: its denominator is answers that passed a
deterministic check, not answers that were produced. An arm that produces
fluent unsupported answers cheaply looks good on cost-per-answer and bad here,
which is the correct direction.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .metric import MetricResult, MetricScope

VERSION = "operational-1"


def percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def latency_by_stage(events: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float | None]]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for event in events:
        payload = event.get("payload") or {}
        duration = payload.get("duration_ms")
        if duration is None:
            continue
        stage = str(payload.get("tool") or event.get("event_type") or "unknown")
        buckets[stage].append(float(duration))
    return {
        stage: {
            "count": float(len(values)),
            "p50_ms": percentile(values, 0.50),
            "p95_ms": percentile(values, 0.95),
            "max_ms": max(values),
        }
        for stage, values in sorted(buckets.items())
    }


def mcp_reliability(calls: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    statuses = Counter(str(call.get("status")) for call in calls)
    attempts = sum(int(call.get("attempt") or 1) for call in calls)
    total = sum(statuses.values())
    return {
        "calls": total,
        "attempts": attempts,
        "succeeded": statuses.get("succeeded", 0),
        "partial": statuses.get("partial", 0),
        "failed": statuses.get("failed", 0),
        "retry_rate": (attempts - total) / total if total else None,
        "success_rate": statuses.get("succeeded", 0) / total if total else None,
        # Partial is its own row. Folding it into success is how a truncated
        # result set becomes a corpus gap in the report.
        "partial_rate": statuses.get("partial", 0) / total if total else None,
    }


def compute(
    *,
    run_id: str,
    answers: Sequence[Mapping[str, Any]],
    cost_events: Sequence[Mapping[str, Any]],
    proven_answers: int,
    verified_sources: int,
    questions: int,
    budget: Mapping[str, Any],
) -> list[MetricResult]:
    scope = MetricScope(run_id=run_id)
    out: list[MetricResult] = []
    total_cost = float(budget.get("committed_usd") or 0.0)

    out.append(
        MetricResult(
            metric="cost_per_verified_source",
            version=VERSION,
            classification="descriptive",
            scope=scope,
            formula="total committed cost / sources with a verified ingest receipt",
            numerator=total_cost,
            denominator=float(verified_sources),
            unit="usd",
            claim_supported="what one persisted, receipted source cost this run",
            claim_not_supported="what it would cost at another scale - the fixed planning cost is "
            "in this numerator",
            limitation="the whole run's cost is in the numerator, including evaluation",
        ).validate()
    )

    out.append(
        MetricResult(
            metric="cost_per_benchmark_question",
            version=VERSION,
            classification="descriptive",
            scope=scope,
            formula="total committed cost / benchmark questions",
            numerator=total_cost,
            denominator=float(questions),
            unit="usd",
            claim_supported="the run's cost amortised over its question set",
            claim_not_supported="the marginal cost of one more question",
            limitation="collection cost dominates and does not scale with question count",
        ).validate()
    )

    out.append(
        MetricResult(
            metric="cost_per_proven_successful_answer",
            version=VERSION,
            classification="primary",
            scope=scope,
            formula="total committed cost / answers that passed a deterministic check",
            numerator=total_cost,
            denominator=float(proven_answers),
            unit="usd",
            claim_supported="what one answer this benchmark can vouch for cost",
            claim_not_supported="what an unverified answer cost, or what a human would have paid",
            limitation="the denominator counts answers with a deterministic pass, so a run whose "
            "benchmark could judge less looks more expensive here - which is the honest direction",
        ).validate()
    )

    latencies = [float(a.get("latency_ms") or 0.0) for a in answers if a.get("latency_ms")]
    out.append(
        MetricResult(
            metric="answer_latency_p95_ms",
            version=VERSION,
            classification="descriptive",
            scope=scope,
            formula="95th percentile of per-answer wall time",
            numerator=percentile(latencies, 0.95),
            denominator=1.0,
            value=percentile(latencies, 0.95),
            unit="ms",
            detail={
                "p50_ms": percentile(latencies, 0.50),
                "max_ms": max(latencies) if latencies else None,
            },
            claim_supported="how long an answer took on this laptop, end to end",
            claim_not_supported="anything about the region's own latency under load",
            limitation="includes the model call and the local process; it is not a server metric",
        ).validate()
    )

    by_role: dict[str, float] = defaultdict(float)
    by_model: dict[str, float] = defaultdict(float)
    tokens_in = tokens_out = 0
    for event in cost_events:
        by_role[str(event.get("role"))] += float(event.get("actual_usd") or 0.0)
        by_model[str(event.get("model"))] += float(event.get("actual_usd") or 0.0)
        tokens_in += int(event.get("input_tokens") or 0)
        tokens_out += int(event.get("output_tokens") or 0)

    out.append(
        MetricResult(
            metric="token_usage",
            version=VERSION,
            classification="descriptive",
            scope=scope,
            formula="sum of reported input and output tokens",
            numerator=float(tokens_in + tokens_out),
            denominator=1.0,
            value=float(tokens_in + tokens_out),
            unit="tokens",
            detail={
                "input_tokens": tokens_in,
                "output_tokens": tokens_out,
                "by_role_usd": dict(sorted(by_role.items())),
                "by_model_usd": dict(sorted(by_model.items())),
            },
            claim_supported="what this run consumed, as the providers reported it",
            claim_not_supported="what a rerun would consume",
            limitation="the deterministic offline provider reports estimated tokens, not billed "
            "ones, and prices them at zero",
        ).validate()
    )
    return out
