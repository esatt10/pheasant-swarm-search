"""Collection metrics.

These describe the *corpus*, not the retrieval over it. Two are deliberately
direction-free and say so: source-family diversity is reported as raw counts
because "more families is better" is false for a field with three groups in
it, and marginal claim yield falling means this search direction is exhausted,
which is compatible with having missed the field entirely.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from ..orchestration.state import CollectionState
from ..settings import Topic
from .metric import MetricResult, MetricScope

VERSION = "collection-1"


def compute(
    state: CollectionState,
    topic: Topic,
    *,
    run_id: str,
    facet_rows: Sequence[Mapping[str, Any]],
    receipt_rate: tuple[int, int],
    index_rate: tuple[int, int],
    digest_verified: tuple[int, int],
) -> list[MetricResult]:
    scope = MetricScope(run_id=run_id, topic_id=topic.id)
    out: list[MetricResult] = []

    met_weight = sum(float(row["weight"]) for row in facet_rows if row["meets_minimum"])
    total_weight = sum(float(row["weight"]) for row in facet_rows)
    out.append(
        MetricResult(
            metric="facet_coverage",
            version=VERSION,
            classification="primary",
            scope=scope,
            formula="sum(w_f * I(facet meets its minimum)) / sum(w_f)",
            numerator=met_weight,
            denominator=total_weight,
            operand_ids=[str(row["facet_id"]) for row in facet_rows],
            claim_supported="the weighted share of this topic's facets that meet their configured "
            "source, family and peer-review minimums",
            claim_not_supported="that the covered facets are covered *well*, or that the facet list "
            "is the right decomposition of the field",
            limitation="the facet list and its weights are configured per topic; this number is "
            "relative to that decomposition and not comparable across topics",
        ).validate()
    )

    duplicates, acquired = state.duplicate_rate()
    out.append(
        MetricResult(
            metric="duplicate_rate",
            version=VERSION,
            classification="primary",
            scope=scope,
            formula="duplicate or equivalent items / acquired items",
            numerator=float(duplicates),
            denominator=float(acquired),
            claim_supported="how much of the collection effort landed on material already held",
            claim_not_supported="that the non-duplicates are distinct in substance",
            limitation="equivalence is DOI or normalised title; a preprint and its published "
            "version that share neither are counted as two",
        ).validate()
    )

    retained = state.retained_sources()
    families = Counter(record.family for record in retained)
    derived = sum(1 for record in retained if record.candidate.family_key is None)
    out.append(
        MetricResult(
            metric="source_family_diversity",
            version=VERSION,
            classification="descriptive",
            scope=scope,
            formula="distinct independent source families / retained sources",
            numerator=float(len(families)),
            denominator=float(len(retained)),
            operand_ids=sorted(families),
            detail={
                "largest_family": max(families.values()) if families else 0,
                "families": dict(families),
            },
            claim_supported="how concentrated the retained corpus is across research groups",
            claim_not_supported="that a higher number is better - a field with three groups has three",
            limitation=f"{derived}/{len(retained)} family keys were derived from first-author "
            "surname rather than affiliation, which is a weaker key",
        ).validate()
    )

    rounds = state.rounds
    if rounds:
        latest = rounds[-1]
        out.append(
            MetricResult(
                metric="marginal_claim_yield",
                version=VERSION,
                classification="descriptive",
                scope=scope,
                formula="new unique eligible claims in the round / eligible claims before the round",
                numerator=float(latest.new_eligible_claims),
                denominator=float(max(1, latest.eligible_claims_before)),
                detail={"by_round": [round_.as_dict() for round_ in rounds]},
                claim_supported="whether this search direction was still producing new evidence",
                claim_not_supported="that the collection is good - a declining yield is exhaustion "
                "of a direction, which is compatible with having missed the field",
                limitation="the denominator is what was already held, so an early round with a "
                "small base produces a large ratio by construction",
            ).validate()
        )

    critical = [c for c in state.contradictions.values() if c.critical]
    closed = [c for c in critical if c.resolved or c.converted_to_benchmark_case]
    out.append(
        MetricResult(
            metric="critical_contradiction_closure",
            version=VERSION,
            classification="primary",
            scope=scope,
            formula="resolved or benchmarked critical contradictions / critical contradictions",
            numerator=float(len(closed)),
            denominator=float(len(critical)),
            operand_ids=[c.contradiction_id for c in critical],
            claim_supported="whether disagreements the collection found were carried forward "
            "rather than dropped",
            claim_not_supported="that the disagreements were resolved correctly - converting one "
            "into a benchmark case closes it here without settling it",
            limitation="severity is assigned during extraction; a disagreement nobody marked "
            "critical is not in this denominator",
        ).validate()
    )

    complete, retained_count = state.provenance_completeness()
    out.append(
        MetricResult(
            metric="provenance_completeness",
            version=VERSION,
            classification="primary",
            scope=scope,
            formula="sources with required provenance / retained sources",
            numerator=float(complete),
            denominator=float(retained_count),
            threshold=1.0,
            claim_supported="whether every retained source can be resolved back through the "
            "lineage chain",
            claim_not_supported="that the provenance recorded is accurate",
            limitation="target is 1.0; anything less means some number in this run cannot be "
            "traced to its source",
        ).validate()
    )

    receipts, submitted = receipt_rate
    out.append(
        MetricResult(
            metric="ingest_receipt_rate",
            version=VERSION,
            classification="primary",
            scope=scope,
            formula="verified receipts / submitted eligible sources",
            numerator=float(receipts),
            denominator=float(submitted),
            claim_supported="whether the region acknowledged what this run submitted",
            claim_not_supported="that the acknowledged content is retrievable - acceptance is not "
            "searchability",
            limitation="a transport success with no receipt counts against this, deliberately",
        ).validate()
    )

    indexed, receipt_count = index_rate
    out.append(
        MetricResult(
            metric="indexed_content_verification",
            version=VERSION,
            classification="primary",
            scope=scope,
            formula="retrievable digest-verified items / receipts sampled",
            numerator=float(indexed),
            denominator=float(receipt_count),
            detail={"digest_verified": list(digest_verified)},
            claim_supported="how much of what was accepted crossed the index barrier",
            claim_not_supported="that the indexed content is what was submitted, unless the "
            "digest check also passed",
            limitation=f"digest comparison was possible for {digest_verified[1]} of {receipt_count} "
            "receipts; where the region reported no digest, the answer is unknown rather than no",
        ).validate()
    )
    return out
