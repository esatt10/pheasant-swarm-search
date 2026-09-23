"""The stopping calculus.

Collection may be declared **sufficient** only when every hard condition
passes. Budget or clock exhaustion produces ``stopped_budget_incomplete`` or
``stopped_time_incomplete`` - never ``sufficient``. That distinction is the
whole point of this module: a run that ran out of money and reported a
complete collection would put every number downstream on a foundation nobody
can check.

Declining marginal yield is evidence that *this search direction* is
exhausted. It is not evidence that the collection is good, and this module
never treats it as such: it is one condition among seven, and the coverage
conditions can fail while it passes.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from ..profiles import PEER_REVIEWED_TYPES
from ..settings import StoppingSection, Topic
from .state import CollectionState

Outcome = Literal[
    "continue",
    "sufficient",
    "stopped_budget_incomplete",
    "stopped_time_incomplete",
    "stopped_round_limit_incomplete",
]

CONDITION_IDS = (
    "facet_minimums",
    "provenance_complete",
    "ingest_receipt_rate",
    "critical_contradictions",
    "marginal_yield_saturated",
    "evaluation_reserve_intact",
    "no_unassigned_critical_gap",
    "duplicate_rate",
)


@dataclass
class Condition:
    id: str
    met: bool
    detail: str
    numerator: float | None = None
    denominator: float | None = None
    threshold: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "met": self.met,
            "detail": self.detail,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "threshold": self.threshold,
        }


@dataclass
class StopDecision:
    outcome: Outcome
    conditions: list[Condition] = field(default_factory=list)
    reason: str = ""
    facet_coverage: float | None = None
    marginal_yield: list[float] = field(default_factory=list)

    @property
    def unmet(self) -> list[str]:
        return [c.id for c in self.conditions if not c.met]

    @property
    def sufficient(self) -> bool:
        return self.outcome == "sufficient"

    def as_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "reason": self.reason,
            "facet_coverage": self.facet_coverage,
            "marginal_yield": list(self.marginal_yield),
            "unmet_conditions": self.unmet,
            "conditions": [c.as_dict() for c in self.conditions],
        }


class StoppingCalculus:
    """Evaluates the hard conditions. Holds no opinion of its own."""

    def __init__(
        self,
        config: StoppingSection,
        topic: Topic,
        *,
        authoritative_types: Iterable[str] = PEER_REVIEWED_TYPES,
    ) -> None:
        self.config = config
        self.topic = topic
        # Which source types satisfy `minimum_review_or_primary_sources`. The
        # collection profile decides: peer-reviewed for scholarly, primary for
        # web, either for balanced.
        self.authoritative_types = frozenset(authoritative_types)

    # -- the three published formulas --------------------------------------
    def facet_coverage(self, state: CollectionState) -> tuple[float, list[dict[str, Any]]]:
        """Weighted facets meeting their minimum, over total facet weight."""

        rows: list[dict[str, Any]] = []
        met_weight = 0.0
        total_weight = 0.0
        for facet in self.topic.facets:
            sources = state.sources_for_facet(facet.id)
            families = state.families_for_facet(facet.id)
            peer_reviewed = state.peer_reviewed_for_facet(facet.id)
            authoritative = state.authoritative_for_facet(facet.id, self.authoritative_types)
            unmet: list[str] = []
            if len(sources) < self.config.minimum_sources_per_subtopic:
                unmet.append(f"sources {len(sources)}<{self.config.minimum_sources_per_subtopic}")
            if len(families) < self.config.minimum_independent_source_families:
                unmet.append(
                    f"families {len(families)}<{self.config.minimum_independent_source_families}"
                )
            if authoritative < self.config.minimum_review_or_primary_sources:
                unmet.append(
                    f"authoritative {authoritative}<{self.config.minimum_review_or_primary_sources}"
                )
            total_weight += facet.weight
            if not unmet:
                met_weight += facet.weight
            rows.append(
                {
                    "facet_id": facet.id,
                    "label": facet.label,
                    "weight": facet.weight,
                    "sources": len(sources),
                    "families": len(families),
                    "peer_reviewed": peer_reviewed,
                    "authoritative": authoritative,
                    "claims": len(state.claims_for_facet(facet.id)),
                    "meets_minimum": not unmet,
                    "unmet": unmet,
                }
            )
        coverage = met_weight / total_weight if total_weight else 0.0
        return coverage, rows

    def duplicate_rate(self, state: CollectionState) -> tuple[float, int, int]:
        duplicates, acquired = state.duplicate_rate()
        return (duplicates / acquired if acquired else 0.0, duplicates, acquired)

    def marginal_yield(self, state: CollectionState) -> list[float]:
        return [round(r.marginal_claim_yield, 6) for r in state.rounds]

    # -- the decision ------------------------------------------------------
    def evaluate(
        self,
        state: CollectionState,
        *,
        receipt_rate: tuple[int, int],
        evaluation_reserve_intact: bool,
        unassigned_critical_gaps: Sequence[str] = (),
        budget_exhausted: bool = False,
        time_exhausted: bool = False,
        round_limit_reached: bool = False,
    ) -> StopDecision:
        coverage, _rows = self.facet_coverage(state)
        duplicate_rate, duplicates, acquired = self.duplicate_rate(state)
        yields = self.marginal_yield(state)
        window = self.config.marginal_unique_claim_window
        recent = yields[-window:] if window else []
        saturated_rounds = sum(
            1 for value in recent if value < self.config.marginal_unique_claim_threshold
        )

        complete, retained = state.provenance_completeness()
        receipts, submitted = receipt_rate
        rate = receipts / submitted if submitted else 0.0

        conditions = [
            Condition(
                "facet_minimums",
                coverage >= 1.0,
                f"{coverage:.2%} of facet weight meets its minimum",
                numerator=coverage,
                denominator=1.0,
                threshold=1.0,
            ),
            Condition(
                "provenance_complete",
                retained > 0 and complete == retained,
                f"{complete}/{retained} retained sources carry complete provenance",
                numerator=complete,
                denominator=retained,
                threshold=1.0,
            ),
            Condition(
                "ingest_receipt_rate",
                submitted > 0 and rate >= self.config.minimum_ingest_receipt_rate,
                f"{receipts}/{submitted} submitted sources carry a receipt",
                numerator=receipts,
                denominator=submitted,
                threshold=self.config.minimum_ingest_receipt_rate,
            ),
            Condition(
                "critical_contradictions",
                len(state.outstanding_critical_contradictions())
                <= self.config.maximum_unresolved_critical_contradictions,
                f"{len(state.outstanding_critical_contradictions())} unresolved critical contradictions",
                numerator=len(state.outstanding_critical_contradictions()),
                threshold=self.config.maximum_unresolved_critical_contradictions,
            ),
            Condition(
                "marginal_yield_saturated",
                saturated_rounds >= self.config.consecutive_saturated_rounds,
                f"{saturated_rounds}/{self.config.consecutive_saturated_rounds} recent rounds below "
                f"{self.config.marginal_unique_claim_threshold} marginal yield",
                numerator=saturated_rounds,
                threshold=self.config.consecutive_saturated_rounds,
            ),
            Condition(
                "evaluation_reserve_intact",
                evaluation_reserve_intact,
                "enough budget remains to run every arm and the reports",
            ),
            Condition(
                "no_unassigned_critical_gap",
                not unassigned_critical_gaps,
                f"unassigned critical gaps: {sorted(unassigned_critical_gaps) or 'none'}",
                numerator=len(unassigned_critical_gaps),
                threshold=0,
            ),
            Condition(
                "duplicate_rate",
                duplicate_rate <= self.config.maximum_duplicate_rate,
                f"{duplicates}/{acquired} acquired items are duplicates ({duplicate_rate:.2%})",
                numerator=duplicates,
                denominator=acquired,
                threshold=self.config.maximum_duplicate_rate,
            ),
        ]

        # Exhaustion is decided before sufficiency, and never renamed as it.
        if budget_exhausted or not evaluation_reserve_intact:
            outcome: Outcome = "stopped_budget_incomplete"
            reason = "the collection allowance is spent or would eat the evaluation reserve"
        elif time_exhausted:
            outcome = "stopped_time_incomplete"
            reason = "the runtime budget is spent"
        elif all(condition.met for condition in conditions):
            outcome = "sufficient"
            reason = "every hard condition in `stopping` passes"
        elif round_limit_reached:
            outcome = "stopped_round_limit_incomplete"
            reason = "the configured round limit was reached with conditions outstanding"
        else:
            outcome = "continue"
            unmet = [c.id for c in conditions if not c.met]
            reason = f"outstanding: {', '.join(unmet)}"

        if outcome != "sufficient" and all(c.met for c in conditions) and outcome != "continue":
            reason += " (every condition passed, but the run stopped for the reason above)"

        return StopDecision(
            outcome=outcome,
            conditions=conditions,
            reason=reason,
            facet_coverage=coverage,
            marginal_yield=yields,
        )
