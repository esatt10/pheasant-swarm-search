"""The coverage auditor.

Reads the trace store and nothing else. It never searches, never reads a
paper, and never sees an answer - which is what makes its findings usable as
an input to the stop decision rather than a restatement of it.

Three things it refuses to do:

* soften a missing facet because the round budget is nearly spent (that is the
  orchestrator's decision, and it needs unsoftened input to make it);
* treat "many sources" as coverage (six papers from one lab are one family);
* treat declining yield as quality (it means this direction is exhausted,
  which is compatible with having missed the field).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from ..budget import CostLedger
from ..models import ModelProvider, ModelRequest
from ..promptlib import load as load_prompt
from ..settings import LabConfig, Topic
from .state import CollectionState
from .stopping import StoppingCalculus


@dataclass
class QualityGap:
    facet_id: str
    kind: str
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {"facet_id": self.facet_id, "kind": self.kind, "detail": self.detail}


@dataclass
class CoverageAudit:
    facet_coverage: list[dict[str, Any]] = field(default_factory=list)
    coverage_fraction: float = 0.0
    duplicate_rate: float = 0.0
    duplicates: int = 0
    acquired: int = 0
    quality_gaps: list[QualityGap] = field(default_factory=list)
    unresolved_critical: list[dict[str, Any]] = field(default_factory=list)
    marginal_claim_yield: list[float] = field(default_factory=list)
    unassigned_critical_gaps: list[str] = field(default_factory=list)
    narrative: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "coverage_fraction": self.coverage_fraction,
            "facet_coverage": list(self.facet_coverage),
            "duplicate_rate": self.duplicate_rate,
            "duplicates": self.duplicates,
            "acquired": self.acquired,
            "quality_gaps": [g.as_dict() for g in self.quality_gaps],
            "unresolved_critical_contradictions": list(self.unresolved_critical),
            "marginal_claim_yield": list(self.marginal_claim_yield),
            "unassigned_critical_gaps": list(self.unassigned_critical_gaps),
            "narrative": self.narrative,
        }


class CoverageAuditor:
    def __init__(
        self,
        config: LabConfig,
        topic: Topic,
        *,
        model: ModelProvider | None = None,
        ledger: CostLedger | None = None,
        tracer: Any = None,
    ) -> None:
        self.config = config
        self.topic = topic
        self.model = model
        self.ledger = ledger
        self.tracer = tracer
        self.calculus = StoppingCalculus(
            config.stopping,
            topic,
            authoritative_types=config.collection.authoritative_source_types,
        )

    def audit(
        self, state: CollectionState, *, assigned_facets: set[str] | None = None
    ) -> CoverageAudit:
        coverage, rows = self.calculus.facet_coverage(state)
        duplicate_rate, duplicates, acquired = self.calculus.duplicate_rate(state)
        assigned = assigned_facets or set()

        gaps: list[QualityGap] = []
        unassigned: list[str] = []
        for row in rows:
            facet_id = str(row["facet_id"])
            sources = state.sources_for_facet(facet_id)
            if not row["meets_minimum"] and facet_id not in assigned:
                unassigned.append(facet_id)
            if not sources:
                gaps.append(
                    QualityGap(facet_id, "no_sources", "no retained source addresses this facet")
                )
                continue
            families = {s.family for s in sources}
            if len(families) == 1:
                gaps.append(
                    QualityGap(
                        facet_id,
                        "single_family",
                        f"all {len(sources)} sources resolve to one family ({next(iter(families))})",
                    )
                )
            if all(s.candidate.source_type == "preprint" for s in sources):
                gaps.append(
                    QualityGap(facet_id, "preprint_only", "no peer-reviewed source on this facet")
                )
            elif not row.get("authoritative"):
                # The web analogue of preprint-only: every source is somebody's
                # account of the thing, and none is the thing speaking for itself.
                gaps.append(
                    QualityGap(
                        facet_id,
                        "no_authoritative_source",
                        "no source of an authoritative type "
                        f"({', '.join(sorted(self.calculus.authoritative_types))}) on this facet",
                    )
                )
            decades = {
                (s.candidate.published_at or "")[:3] for s in sources if s.candidate.published_at
            }
            if len(decades) == 1 and len(sources) >= 3:
                gaps.append(
                    QualityGap(
                        facet_id,
                        "single_decade",
                        f"every source is from the {next(iter(decades))}0s",
                    )
                )

        unresolved = [
            {
                "contradiction_id": contradiction.contradiction_id,
                "about": contradiction.about,
                "severity": contradiction.severity,
                # A critical disagreement the collection cannot resolve is not
                # a blocker if it can become a benchmark uncertainty case -
                # which is the honest outcome for a genuinely open question.
                "convertible_to_benchmark_case": len(contradiction.source_ids) >= 2,
            }
            for contradiction in state.outstanding_critical_contradictions()
        ]

        audit = CoverageAudit(
            facet_coverage=rows,
            coverage_fraction=coverage,
            duplicate_rate=duplicate_rate,
            duplicates=duplicates,
            acquired=acquired,
            quality_gaps=gaps,
            unresolved_critical=unresolved,
            marginal_claim_yield=self.calculus.marginal_yield(state),
            unassigned_critical_gaps=sorted(unassigned),
        )
        self._narrate(audit)
        if self.tracer is not None:
            self.tracer.emit("collection.audit", payload=audit.as_dict(), topic_id=self.topic.id)
        return audit

    def _narrate(self, audit: CoverageAudit) -> None:
        """Optionally let a model put the numbers into words.

        The numbers are computed before this runs and are not passed back
        through it: a narrator that could change a coverage figure would be a
        second, unauditable coverage calculation.
        """

        if self.model is None or self.ledger is None:
            return
        spec = self.config.role("auditor")
        request = ModelRequest(
            role="auditor",
            schema="audit",
            system=load_prompt("auditor"),
            user="Summarise the audit below in at most five sentences. Do not change any number.",
            context={"computed": audit.as_dict()},
            max_output_tokens=spec.max_output_tokens,
            temperature=spec.temperature,
        )
        with self.ledger.spend(
            bucket="planning",
            role="auditor",
            model=spec.model,
            prompt=request.prompt_text,
            max_output_tokens=spec.max_output_tokens,
        ) as cost:
            response = self.model.complete(request)
            cost.input_tokens = response.input_tokens
            cost.output_tokens = response.output_tokens
        if self.tracer is not None:
            self.tracer.emit("cost.model_call", payload=cost.as_payload())
        audit.narrative = str(response.data.get("narrative") or "")

    # -- collection metrics the report reuses ------------------------------
    def source_family_diversity(self, state: CollectionState) -> dict[str, Any]:
        retained = state.retained_sources()
        families = Counter(s.family for s in retained)
        return {
            "retained_sources": len(retained),
            "distinct_families": len(families),
            "largest_family": max(families.values()) if families else 0,
            # Reported as raw counts, because "more families is better" is not
            # true in general: a field with three groups in it has three.
            "families": dict(families.most_common()),
        }
