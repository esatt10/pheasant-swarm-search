"""The metric result contract.

Every metric this lab publishes carries: its name and version, its
classification, its scope, the exact formula, the substituted calculation, its
numerator and denominator, its operands, its exclusions, its status, the claim
it supports **and the claim it does not**.

Two rules are structural rather than remembered:

* a metric that cannot carry its denominator reports ``insufficient_evidence``
  with ``value: None``. It never reports ``0.0`` - a point that could not be
  measured is not one that measured badly;
* every result carries one ``limitation``. A number whose limits are written
  down somewhere else arrives at the reader after the mistake.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .. import ids
from ..hashing import digest

MetricStatus = Literal["ok", "insufficient_evidence", "not_available", "not_comparable"]
Classification = Literal["primary", "diagnostic", "gate", "descriptive"]


@dataclass
class MetricScope:
    run_id: str | None = None
    arm_id: str | None = None
    question_id: str | None = None
    cohort: str | None = None
    snapshot_id: str | None = None
    topic_id: str | None = None
    subgroup: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class MetricResult:
    metric: str
    version: str
    classification: Classification
    scope: MetricScope
    formula: str
    value: float | None = None
    numerator: float | None = None
    denominator: float | None = None
    unit: str = "ratio"
    substituted: str = ""
    excluded: int = 0
    exclusion_reasons: dict[str, int] = field(default_factory=dict)
    operand_ids: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    baseline: float | None = None
    treatment: float | None = None
    absolute_delta: float | None = None
    relative_delta: float | None = None
    interval: tuple[float, float] | None = None
    status: MetricStatus = "ok"
    threshold: float | None = None
    claim_supported: str = ""
    claim_not_supported: str = ""
    limitation: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.substituted:
            self.substituted = self._substitute()

    def _substitute(self) -> str:
        if self.numerator is None or self.denominator is None:
            return ""
        if not self.denominator:
            return f"{self.numerator:g} / 0 = undefined"
        return (
            f"{self.numerator:g} / {self.denominator:g} = {self.numerator / self.denominator:.6g}"
        )

    @property
    def metric_result_id(self) -> str:
        return ids.metric_result_id(
            self.metric,
            self.version,
            self.scope.as_dict(),
            digest({"o": sorted(self.operand_ids), "n": self.numerator, "d": self.denominator}),
        )

    def validate(self, *, minimum_denominator: float = 1.0) -> MetricResult:
        """Withhold a result that cannot be defended.

        Called before a result is published, not after: a number that reached
        a report and was corrected there has already been read.
        """

        if self.status != "ok":
            self.value = None
            return self
        if self.denominator is None or self.denominator < minimum_denominator:
            self.status = "insufficient_evidence"
            self.value = None
            if not self.limitation:
                self.limitation = (
                    f"denominator {self.denominator} is below the minimum {minimum_denominator}; "
                    "this is not a score of zero, it is an absence of evidence"
                )
            return self
        if self.value is None and self.numerator is not None and self.denominator:
            self.value = self.numerator / self.denominator
        if not self.limitation:
            raise ValueError(
                f"metric '{self.metric}' was published with no limitation. Every number here "
                "states what it does not support; one that does not is a number waiting to be "
                "over-read."
            )
        return self

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric_result_id": self.metric_result_id,
            "metric": self.metric,
            "version": self.version,
            "classification": self.classification,
            **self.scope.as_dict(),
            "scope": self.scope.as_dict(),
            "formula": self.formula,
            "substituted": self.substituted,
            "value": self.value,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "unit": self.unit,
            "excluded": self.excluded,
            "exclusion_reasons": dict(self.exclusion_reasons),
            "operand_ids": list(self.operand_ids),
            "evidence_refs": list(self.evidence_refs),
            "baseline": self.baseline,
            "treatment": self.treatment,
            "absolute_delta": self.absolute_delta,
            "relative_delta": self.relative_delta,
            "interval": list(self.interval) if self.interval else None,
            "status": self.status,
            "threshold": self.threshold,
            "claim_supported": self.claim_supported,
            "claim_not_supported": self.claim_not_supported,
            "limitation": self.limitation,
            "detail": self.detail,
        }


def insufficient(
    metric: str,
    version: str,
    scope: MetricScope,
    *,
    formula: str,
    reason: str,
    classification: Classification = "primary",
) -> MetricResult:
    """The result to publish when the evidence is not there."""

    return MetricResult(
        metric=metric,
        version=version,
        classification=classification,
        scope=scope,
        formula=formula,
        value=None,
        status="insufficient_evidence",
        limitation=reason,
        claim_supported="nothing: the evidence minimum was not met",
        claim_not_supported="any statement about this metric's level or direction",
    )


def not_available(
    metric: str,
    version: str,
    scope: MetricScope,
    *,
    formula: str,
    reason: str,
    classification: Classification = "diagnostic",
) -> MetricResult:
    """An optional diagnostic that could not run.

    Distinct from ``insufficient_evidence`` and from a bad score: a diagnostic
    that failed must not turn a core metric into ``0.0``.
    """

    return MetricResult(
        metric=metric,
        version=version,
        classification=classification,
        scope=scope,
        formula=formula,
        value=None,
        status="not_available",
        limitation=reason,
        claim_supported="nothing: this diagnostic did not run",
        claim_not_supported="any statement derived from this diagnostic",
    )
