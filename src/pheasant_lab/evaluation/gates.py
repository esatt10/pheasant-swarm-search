"""Gates.

Gates are not metrics. They are evaluated **before** aggregation, so a good
score cannot offset a failed one.

Two structural rules, both learned expensively elsewhere and enforced here in
the constructor rather than remembered at each call site:

* ``all([])`` is ``True``, so a :class:`GateSet` refuses to be built empty.
  "No gates" is the *absence* of a gate set, and absence has no ``passed`` to
  misread.
* a verdict is **tri-state**. ``PASS`` requires that every gate in the set was
  *evaluated* and passed. A set with three of four gates skipped reports
  ``INCOMPLETE``, because an unchecked box and a failed one are equally
  disqualifying for a result somebody will publish.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from ..settings import GateSpec


class Verdict(StrEnum):
    passed = "PASS"
    failed = "FAIL"
    incomplete = "INCOMPLETE"

    @property
    def go(self) -> bool:
        return self is Verdict.passed


@dataclass
class GateOutcome:
    gate: str
    evaluated: bool
    passed: bool | None
    observed: float | int | None
    threshold: float | int | None
    detail: str
    evidence_refs: list[str] = field(default_factory=list)
    skip_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "evaluated": self.evaluated,
            "passed": self.passed,
            "observed": self.observed,
            "threshold": self.threshold,
            "detail": self.detail,
            "evidence_refs": list(self.evidence_refs),
            "skip_reason": self.skip_reason,
        }


class GateSet:
    """A named, non-empty set of gates."""

    def __init__(self, name: str, outcomes: Sequence[GateOutcome]) -> None:
        if not outcomes:
            raise ValueError(
                f"gate set '{name}' cannot be empty. `all([])` is True, so an empty set would "
                "report that it passed the gates it never had."
            )
        self.name = name
        self.outcomes = list(outcomes)

    @property
    def evaluated(self) -> list[GateOutcome]:
        return [outcome for outcome in self.outcomes if outcome.evaluated]

    @property
    def skipped(self) -> list[GateOutcome]:
        return [outcome for outcome in self.outcomes if not outcome.evaluated]

    @property
    def failures(self) -> list[GateOutcome]:
        return [outcome for outcome in self.evaluated if outcome.passed is False]

    def verdict(self) -> Verdict:
        if self.failures:
            return Verdict.failed
        if self.skipped:
            return Verdict.incomplete
        return Verdict.passed

    def heading(self) -> str:
        verdict = self.verdict()
        if verdict is Verdict.incomplete:
            return f"INCOMPLETE ({len(self.evaluated)} of {len(self.outcomes)} gates evaluated)"
        return verdict.value

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "verdict": self.verdict().value,
            "heading": self.heading(),
            "evaluated": len(self.evaluated),
            "total": len(self.outcomes),
            "failed": [outcome.gate for outcome in self.failures],
            "skipped": [outcome.gate for outcome in self.skipped],
            "gates": [outcome.as_dict() for outcome in self.outcomes],
        }


@dataclass
class GateSetResult:
    sets: dict[str, GateSet] = field(default_factory=dict)

    @property
    def verdict(self) -> Verdict:
        if not self.sets:
            return Verdict.incomplete
        verdicts = [gate_set.verdict() for gate_set in self.sets.values()]
        if Verdict.failed in verdicts:
            return Verdict.failed
        if Verdict.incomplete in verdicts:
            return Verdict.incomplete
        return Verdict.passed

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "gate_sets": {name: gate_set.as_dict() for name, gate_set in sorted(self.sets.items())},
        }


def _threshold(spec: GateSpec) -> tuple[str, float | int | None]:
    for field_name in (
        "max_violations",
        "max_rate",
        "min_rate",
        "min_accuracy",
        "max_delta",
        "max_findings",
        "max_items",
        "max_drifted_sections",
        "max_overrun_usd",
    ):
        value = getattr(spec, field_name)
        if value is not None:
            return field_name, value
    return ("none", None)


def evaluate_one(
    name: str,
    spec: GateSpec,
    observed: float | int | None,
    *,
    detail: str,
    evidence: Sequence[str] = (),
) -> GateOutcome:
    """Evaluate one gate against the observation, or record why it could not.

    ``observed is None`` is a **skip**, never a pass. A gate whose input was
    not produced has not been checked, and reporting it as clean is the exact
    shape this module exists to refuse.
    """

    kind, threshold = _threshold(spec)
    if not spec.enabled:
        return GateOutcome(
            name, False, None, observed, threshold, detail, skip_reason="gate disabled in config"
        )
    if observed is None:
        # The caller's detail *is* the reason: "no principal-scoped ACL is
        # configured" and "no observation produced" are the same fact told at
        # two different levels of usefulness.
        return GateOutcome(
            name,
            False,
            None,
            None,
            threshold,
            detail,
            list(evidence),
            skip_reason=detail or "no observation produced",
        )
    if threshold is None:
        return GateOutcome(
            name,
            False,
            None,
            observed,
            None,
            detail,
            list(evidence),
            skip_reason="no threshold configured",
        )
    passed = observed <= threshold if kind.startswith("max") else observed >= threshold
    return GateOutcome(name, True, passed, observed, threshold, detail, list(evidence))


def evaluate_gates(
    specs: Mapping[str, GateSpec],
    observations: Mapping[str, tuple[float | int | None, str]],
    *,
    grouping: Mapping[str, Sequence[str]] | None = None,
) -> GateSetResult:
    """Build the gate sets from configured specs and observed values.

    ``observations`` maps a gate name to ``(observed, detail)``. A gate in the
    config with no observation is skipped-and-counted; an observation with no
    configured gate is ignored, because a threshold nobody set is not a gate.
    """

    default_grouping = {
        "core": [
            "ingest_receipt_rate",
            "silent_loss",
            "benchmark_leakage",
            "snapshot_drift",
        ],
        "isolation": [
            "acl_leak",
            "stale_memory_leak",
            "as_of_correctness",
            "known_positive_exclusion",
        ],
        "comparison": ["control_regression", "negative_exposure_increase", "abstention"],
        "budget": ["budget"],
    }
    groups = {name: list(members) for name, members in (grouping or default_grouping).items()}

    result = GateSetResult()
    for set_name, members in groups.items():
        outcomes: list[GateOutcome] = []
        for gate_name in members:
            spec = specs.get(gate_name)
            if spec is None:
                continue
            observed, detail = observations.get(gate_name, (None, "no observation was produced"))
            outcomes.append(evaluate_one(gate_name, spec, observed, detail=detail))
        if outcomes:
            result.sets[set_name] = GateSet(set_name, outcomes)
    return result
