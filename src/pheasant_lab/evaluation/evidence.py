"""Typed proof.

Served, considered and included are **unknown**, weight zero. So is
``not_selected``: the reader may have found the answer at rank one, and
treating silence as a negative manufactures negatives at exactly the rate the
region serves results.

Weight is the product of four **reported** multipliers - directness,
independence, specificity, recency - never a single opaque number. Positive
and negative sums never cancel: ``P``, ``N``, ``Net`` and the conflict rate
are published separately.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .. import ids
from ..lifecycle import isonow
from ..settings import ProofPolicyFile

MULTIPLIER_AXES = ("directness", "independence", "specificity", "recency")


@dataclass
class ProofEvent:
    proof_id: str
    run_id: str
    question_id: str
    arm_id: str | None
    target_id: str
    target_type: str
    event_type: str
    polarity: str
    base_weight: float
    multipliers: dict[str, float] = field(default_factory=dict)
    weight: float = 0.0
    reported_by: str = "lab"
    recorded_at: str = field(default_factory=isonow)
    answer_id: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "proof_id": self.proof_id,
            "run_id": self.run_id,
            "question_id": self.question_id,
            "arm_id": self.arm_id,
            "answer_id": self.answer_id,
            "target_id": self.target_id,
            "target_type": self.target_type,
            "event_type": self.event_type,
            "polarity": self.polarity,
            "base_weight": self.base_weight,
            "multipliers": dict(self.multipliers),
            "weight": self.weight,
            "reported_by": self.reported_by,
            "recorded_at": self.recorded_at,
            "detail": self.detail,
        }


@dataclass
class ProofSummary:
    positive_weight: float = 0.0
    negative_weight: float = 0.0
    events: int = 0
    unknown_events: int = 0
    conflicted_targets: int = 0
    judged_targets: int = 0

    @property
    def net_weight(self) -> float:
        return self.positive_weight - self.negative_weight

    @property
    def conflict_rate(self) -> float | None:
        if not self.judged_targets:
            return None
        return self.conflicted_targets / self.judged_targets

    def as_dict(self) -> dict[str, Any]:
        return {
            "positive_weight": round(self.positive_weight, 6),
            "negative_weight": round(self.negative_weight, 6),
            "net_weight": round(self.net_weight, 6),
            "events": self.events,
            "unknown_events": self.unknown_events,
            "judged_targets": self.judged_targets,
            "conflicted_targets": self.conflicted_targets,
            "conflict_rate": self.conflict_rate,
        }


class ProofLedger:
    """Every proof event this run produced, and what they weigh."""

    def __init__(self, policy: ProofPolicyFile, *, run_id: str) -> None:
        self.policy = policy
        self.run_id = run_id
        self.events: list[ProofEvent] = []

    def record(
        self,
        *,
        question_id: str,
        target_id: str,
        event_type: str,
        arm_id: str | None = None,
        answer_id: str | None = None,
        target_type: str = "artifact",
        directness: str = "direct_text",
        independence: str = "same_arm",
        specificity: str = "document_level",
        recency: str = "current",
        reported_by: str = "lab",
        detail: dict[str, Any] | None = None,
    ) -> ProofEvent:
        spec = self.policy.proof.event_types.get(event_type)
        if spec is None:
            raise KeyError(
                f"proof event type '{event_type}' is not in the proof policy. A vocabulary with "
                "two homes has one home nobody reads."
            )
        multipliers = {
            "directness": self._multiplier("directness", directness),
            "independence": self._multiplier("independence", independence),
            "specificity": self._multiplier("specificity", specificity),
            "recency": self._multiplier("recency", recency),
        }
        weight = spec.base_weight
        for value in multipliers.values():
            weight *= value
        event = ProofEvent(
            proof_id=ids.proof_id(question_id, arm_id or "", target_id, event_type),
            run_id=self.run_id,
            question_id=question_id,
            arm_id=arm_id,
            answer_id=answer_id,
            target_id=target_id,
            target_type=target_type,
            event_type=event_type,
            polarity=spec.polarity,
            base_weight=spec.base_weight,
            multipliers=multipliers,
            weight=weight if spec.polarity != "unknown" else 0.0,
            reported_by=reported_by,
            detail=dict(detail or {}),
        )
        self.events.append(event)
        return event

    def _multiplier(self, axis: str, key: str) -> float:
        table = self.policy.proof.multipliers.get(axis, {})
        if key not in table:
            # An unknown level is worth nothing rather than everything: a
            # multiplier defaulting to 1.0 silently promotes evidence nobody
            # classified.
            return 0.0
        return float(table[key])

    # -- reading -----------------------------------------------------------
    def for_question(self, question_id: str, arm_id: str | None = None) -> list[ProofEvent]:
        return [
            event
            for event in self.events
            if event.question_id == question_id and (arm_id is None or event.arm_id == arm_id)
        ]

    def summarise(self, events: Sequence[ProofEvent] | None = None) -> ProofSummary:
        rows = list(self.events if events is None else events)
        summary = ProofSummary(events=len(rows))
        by_target: dict[str, set[str]] = defaultdict(set)
        for event in rows:
            if event.polarity == "unknown":
                summary.unknown_events += 1
                continue
            by_target[event.target_id].add(event.polarity)
            if event.polarity == "positive":
                summary.positive_weight += event.weight
            else:
                summary.negative_weight += event.weight
        summary.judged_targets = len(by_target)
        summary.conflicted_targets = sum(
            1 for polarities in by_target.values() if len(polarities) > 1
        )
        return summary

    def coverage(self, question_ids: Iterable[str]) -> tuple[int, int]:
        """Questions carrying at least the minimum proof, over all questions."""

        minimum = self.policy.proof.minimum_evidence.per_question_proof_events
        wanted = list(question_ids)
        judged = {event.question_id for event in self.events if event.polarity != "unknown"}
        covered = sum(
            1
            for question_id in wanted
            if len([e for e in self.for_question(question_id) if e.polarity != "unknown"])
            >= minimum
            and question_id in judged
        )
        return (covered, len(wanted))

    def as_records(self) -> list[dict[str, Any]]:
        return [event.as_dict() for event in self.events]
