"""Questions, expected facts, and the deterministic matchers that score them.

A required fact carries **its own matcher**. That is the design decision the
whole answer-metric plane rests on: a fact nobody can match deterministically
is not eligible as an operand, and is reported as an *exclusion* rather than
as a miss. A miss says the arm failed; an exclusion says the benchmark could
not judge. Collapsing the two makes every arm look worse in proportion to how
hard the benchmark was to write.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from ..textkit import (
    ABSTENTION_MARKERS,
    DISAGREEMENT_MARKERS,
    normalise,
)
from ..textkit import (
    numbers as extract_numbers,
)
from ..textkit import (
    tokens as extract_tokens,
)

QUESTION_TYPES = (
    "atomic_fact",
    "synthesis",
    "mechanism",
    "contradiction",
    "temporal",
    "source_id",
    "abstention",
)

# The composition keys in the config map onto the question types above.
COMPOSITION_ALIASES = {
    "atomic_fact": "atomic_fact",
    "multi_source_synthesis": "synthesis",
    "mechanism_or_causality": "mechanism",
    "contradiction_or_uncertainty": "contradiction",
    "temporal_or_versioned": "temporal",
    "source_identification": "source_id",
    "abstention": "abstention",
}

COHORTS = ("anchor", "learned", "temporal_holdout", "control", "invariant")

DIFFICULTIES = ("direct", "multi_hop", "adversarial")

MatcherKind = Literal["all_of", "any_of", "numeric", "regex", "abstain", "source", "disagreement"]


@dataclass
class FactMatcher:
    """How one expected fact is recognised in an answer.

    ``groups`` is a conjunction of disjunctions: every group must be
    satisfied, and a group is satisfied by any of its alternatives. That shape
    is what lets a fact accept the two spellings a field uses for one thing
    without accepting an answer that used neither.
    """

    kind: MatcherKind = "all_of"
    groups: list[list[str]] = field(default_factory=list)
    value: float | None = None
    unit: str | None = None
    relative_tolerance: float | None = None
    pattern: str | None = None
    identifiers: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "groups": [list(group) for group in self.groups],
            "value": self.value,
            "unit": self.unit,
            "relative_tolerance": self.relative_tolerance,
            "pattern": self.pattern,
            "identifiers": list(self.identifiers),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FactMatcher:
        return cls(
            kind=payload.get("kind", "all_of"),
            groups=[list(group) for group in payload.get("groups", [])],
            value=payload.get("value"),
            unit=payload.get("unit"),
            relative_tolerance=payload.get("relative_tolerance"),
            pattern=payload.get("pattern"),
            identifiers=[str(i) for i in payload.get("identifiers", [])],
        )

    @property
    def deterministic(self) -> bool:
        """Can this matcher decide, on its own, without a judge?"""

        if self.kind in {"abstain", "disagreement"}:
            return True
        if self.kind == "numeric":
            return self.value is not None
        if self.kind == "regex":
            return bool(self.pattern)
        if self.kind == "source":
            return bool(self.identifiers)
        return bool(self.groups)

    def matches(
        self,
        text: str,
        *,
        abstained: bool = False,
        citations: list[str] | None = None,
        tolerance: float = 0.02,
    ) -> bool:
        folded = normalise(text)
        if self.kind == "abstain":
            return abstained or any(marker in folded for marker in ABSTENTION_MARKERS)
        if abstained:
            # An abstention answers an abstention case and nothing else. It is
            # not a wrong answer to a factual question - it is no answer - and
            # the recall denominator is what records that.
            return False
        if self.kind == "regex" and self.pattern:
            return re.search(self.pattern, folded) is not None
        if self.kind == "numeric" and self.value is not None:
            allowed = self.relative_tolerance if self.relative_tolerance is not None else tolerance
            span = abs(self.value) * allowed if self.value else allowed
            return any(abs(found - self.value) <= span for found in extract_numbers(text))
        if self.kind == "source":
            haystack = folded + " " + normalise(" ".join(citations or []))
            return any(normalise(identifier) in haystack for identifier in self.identifiers)
        if self.kind == "disagreement":
            if not (set(extract_tokens(text)) & DISAGREEMENT_MARKERS):
                return False
            return all(_group_matches(group, folded) for group in self.groups)
        if self.kind == "any_of":
            return any(_group_matches(group, folded) for group in self.groups)
        return all(_group_matches(group, folded) for group in self.groups)


def _group_matches(group: list[str], folded: str) -> bool:
    return any(normalise(alternative) in folded for alternative in group if alternative)


@dataclass
class ExpectedFact:
    fact_id: str
    text: str
    matcher: FactMatcher
    claim_ids: list[str] = field(default_factory=list)
    source_ids: list[str] = field(default_factory=list)
    facet_id: str | None = None
    required: bool = True
    confidence_basis: str = "direct_text"

    @property
    def eligible(self) -> bool:
        """Only deterministic matchers over direct or structured evidence.

        A ``researcher_inference`` claim may have guided collection; it can
        never be the operand an arm is scored against.
        """

        return self.matcher.deterministic and self.confidence_basis in {
            "direct_text",
            "structured_metadata",
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "text": self.text,
            "matcher": self.matcher.as_dict(),
            "claim_ids": list(self.claim_ids),
            "source_ids": list(self.source_ids),
            "facet_id": self.facet_id,
            "required": self.required,
            "confidence_basis": self.confidence_basis,
            "eligible": self.eligible,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ExpectedFact:
        return cls(
            fact_id=str(payload["fact_id"]),
            text=str(payload.get("text", "")),
            matcher=FactMatcher.from_dict(payload.get("matcher", {})),
            claim_ids=[str(c) for c in payload.get("claim_ids", [])],
            source_ids=[str(s) for s in payload.get("source_ids", [])],
            facet_id=payload.get("facet_id"),
            required=bool(payload.get("required", True)),
            confidence_basis=str(payload.get("confidence_basis", "direct_text")),
        )


@dataclass
class ExpectedEvidence:
    question_id: str
    acceptable_source_ids: list[str] = field(default_factory=list)
    acceptable_artifact_ids: list[str] = field(default_factory=list)
    known_negative_source_ids: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "acceptable_source_ids": list(self.acceptable_source_ids),
            "acceptable_artifact_ids": list(self.acceptable_artifact_ids),
            "known_negative_source_ids": list(self.known_negative_source_ids),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ExpectedEvidence:
        return cls(
            question_id=str(payload["question_id"]),
            acceptable_source_ids=[str(s) for s in payload.get("acceptable_source_ids", [])],
            acceptable_artifact_ids=[str(a) for a in payload.get("acceptable_artifact_ids", [])],
            known_negative_source_ids=[
                str(s) for s in payload.get("known_negative_source_ids", [])
            ],
        )


@dataclass
class Question:
    question_id: str
    topic_id: str
    text: str
    type: str
    difficulty: str = "direct"
    required_fact_ids: list[str] = field(default_factory=list)
    acceptable_evidence_source_ids: list[str] = field(default_factory=list)
    known_negative_source_ids: list[str] = field(default_factory=list)
    as_of: str | None = None
    cohorts: list[str] = field(default_factory=list)
    created_from_claim_ids: list[str] = field(default_factory=list)
    facet_ids: list[str] = field(default_factory=list)
    subtopic_id: str | None = None

    @property
    def is_abstention(self) -> bool:
        return self.type == "abstention"

    def as_dict(self, *, run_id: str | None = None) -> dict[str, Any]:
        payload = {
            "question_id": self.question_id,
            "topic_id": self.topic_id,
            "text": self.text,
            "type": self.type,
            "difficulty": self.difficulty,
            "required_fact_ids": list(self.required_fact_ids),
            "acceptable_evidence_source_ids": list(self.acceptable_evidence_source_ids),
            "known_negative_source_ids": list(self.known_negative_source_ids),
            "as_of": self.as_of,
            "cohorts": list(self.cohorts),
            "created_from_claim_ids": list(self.created_from_claim_ids),
            "facet_ids": list(self.facet_ids),
            "subtopic_id": self.subtopic_id,
        }
        if run_id:
            payload["run_id"] = run_id
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Question:
        return cls(
            question_id=str(payload["question_id"]),
            topic_id=str(payload.get("topic_id", "")),
            text=str(payload.get("text", "")),
            type=str(payload.get("type", "atomic_fact")),
            difficulty=str(payload.get("difficulty", "direct")),
            required_fact_ids=[str(f) for f in payload.get("required_fact_ids", [])],
            acceptable_evidence_source_ids=[
                str(s) for s in payload.get("acceptable_evidence_source_ids", [])
            ],
            known_negative_source_ids=[
                str(s) for s in payload.get("known_negative_source_ids", [])
            ],
            as_of=payload.get("as_of"),
            cohorts=[str(c) for c in payload.get("cohorts", [])],
            created_from_claim_ids=[str(c) for c in payload.get("created_from_claim_ids", [])],
            facet_ids=[str(f) for f in payload.get("facet_ids", [])],
            subtopic_id=payload.get("subtopic_id"),
        )
