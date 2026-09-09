"""Collection state: the source registry, the evidence ledger, the rounds.

Two decisions here carry weight.

**Rejections and blocks are retained.** A candidate that was paywalled, or
out of scope, or a duplicate, stays in the registry with its reason code, so
the same failure is not rediscovered every round. A registry that only
remembers what it kept spends its budget re-finding what it already refused.

**Claims carry a confidence basis, and only two of the three are eligible.**
``researcher_inference`` may steer the next search and may never be an
expected-answer operand. That distinction is enforced here rather than
remembered downstream.
"""

from __future__ import annotations

import threading
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .. import ids
from ..lifecycle import isonow
from ..providers.base import SourceCandidate

ELIGIBLE_BASES = frozenset({"direct_text", "structured_metadata"})
EXTRACTION_PROFILE = "sentence-atomic-v1"


class SourceState(StrEnum):
    """The lifecycle a candidate moves through.

    Terminal-but-kept states (``rejected``, ``blocked``, ``failed_terminal``)
    are as much a part of the registry as the retained ones.
    """

    discovered = "discovered"
    validated = "validated"
    acquired = "acquired"
    extracted = "extracted"
    submitted = "submitted"
    indexed = "indexed"
    verified = "verified"
    rejected = "rejected"
    blocked = "blocked"
    failed_retryable = "failed_retryable"
    failed_terminal = "failed_terminal"

    @property
    def retained(self) -> bool:
        return self in {
            SourceState.validated,
            SourceState.acquired,
            SourceState.extracted,
            SourceState.submitted,
            SourceState.indexed,
            SourceState.verified,
        }


REASON_CODES = {
    "duplicate": "an equivalent item is already retained",
    "no_stable_identifier": "no DOI or other stable identifier, and the config requires one",
    "disallowed_type": "source type is not in collection.allowed_source_types",
    "out_of_window": "published outside the topic's date range",
    "no_text": "no abstract and no licensed full text, so nothing to extract",
    "licence": "licence does not permit retrieval of the full text",
    "paywalled": "not retrievable without a subscription",
    "provider_error": "the provider failed terminally for this item",
    "quota": "the subtopic's source budget was already spent",
}


@dataclass
class AcquisitionAttempt:
    at: str
    outcome: str
    detail: str | None = None
    bytes: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"at": self.at, "outcome": self.outcome, "detail": self.detail, "bytes": self.bytes}


@dataclass
class SourceRecord:
    """One candidate, wherever it got to."""

    source_id: str
    candidate: SourceCandidate
    subtopic_id: str
    facet_ids: list[str]
    state: SourceState = SourceState.discovered
    reason_code: str | None = None
    researcher_agent_id: str | None = None
    discovery_event_id: str | None = None
    acquisition_event_id: str | None = None
    acquisition_attempts: list[AcquisitionAttempt] = field(default_factory=list)
    artifact_id: str | None = None
    idempotency_key: str | None = None
    round: int = 0
    duplicate_of: str | None = None

    @property
    def retained(self) -> bool:
        return self.state.retained

    @property
    def family(self) -> str:
        return self.candidate.derived_family()

    def transition(self, state: SourceState, *, reason_code: str | None = None) -> None:
        self.state = state
        if reason_code is not None:
            self.reason_code = reason_code

    def provenance_complete(self) -> bool:
        """Everything a retained source must carry to be citable.

        Deliberately strict: a retained source with no locator, no identifier
        or no discovery event cannot be resolved back through the lineage
        chain, and a number resting on it cannot be defended.
        """

        candidate = self.candidate
        return bool(
            (candidate.stable_identifier or candidate.canonical_url)
            and candidate.title
            and self.discovery_event_id
            and self.researcher_agent_id
            and candidate.content_digest
        )

    def as_record(self, run_id: str) -> dict[str, Any]:
        return self.candidate.as_record(
            run_id=run_id,
            subtopic_id=self.subtopic_id,
            facet_ids=list(self.facet_ids),
            state=self.state.value,
            reason_code=self.reason_code,
            researcher_agent_id=self.researcher_agent_id,
            discovery_event_id=self.discovery_event_id,
            acquisition_event_id=self.acquisition_event_id,
            acquisition_attempts=[a.as_dict() for a in self.acquisition_attempts],
            artifact_id=self.artifact_id,
            idempotency_key=self.idempotency_key,
            round=self.round,
            duplicate_of=self.duplicate_of,
            provenance_complete=self.provenance_complete(),
        )


@dataclass
class ClaimRecord:
    claim_id: str
    claim_text: str
    claim_type: str
    source_id: str
    locator: str
    support: str
    confidence_basis: str
    researcher_agent_id: str
    subtopic_id: str
    facet_ids: list[str]
    extracted_at: str = field(default_factory=isonow)
    quotation_digest: str | None = None
    round: int = 0

    @property
    def eligible(self) -> bool:
        """May this claim be an expected-answer operand?"""

        return self.confidence_basis in ELIGIBLE_BASES

    def as_record(self, run_id: str) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "run_id": run_id,
            "claim_text": self.claim_text,
            "claim_type": self.claim_type,
            "source_id": self.source_id,
            "locator": self.locator,
            "quotation_digest": self.quotation_digest,
            "support": self.support,
            "confidence_basis": self.confidence_basis,
            "eligible": self.eligible,
            "researcher_agent_id": self.researcher_agent_id,
            "subtopic_id": self.subtopic_id,
            "facet_ids": list(self.facet_ids),
            "extracted_at": self.extracted_at,
            "round": self.round,
        }


@dataclass
class ContradictionRecord:
    contradiction_id: str
    about: str
    claim_ids: list[str]
    source_ids: list[str]
    severity: str
    subtopic_id: str
    resolved: bool = False
    resolution_note: str | None = None
    converted_to_benchmark_case: bool = False

    @property
    def critical(self) -> bool:
        return self.severity == "critical"

    @property
    def outstanding(self) -> bool:
        return self.critical and not self.resolved and not self.converted_to_benchmark_case

    def as_dict(self) -> dict[str, Any]:
        return {
            "contradiction_id": self.contradiction_id,
            "about": self.about,
            "claim_ids": list(self.claim_ids),
            "source_ids": list(self.source_ids),
            "severity": self.severity,
            "subtopic_id": self.subtopic_id,
            "resolved": self.resolved,
            "resolution_note": self.resolution_note,
            "converted_to_benchmark_case": self.converted_to_benchmark_case,
        }


@dataclass
class RoundRecord:
    number: int
    subtopics: list[str]
    new_eligible_claims: int
    eligible_claims_before: int
    acquired: int
    duplicates: int
    cost_usd: float

    @property
    def marginal_claim_yield(self) -> float:
        """New unique eligible claims over what was held before the round."""

        return self.new_eligible_claims / max(1, self.eligible_claims_before)

    def as_dict(self) -> dict[str, Any]:
        return {
            "round": self.number,
            "subtopics": list(self.subtopics),
            "new_eligible_claims": self.new_eligible_claims,
            "eligible_claims_before": self.eligible_claims_before,
            "acquired": self.acquired,
            "duplicates": self.duplicates,
            "cost_usd": self.cost_usd,
            "marginal_claim_yield": self.marginal_claim_yield,
        }


class CollectionState:
    """Everything the collection stage accumulated.

    Guarded by one re-entrant lock. Branches run concurrently and every one of
    them asks "have I seen this already?" and then writes - a check and a
    write that must not be separable, or two branches admit the same source
    and a third marks it a duplicate depending on scheduling. That is not a
    correctness problem you can see: it makes two runs of one configuration
    disagree about their own corpus by one or two sources, which is exactly
    enough to move a metric and not enough for anyone to notice why.
    """

    def __init__(self, run_id: str, topic_id: str) -> None:
        self.lock = threading.RLock()
        self.run_id = run_id
        self.topic_id = topic_id
        self.sources: dict[str, SourceRecord] = {}
        self.claims: dict[str, ClaimRecord] = {}
        self.contradictions: dict[str, ContradictionRecord] = {}
        self.rounds: list[RoundRecord] = []
        self.subtopics: dict[str, dict[str, Any]] = {}
        self.gaps: list[str] = []
        self.seen_identifiers: dict[str, str] = {}
        self.seen_titles: dict[str, str] = {}

    # -- sources -----------------------------------------------------------
    def known(self, candidate: SourceCandidate) -> str | None:
        """Has this item been seen before, in any state?

        Checked before a candidate is even validated, so a rejected item is
        not re-fetched, re-validated and re-rejected on every round.
        """

        if candidate.source_id in self.sources:
            return candidate.source_id
        if candidate.stable_identifier and candidate.stable_identifier in self.seen_identifiers:
            return self.seen_identifiers[candidate.stable_identifier]
        title = candidate.normalised_title
        if title and title in self.seen_titles:
            return self.seen_titles[title]
        return None

    def add_source(self, record: SourceRecord) -> SourceRecord:
        with self.lock:
            self.sources[record.source_id] = record
            candidate = record.candidate
            if candidate.stable_identifier:
                self.seen_identifiers.setdefault(candidate.stable_identifier, record.source_id)
            if candidate.normalised_title:
                self.seen_titles.setdefault(candidate.normalised_title, record.source_id)
        return record

    def retained_sources(self) -> list[SourceRecord]:
        return [s for s in self.sources.values() if s.retained]

    def sources_for_facet(self, facet_id: str) -> list[SourceRecord]:
        return [s for s in self.retained_sources() if facet_id in s.facet_ids]

    def sources_for_subtopic(self, subtopic_id: str) -> list[SourceRecord]:
        return [s for s in self.retained_sources() if s.subtopic_id == subtopic_id]

    # -- claims ------------------------------------------------------------
    def add_claim(self, claim: ClaimRecord) -> bool:
        """Returns True when the claim is new."""

        with self.lock:
            if claim.claim_id in self.claims:
                return False
            self.claims[claim.claim_id] = claim
            return True

    def eligible_claims(self) -> list[ClaimRecord]:
        return [c for c in self.claims.values() if c.eligible]

    def claims_for_source(self, source_id: str) -> list[ClaimRecord]:
        return [c for c in self.claims.values() if c.source_id == source_id]

    def claims_for_facet(self, facet_id: str) -> list[ClaimRecord]:
        return [c for c in self.claims.values() if facet_id in c.facet_ids]

    # -- derived numbers ---------------------------------------------------
    def duplicate_rate(self) -> tuple[int, int]:
        acquired = [s for s in self.sources.values() if s.state != SourceState.discovered]
        duplicates = [s for s in acquired if s.reason_code == "duplicate" or s.duplicate_of]
        return (len(duplicates), len(acquired))

    def families_for_facet(self, facet_id: str) -> set[str]:
        return {s.family for s in self.sources_for_facet(facet_id)}

    def peer_reviewed_for_facet(self, facet_id: str) -> int:
        return sum(1 for s in self.sources_for_facet(facet_id) if s.candidate.peer_reviewed)

    def outstanding_critical_contradictions(self) -> list[ContradictionRecord]:
        return [c for c in self.contradictions.values() if c.outstanding]

    def reason_code_counts(self) -> dict[str, int]:
        return dict(Counter(s.reason_code for s in self.sources.values() if s.reason_code))

    def provenance_completeness(self) -> tuple[int, int]:
        retained = self.retained_sources()
        return (sum(1 for s in retained if s.provenance_complete()), len(retained))

    def summary(self) -> dict[str, Any]:
        duplicates, acquired = self.duplicate_rate()
        complete, retained = self.provenance_completeness()
        return {
            "topic_id": self.topic_id,
            "sources_discovered": len(self.sources),
            "sources_retained": retained,
            "claims_total": len(self.claims),
            "claims_eligible": len(self.eligible_claims()),
            "contradictions": len(self.contradictions),
            "critical_outstanding": len(self.outstanding_critical_contradictions()),
            "duplicates": duplicates,
            "acquired": acquired,
            "provenance_complete": complete,
            "rounds": len(self.rounds),
            "reason_codes": self.reason_code_counts(),
        }


def make_claim(
    payload: dict[str, Any],
    *,
    subtopic_id: str,
    facet_ids: Iterable[str],
    researcher_agent_id: str,
    round_: int,
) -> ClaimRecord:
    text = str(payload.get("claim_text", "")).strip()
    source = str(payload.get("source_id", ""))
    locator = str(payload.get("locator") or "unspecified")
    return ClaimRecord(
        claim_id=ids.claim_id(text, source, locator, EXTRACTION_PROFILE),
        claim_text=text,
        claim_type=str(payload.get("claim_type") or "observation"),
        source_id=source,
        locator=locator,
        support=str(payload.get("support") or "supports"),
        confidence_basis=str(payload.get("confidence_basis") or "researcher_inference"),
        researcher_agent_id=researcher_agent_id,
        subtopic_id=subtopic_id,
        facet_ids=list(facet_ids),
        quotation_digest=payload.get("quotation_digest"),
        round=round_,
    )


def rehydrate(
    run_id: str,
    topic_id: str,
    source_rows: Iterable[dict[str, Any]],
    claim_rows: Iterable[dict[str, Any]],
    *,
    contradiction_rows: Iterable[dict[str, Any]] = (),
    round_rows: Iterable[dict[str, Any]] = (),
) -> CollectionState:
    """Rebuild collection state from the raw trace.

    The checkpoint is a convenience; this is the authority. Every command
    after ``collect`` reads the state back through here, so a run resumed in a
    fresh process sees exactly what the events recorded and nothing a
    checkpoint might have drifted from.
    """

    from ..providers.base import normalise_record

    state = CollectionState(run_id, topic_id)
    for row in source_rows:
        candidate = normalise_record(row, str(row.get("provider") or "unknown"))
        record = SourceRecord(
            source_id=str(row["source_id"]),
            candidate=candidate,
            subtopic_id=str(row.get("subtopic_id") or ""),
            facet_ids=[str(f) for f in row.get("facet_ids", []) or []],
            state=SourceState(str(row.get("state") or "discovered")),
            reason_code=row.get("reason_code"),
            researcher_agent_id=row.get("researcher_agent_id"),
            discovery_event_id=row.get("discovery_event_id"),
            acquisition_event_id=row.get("acquisition_event_id"),
            acquisition_attempts=[
                AcquisitionAttempt(
                    at=str(attempt.get("at", "")),
                    outcome=str(attempt.get("outcome", "")),
                    detail=attempt.get("detail"),
                    bytes=attempt.get("bytes"),
                )
                for attempt in row.get("acquisition_attempts", []) or []
            ],
            artifact_id=row.get("artifact_id"),
            idempotency_key=row.get("idempotency_key"),
            round=int(row.get("round") or 0),
            duplicate_of=row.get("duplicate_of"),
        )
        state.add_source(record)

    for row in claim_rows:
        claim = ClaimRecord(
            claim_id=str(row["claim_id"]),
            claim_text=str(row.get("claim_text", "")),
            claim_type=str(row.get("claim_type", "observation")),
            source_id=str(row.get("source_id", "")),
            locator=str(row.get("locator", "unspecified")),
            support=str(row.get("support", "supports")),
            confidence_basis=str(row.get("confidence_basis", "researcher_inference")),
            researcher_agent_id=str(row.get("researcher_agent_id", "")),
            subtopic_id=str(row.get("subtopic_id", "")),
            facet_ids=[str(f) for f in row.get("facet_ids", []) or []],
            extracted_at=str(row.get("extracted_at", "")),
            quotation_digest=row.get("quotation_digest"),
            round=int(row.get("round") or 0),
        )
        state.add_claim(claim)

    for row in contradiction_rows:
        contradiction = ContradictionRecord(
            contradiction_id=str(row["contradiction_id"]),
            about=str(row.get("about", "")),
            claim_ids=[str(c) for c in row.get("claim_ids", []) or []],
            source_ids=[str(s) for s in row.get("source_ids", []) or []],
            severity=str(row.get("severity", "material")),
            subtopic_id=str(row.get("subtopic_id", "")),
            resolved=bool(row.get("resolved")),
            resolution_note=row.get("resolution_note"),
            converted_to_benchmark_case=bool(row.get("converted_to_benchmark_case")),
        )
        state.contradictions[contradiction.contradiction_id] = contradiction

    for row in round_rows:
        state.rounds.append(
            RoundRecord(
                number=int(row.get("round") or len(state.rounds) + 1),
                subtopics=[str(s) for s in row.get("subtopics", []) or []],
                new_eligible_claims=int(row.get("new_eligible_claims") or 0),
                eligible_claims_before=int(row.get("eligible_claims_before") or 0),
                acquired=int(row.get("acquired") or 0),
                duplicates=int(row.get("duplicates") or 0),
                cost_usd=float(row.get("cost_usd") or 0.0),
            )
        )
    return state


def research_package(state: CollectionState) -> list[dict[str, Any]]:
    """The package the source-aware specialist answers from.

    Only retained sources, and only what this run actually collected. It is
    built here rather than in the arm so that exactly one function decides
    what "the specialist saw" means.
    """

    package: list[dict[str, Any]] = []
    for record in sorted(state.retained_sources(), key=lambda r: r.source_id):
        candidate = record.candidate
        claims = state.claims_for_source(record.source_id)
        package.append(
            {
                "source_id": record.source_id,
                "artifact_id": record.artifact_id or record.source_id,
                "title": candidate.title,
                "published_at": candidate.published_at,
                "text": candidate.abstract or candidate.title,
                "locator": "abstract" if candidate.abstract else "title",
                "claims": [claim.claim_text for claim in claims],
            }
        )
    return package
