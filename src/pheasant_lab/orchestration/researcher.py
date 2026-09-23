"""One bounded research branch.

The branch is the unit of concurrency and the unit of the budget. Everything
it does is recorded: the candidates it rejected and why, the acquisitions it
attempted and what happened, the claims it extracted and on what basis, and
the receipt for every item it submitted to Pheasant.

A transport success is not an ingest, and this module never records one as
one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .. import ids
from ..budget import BudgetExceeded, CostLedger
from ..lifecycle import isonow
from ..models import ModelProvider, ModelRequest
from ..pheasant.ingestion import Ingestor, IngestRequest, build_requests
from ..promptlib import load as load_prompt
from ..providers.base import LiteratureProvider, ProviderError, SourceCandidate
from ..settings import LabConfig, Topic
from ..textkit import truncate
from ..tracing.errors import ErrorImpact
from .planner import Subtopic
from .state import (
    AcquisitionAttempt,
    ClaimRecord,
    CollectionState,
    ContradictionRecord,
    SourceRecord,
    SourceState,
    make_claim,
)

PERMISSIVE_LICENCES = ("cc-by", "cc0", "cc-by-sa", "public-domain", "arxiv-nonexclusive")


@dataclass
class BranchResult:
    subtopic_id: str
    agent_id: str
    discovered: int = 0
    retained: int = 0
    rejected: int = 0
    blocked: int = 0
    duplicates: int = 0
    new_claims: int = 0
    submitted: int = 0
    receipts: int = 0
    contradictions: int = 0
    errors: int = 0
    cost_usd: float = 0.0
    gaps: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "subtopic_id": self.subtopic_id,
            "agent_id": self.agent_id,
            "discovered": self.discovered,
            "retained": self.retained,
            "rejected": self.rejected,
            "blocked": self.blocked,
            "duplicates": self.duplicates,
            "new_claims": self.new_claims,
            "submitted": self.submitted,
            "receipts": self.receipts,
            "contradictions": self.contradictions,
            "errors": self.errors,
            "cost_usd": round(self.cost_usd, 6),
            "gaps": list(self.gaps),
        }


class Researcher:
    """Runs one subtopic to its bounds and stops."""

    def __init__(
        self,
        config: LabConfig,
        *,
        topic: Topic,
        providers: Sequence[LiteratureProvider],
        model: ModelProvider,
        ledger: CostLedger,
        ingestor: Ingestor | None,
        tracer: Any,
        run_id: str,
    ) -> None:
        self.config = config
        self.topic = topic
        self.providers = list(providers)
        self.model = model
        self.ledger = ledger
        self.ingestor = ingestor
        self.tracer = tracer
        self.run_id = run_id
        self.prompt = load_prompt("researcher")

    # -- the branch, in three phases ---------------------------------------
    #
    # Discovery is concurrent and touches no shared state. **Admission is
    # sequential and ordered**, because it is the only phase that reads and
    # writes the shared source registry: which branch reaches a shared
    # candidate first decides which subtopic's question drives its extraction,
    # and letting thread scheduling decide that makes two runs of one
    # configuration disagree about their own corpus. Extraction and ingestion
    # are concurrent again, because by then each branch owns its own records.

    def agent_for(self, ordinal: int) -> str:
        return ids.agent_id("researcher", self.run_id, ordinal)

    def search_candidates(
        self,
        subtopic: Subtopic,
        result: BranchResult,
        *,
        max_rounds: int,
        max_sources: int,
    ) -> list[SourceCandidate]:
        """Phase 1: ask the providers. Concurrent, shares nothing."""

        found: list[SourceCandidate] = []
        seen: set[str] = set()
        queries = subtopic.queries(
            rounds=min(max_rounds, self.config.collection.max_search_rounds_per_agent)
        )
        with self.tracer.span(
            "research.search",
            agent_id=result.agent_id,
            agent_role="researcher",
            topic_id=self.topic.id,
            stage="discovery",
            subtopic_id=subtopic.subtopic_id,
        ):
            for query in queries:
                if len(found) >= max_sources:
                    break
                per_provider = self.config.collection.max_results_per_provider
                for provider in self._providers_for(subtopic):
                    if len(found) >= max_sources:
                        break
                    limit = max_sources - len(found)
                    if per_provider is not None:
                        limit = min(limit, per_provider)
                    for candidate in self._search(provider, query, subtopic, result, limit):
                        if candidate.source_id in seen:
                            continue
                        seen.add(candidate.source_id)
                        found.append(candidate)
                        if len(found) >= max_sources:
                            break
        return found

    def admit_one(
        self,
        subtopic: Subtopic,
        candidate: SourceCandidate,
        state: CollectionState,
        result: BranchResult,
        *,
        round_: int,
    ) -> SourceRecord | None:
        """Phase 2, one candidate. Returns the record only if it was retained."""

        record = self._admit(candidate, subtopic, state, result, result.agent_id, round_)
        return record if record is not None and record.retained else None

    def admit_candidates(
        self,
        subtopic: Subtopic,
        candidates: Sequence[SourceCandidate],
        state: CollectionState,
        result: BranchResult,
        *,
        round_: int,
        max_sources: int,
    ) -> list[SourceRecord]:
        """Phase 2 for one branch in isolation. Used by the tests."""

        fresh: list[SourceRecord] = []
        for candidate in candidates:
            if len(fresh) >= max_sources:
                break
            record = self.admit_one(subtopic, candidate, state, result, round_=round_)
            if record is not None:
                fresh.append(record)
        return fresh

    def extract_and_persist(
        self,
        subtopic: Subtopic,
        state: CollectionState,
        result: BranchResult,
        fresh: list[SourceRecord],
        *,
        round_: int,
    ) -> BranchResult:
        """Phase 3: extract claims and submit. Concurrent again."""

        with self.tracer.span(
            "research.branch",
            agent_id=result.agent_id,
            agent_role="researcher",
            topic_id=self.topic.id,
            stage="extraction",
            subtopic_id=subtopic.subtopic_id,
        ):
            self._extract(subtopic, state, result, fresh, result.agent_id, round_)
            self._persist(subtopic, state, result, fresh, result.agent_id)
        return result

    def _providers_for(self, subtopic: Subtopic) -> list[LiteratureProvider]:
        wanted = set(subtopic.providers) if subtopic.providers else None
        chosen = [p for p in self.providers if wanted is None or p.name in wanted]
        return chosen or list(self.providers)

    def _search(
        self,
        provider: LiteratureProvider,
        query: str,
        subtopic: Subtopic,
        result: BranchResult,
        limit: int,
    ) -> list[SourceCandidate]:
        with self.tracer.operation(
            "collection.search",
            payload={
                "provider": provider.name,
                "query": truncate(query, 200),
                "subtopic_id": subtopic.subtopic_id,
            },
        ) as outcome:
            try:
                if provider.usd_per_request:
                    self.ledger.record_external(
                        bucket="collection",
                        role="researcher",
                        label=f"provider:{provider.name}",
                        amount_usd=provider.usd_per_request,
                    )
                candidates = provider.search(
                    query,
                    limit=max(1, limit),
                    date_from=subtopic.date_from,
                    date_to=subtopic.date_to,
                )
            except (ProviderError, BudgetExceeded) as exc:
                result.errors += 1
                self.tracer.errors.record(
                    exc,
                    stage="discovery",
                    component=f"providers.{provider.name}",
                    operation="search",
                    resolution="skipped",
                    trace_id=self.tracer.trace_id,
                    span_id=self.tracer.span_id,
                    impact=ErrorImpact(comparability="partial"),
                )
                outcome["_status"] = "failed"
                outcome["error"] = str(exc)
                return []
            outcome["results"] = len(candidates)
            result.discovered += len(candidates)
            return candidates

    def _admit(
        self,
        candidate: SourceCandidate,
        subtopic: Subtopic,
        state: CollectionState,
        result: BranchResult,
        agent_id: str,
        round_: int,
    ) -> SourceRecord | None:
        # The check and the write are one step: two branches that both find
        # nothing and both write produce a corpus whose size depends on thread
        # scheduling.
        with state.lock:
            known = state.known(candidate)
            if known is None:
                placeholder = SourceRecord(
                    source_id=candidate.source_id,
                    candidate=candidate,
                    subtopic_id=subtopic.subtopic_id,
                    facet_ids=list(subtopic.facet_ids),
                    researcher_agent_id=agent_id,
                    round=round_,
                )
                state.add_source(placeholder)
        if known is not None:
            existing = state.sources[known]
            if existing.source_id != candidate.source_id:
                result.duplicates += 1
                duplicate = SourceRecord(
                    source_id=candidate.source_id,
                    candidate=candidate,
                    subtopic_id=subtopic.subtopic_id,
                    facet_ids=list(subtopic.facet_ids),
                    state=SourceState.rejected,
                    reason_code="duplicate",
                    researcher_agent_id=agent_id,
                    duplicate_of=known,
                    round=round_,
                )
                state.add_source(duplicate)
            return None

        event = self.tracer.emit(
            "collection.discovered",
            payload={
                "source_id": candidate.source_id,
                "provider": candidate.provider,
                "title": truncate(candidate.title, 200),
                "stable_identifier": candidate.stable_identifier,
                "subtopic_id": subtopic.subtopic_id,
            },
            agent_id=agent_id,
            agent_role="researcher",
            topic_id=self.topic.id,
        )
        record = placeholder
        record.discovery_event_id = event.event_id

        reason = self._validate(candidate)
        if reason is not None:
            record.transition(
                SourceState.blocked if reason in {"licence", "paywalled"} else SourceState.rejected,
                reason_code=reason,
            )
            if record.state is SourceState.blocked:
                result.blocked += 1
            else:
                result.rejected += 1
            return record

        record.transition(SourceState.validated)
        self._acquire(record, result)
        return record

    def _validate(self, candidate: SourceCandidate) -> str | None:
        collection = self.config.collection
        if candidate.source_type not in collection.allowed_source_types:
            return "disallowed_type"
        if collection.require_stable_identifier and not candidate.stable_identifier:
            return "no_stable_identifier"
        window = self.topic.date_range
        if window.from_ and candidate.published_at and candidate.published_at < window.from_:
            return "out_of_window"
        if window.to and candidate.published_at and candidate.published_at > window.to:
            return "out_of_window"
        if not candidate.has_text and not collection.permit_abstract_only:
            return "no_text"
        if not candidate.has_text:
            return "no_text"
        return None

    def _acquire(self, record: SourceRecord, result: BranchResult) -> None:
        """Acquire what policy permits, and record what it refused.

        Full text is fetched only when the licence permits it *and* the config
        asks for it; otherwise the abstract and the canonical locator are what
        is retained, and the record says so rather than looking like a full
        acquisition that happened to be short.
        """

        candidate = record.candidate
        event = self.tracer.emit(
            "collection.acquired",
            payload={
                "source_id": record.source_id,
                "mode": "abstract",
                "licence": candidate.license,
                "open_access": candidate.open_access,
            },
            agent_id=record.researcher_agent_id,
            agent_role="researcher",
            topic_id=self.topic.id,
        )
        record.acquisition_event_id = event.event_id
        licensed = any(token in (candidate.license or "").lower() for token in PERMISSIVE_LICENCES)
        if self.config.collection.download_full_text_only_when_licensed and not licensed:
            record.acquisition_attempts.append(
                AcquisitionAttempt(
                    isonow(), "abstract_only", "licence does not permit full-text retrieval"
                )
            )
        else:
            record.acquisition_attempts.append(
                AcquisitionAttempt(
                    isonow(), "abstract_only", "full-text acquisition not requested for this run"
                )
            )
        record.transition(SourceState.acquired)
        result.retained += 1

    # -- extraction --------------------------------------------------------
    def _extract(
        self,
        subtopic: Subtopic,
        state: CollectionState,
        result: BranchResult,
        fresh: list[SourceRecord],
        agent_id: str,
        round_: int,
    ) -> None:
        if not fresh:
            return
        spec = self.config.role("researcher")
        context = {
            "subtopic_id": subtopic.subtopic_id,
            "question": subtopic.question,
            "candidates": [
                {
                    "source_id": record.source_id,
                    "title": record.candidate.title,
                    "abstract": record.candidate.abstract,
                    "published_at": record.candidate.published_at,
                    "source_type": record.candidate.source_type,
                }
                for record in fresh
            ],
            "known_gaps": [],
        }
        request = ModelRequest(
            role="researcher",
            schema="research",
            system=self.prompt,
            user=_render_extraction(subtopic, fresh),
            context=context,
            max_output_tokens=spec.max_output_tokens,
            temperature=spec.temperature,
            seed=self.config.experiment.seed,
        )
        with self.tracer.span(
            "research.extract", stage="extraction", agent_id=agent_id, agent_role="researcher"
        ):
            with self.ledger.spend(
                bucket="collection",
                role="researcher",
                model=spec.model,
                prompt=request.prompt_text,
                max_output_tokens=spec.max_output_tokens,
                tool_calls=0,
            ) as cost:
                response = self.model.complete(request)
                cost.input_tokens = response.input_tokens
                cost.output_tokens = response.output_tokens
            result.cost_usd += cost.actual_usd or 0.0
            self.tracer.emit("cost.model_call", payload=cost.as_payload())

        known_sources = {record.source_id for record in fresh}
        claims: list[ClaimRecord] = []
        for payload in response.data.get("claims", []):
            if str(payload.get("source_id")) not in known_sources:
                # A claim naming a source this branch did not retrieve cannot
                # be resolved through the lineage chain, so it is dropped and
                # counted rather than stored as evidence.
                self.tracer.emit(
                    "collection.claim_rejected",
                    status="skipped",
                    payload={"reason": "unknown_source", "source_id": payload.get("source_id")},
                )
                continue
            claim = make_claim(
                payload,
                subtopic_id=subtopic.subtopic_id,
                facet_ids=subtopic.facet_ids,
                researcher_agent_id=agent_id,
                round_=round_,
            )
            if not claim.claim_text:
                continue
            if state.add_claim(claim):
                claims.append(claim)
                result.new_claims += 1

        for record in fresh:
            if state.claims_for_source(record.source_id):
                record.transition(SourceState.extracted)

        for payload in response.data.get("contradictions", []):
            source_ids = [str(s) for s in payload.get("source_ids", []) if s]
            claim_ids = [str(c) for c in payload.get("claim_ids", []) if c]
            about = str(payload.get("about") or "")
            contradiction = ContradictionRecord(
                contradiction_id=ids.contradiction_id(subtopic.subtopic_id, about),
                about=about,
                claim_ids=claim_ids,
                source_ids=source_ids,
                severity=str(payload.get("severity") or "material"),
                subtopic_id=subtopic.subtopic_id,
                resolved=bool(payload.get("resolved")),
                resolution_note=payload.get("resolution_note"),
            )
            state.contradictions[contradiction.contradiction_id] = contradiction
            self.tracer.append("contradictions.jsonl", contradiction.as_dict())
            result.contradictions += 1

        result.gaps.extend(str(gap) for gap in response.data.get("gaps", []) if gap)
        for claim in claims:
            self.tracer.append("claims.jsonl", claim.as_record(self.run_id))

    # -- persistence -------------------------------------------------------
    def _persist(
        self,
        subtopic: Subtopic,
        state: CollectionState,
        result: BranchResult,
        fresh: list[SourceRecord],
        agent_id: str,
    ) -> None:
        submitable = [r for r in fresh if r.state in {SourceState.extracted, SourceState.acquired}]
        if submitable and self.ingestor is not None:
            requests = build_requests(
                (
                    {
                        "source_id": record.source_id,
                        "title": record.candidate.title,
                        "stable_identifier": record.candidate.stable_identifier,
                        "canonical_url": record.candidate.canonical_url,
                        "authors": record.candidate.authors,
                        "published_at": record.candidate.published_at,
                        "license": record.candidate.license,
                        "text": _document(record),
                        "media_type": "text/markdown",
                        "discovery_event_id": record.discovery_event_id,
                        "acquisition_event_id": record.acquisition_event_id,
                        "researcher_agent_id": agent_id,
                    }
                    for record in submitable
                ),
                run_id=self.run_id,
                topic_id=self.topic.id,
                namespace=self.config.pheasant.knowledge_base,
            )
            by_key = {request.idempotency_key: request for request in requests}
            for record, request in zip(submitable, requests, strict=True):
                record.idempotency_key = request.idempotency_key
                record.transition(SourceState.submitted)
            result.submitted += len(requests)

            with self.tracer.span(
                "research.ingest", stage="ingest", agent_id=agent_id, agent_role="researcher"
            ):
                receipts = self.ingestor.submit(requests)
            for receipt in receipts:
                request = by_key.get(receipt.idempotency_key)
                record = state.sources.get(
                    receipt.source_id or (request.source_id if request else "")
                )
                if record is None:
                    continue
                if receipt.accepted:
                    record.artifact_id = receipt.artifact_id
                    record.transition(
                        SourceState.indexed if receipt.indexed else SourceState.submitted
                    )
                    result.receipts += 1
                else:
                    record.transition(
                        SourceState.failed_terminal, reason_code=receipt.error_code or "no_receipt"
                    )

        for record in fresh:
            self.tracer.append("sources.jsonl", record.as_record(self.run_id))
        state.subtopics[subtopic.subtopic_id] = subtopic.as_dict()


def _document(record: SourceRecord) -> str:
    """The markdown a source becomes in the region.

    Front matter carries the provenance a retrieval result needs to be traced
    back to its source, because a result the lab cannot resolve to a source is
    a result no metric can use.
    """

    candidate = record.candidate
    authors = ", ".join(candidate.authors[:12])
    lines = [
        "---",
        f"lab_source_id: {record.source_id}",
        f"title: {candidate.title}",
        f"stable_identifier: {candidate.stable_identifier or ''}",
        f"canonical_url: {candidate.canonical_url or ''}",
        f"published_at: {candidate.published_at or ''}",
        f"source_type: {candidate.source_type}",
        f"venue: {candidate.venue or ''}",
        f"license: {candidate.license}",
        f"authors: {authors}",
        f"provider: {candidate.provider}",
        "---",
        "",
        f"# {candidate.title}",
        "",
        candidate.abstract or "",
    ]
    return "\n".join(lines).strip() + "\n"


def _render_extraction(subtopic: Subtopic, records: list[SourceRecord]) -> str:
    blocks = []
    for record in records:
        candidate = record.candidate
        blocks.append(
            f"### {record.source_id}\n"
            f"title: {candidate.title}\n"
            f"published: {candidate.published_at or 'unknown'}\n"
            f"type: {candidate.source_type}\n\n"
            f"{candidate.abstract or '(no abstract)'}\n"
        )
    return (
        f"Subtopic: {subtopic.question}\n"
        f"Terminology: {', '.join(subtopic.terminology)}\n\n"
        "Extract atomic claims from the sources below. Return the JSON object described above "
        "and nothing else.\n\n" + "\n".join(blocks)
    )


def ingest_requests_for(
    records: Sequence[SourceRecord], *, run_id: str, topic_id: str, namespace: str
) -> list[IngestRequest]:
    """Exposed for the contract tests: the exact documents a branch submits."""

    return build_requests(
        (
            {
                "source_id": record.source_id,
                "title": record.candidate.title,
                "stable_identifier": record.candidate.stable_identifier,
                "canonical_url": record.candidate.canonical_url,
                "authors": record.candidate.authors,
                "published_at": record.candidate.published_at,
                "license": record.candidate.license,
                "text": _document(record),
                "media_type": "text/markdown",
            }
            for record in records
        ),
        run_id=run_id,
        topic_id=topic_id,
        namespace=namespace,
    )
