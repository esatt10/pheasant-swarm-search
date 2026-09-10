"""The search adapter.

The lab's search request is stable and independent of the server's external
names; the response is normalised into rows that carry everything a retrieval
metric or a lineage walk needs, including the fields that are *absent*. An
absent field is recorded as absent rather than defaulted, because a default
that looks like a measurement is how a metric ends up describing the adapter.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .. import ids
from ..hashing import digest_text
from ..settings import PheasantFile
from .capabilities import CapabilityMap
from .client import PheasantClient


@dataclass
class MemoryOptions:
    """How memory takes part in one search.

    ``enabled=False`` is spelled as the region's own ``"off"`` rather than by
    omitting the argument: omitting it selects the region's default, which is
    exactly the difference P0 and P1 exist to measure.
    """

    enabled: bool = False
    scopes: list[str] = field(default_factory=list)
    subject: str | None = None
    current_only: bool = True
    as_of: str | None = None
    max_results: int | None = None
    include_rules: bool = False
    steering: list[str] = field(default_factory=list)

    def as_argument(self) -> Any:
        if not self.enabled:
            return "off"
        options: dict[str, Any] = {
            "current_only": self.current_only,
            "include_rules": self.include_rules,
        }
        if self.scopes:
            options["scopes"] = list(self.scopes)
        if self.subject:
            options["subject"] = self.subject
        if self.as_of:
            options["as_of"] = self.as_of
        if self.max_results is not None:
            options["max_results"] = self.max_results
        return options


@dataclass
class SearchRequest:
    """Spec section 7.5, as data."""

    run_id: str
    arm_id: str
    question_id: str
    query: str
    namespace: str
    top_k: int = 10
    mode: str = "hybrid"
    snapshot_id: str | None = None
    as_of: str | None = None
    memory: MemoryOptions = field(default_factory=MemoryOptions)
    filters: dict[str, Any] = field(default_factory=dict)
    round: int = 1
    session: str | None = None
    principal: str | None = None

    @property
    def search_request_id(self) -> str:
        return ids.search_request_id(
            self.run_id, self.arm_id, self.question_id, self.query, self.round
        )

    def pin_sent(self, argument_map: Mapping[str, str]) -> bool:
        """Was this request actually pinned to a snapshot on the wire?"""

        return bool(self.snapshot_id) and "snapshot_id" in argument_map

    def as_arguments(
        self, argument_map: Mapping[str, str], kb_field: str, knowledge_base: str
    ) -> dict[str, Any]:
        arguments: dict[str, Any] = {
            kb_field: knowledge_base,
            argument_map.get("query", "query"): self.query,
            argument_map.get("max_results", "max_results"): self.top_k,
            argument_map.get("mode", "mode"): self.mode,
            argument_map.get("memory", "memory"): self.memory.as_argument(),
        }
        if self.session:
            arguments[argument_map.get("session", "session")] = self.session
        if self.principal:
            arguments[argument_map.get("principal", "principal")] = self.principal
        # The pin and the instant are sent **only when the region declares a
        # name for them**. Recording a snapshot id and then not sending it
        # makes a run that *looks* pinned and is not - the one failure a
        # sealed snapshot exists to prevent - so `pin_sent` records which
        # happened rather than leaving a reader to assume. Sending a name the
        # tool does not accept is the same lie with an error attached: a
        # region that has no pin gets no pin, and says so.
        if self.snapshot_id and "snapshot_id" in argument_map:
            arguments[argument_map["snapshot_id"]] = self.snapshot_id
        if self.as_of and "as_of" in argument_map:
            arguments[argument_map["as_of"]] = self.as_of
        for key, value in self.filters.items():
            arguments[argument_map.get(key, key)] = value
        return arguments


@dataclass
class SearchResult:
    """One returned row, with its provenance and what was not reported."""

    rank: int
    artifact_id: str | None
    content_digest: str | None
    score: float | None
    source_id: str | None
    locator: str | None
    retrieval_arm: str | None
    contributing_arms: list[str] = field(default_factory=list)
    memory_ids: list[str] = field(default_factory=list)
    steering_rules: list[str] = field(default_factory=list)
    matched_text: str | None = None
    matched_text_digest: str | None = None
    snapshot_id: str | None = None
    index_version: str | None = None
    source_type: str | None = None
    title: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "artifact_id": self.artifact_id,
            "content_digest": self.content_digest,
            "score": self.score,
            "source_id": self.source_id,
            "locator": self.locator,
            "retrieval_arm": self.retrieval_arm,
            "contributing_arms": list(self.contributing_arms),
            "memory_ids": list(self.memory_ids),
            "steering_rules": list(self.steering_rules),
            "matched_text_digest": self.matched_text_digest,
            "snapshot_id": self.snapshot_id,
            "index_version": self.index_version,
            "source_type": self.source_type,
            "title": self.title,
        }


@dataclass
class SearchResponse:
    request: SearchRequest
    results: list[SearchResult]
    latency_ms: float
    server_trace_id: str | None = None
    snapshot_id: str | None = None
    graph_generation: str | None = None
    warnings: list[str] = field(default_factory=list)
    partial: bool = False
    next_cursor: str | None = None
    raw_digest: str | None = None
    status: str = "succeeded"
    error: str | None = None
    #: Whether the snapshot pin reached the region, as opposed to being
    #: recorded locally. False with a snapshot_id set means the region offers
    #: no pin and the snapshot is a drift check only.
    pin_sent: bool = False

    @property
    def artifact_ids(self) -> list[str]:
        return [r.artifact_id for r in self.results if r.artifact_id]

    def as_record(self, *, store_text: bool = False) -> dict[str, Any]:
        return {
            "search_request_id": self.request.search_request_id,
            "run_id": self.request.run_id,
            "arm_id": self.request.arm_id,
            "question_id": self.request.question_id,
            "query": self.request.query,
            "round": self.request.round,
            "mode": self.request.mode,
            "top_k": self.request.top_k,
            "memory_enabled": self.request.memory.enabled,
            "snapshot_id": self.snapshot_id or self.request.snapshot_id,
            "pin_sent": self.pin_sent,
            "as_of": self.request.as_of,
            "graph_generation": self.graph_generation,
            "latency_ms": self.latency_ms,
            "server_trace_id": self.server_trace_id,
            "warnings": list(self.warnings),
            "partial": self.partial,
            "status": self.status,
            "error": self.error,
            "results": [
                {**r.as_dict(), **({"matched_text": r.matched_text} if store_text else {})}
                for r in self.results
            ],
        }


class Retriever:
    """One search surface, used identically by P0, P1 and P2."""

    def __init__(
        self,
        client: PheasantClient,
        capabilities: CapabilityMap,
        config: PheasantFile,
        *,
        store_text: bool = True,
    ) -> None:
        self.client = client
        self.capabilities = capabilities
        self.config = config
        self.store_text = store_text
        self._argument_map: dict[str, str] = dict(config.argument_map.get("search") or {})
        self._kb_field = str(config.argument_map.get("knowledge_base_field", "knowledge_base"))

    @property
    def supports_pinning(self) -> bool:
        """Does this region's search tool take a snapshot pin?

        Read from the configured argument map rather than guessed, and checked
        against the tool's advertised schema at preflight.
        """

        return "snapshot_id" in self._argument_map

    @property
    def supports_corpus_as_of(self) -> bool:
        return "as_of" in self._argument_map

    def search(self, request: SearchRequest) -> SearchResponse:
        tool = self.capabilities.tool("search")
        arguments = request.as_arguments(
            self._argument_map, self._kb_field, self.config.knowledge_base
        )
        outcome = self.client.call(
            tool,
            arguments,
            idempotent=True,
            stage="retrieval",
            question_id=request.question_id,
            arm_id=request.arm_id,
        )
        payload = outcome.result.payload() if outcome.result else {}
        body = payload if isinstance(payload, Mapping) else {}
        results = normalise_results(body, request)
        return SearchResponse(
            request=request,
            results=results,
            latency_ms=outcome.duration_ms,
            server_trace_id=outcome.server_trace_id,
            snapshot_id=str(body.get("snapshot_id") or request.snapshot_id or "") or None,
            graph_generation=_first_str(body, "graph_generation", "generation_id"),
            warnings=list(outcome.warnings),
            partial=outcome.partial,
            next_cursor=_first_str(body, "next_cursor", "cursor"),
            status=outcome.status,
            pin_sent=request.pin_sent(self._argument_map),
        )

    def describe(self) -> dict[str, Any]:
        if not self.capabilities.has("describe_retrieval"):
            return {}
        outcome = self.client.call(
            self.capabilities.tool("describe_retrieval"),
            {self._kb_field: self.config.knowledge_base},
            idempotent=True,
            stage="retrieval",
        )
        payload = outcome.result.payload() if outcome.result else {}
        return dict(payload) if isinstance(payload, Mapping) else {}

    def fetch(self, artifact_id: str) -> dict[str, Any] | None:
        if not self.capabilities.has("fetch"):
            return None
        outcome = self.client.call(
            self.capabilities.tool("fetch"),
            {
                self._kb_field: self.config.knowledge_base,
                "artifact_id": artifact_id,
                "file_path": artifact_id,
            },
            idempotent=True,
            stage="retrieval",
            allow_error=True,
        )
        payload = outcome.result.payload() if outcome.result else None
        return dict(payload) if isinstance(payload, Mapping) else None


def normalise_results(body: Mapping[str, Any], request: SearchRequest) -> list[SearchResult]:
    """Flatten whatever shape the region returned into ranked rows."""

    rows = body.get("results") or body.get("hits") or body.get("chunks") or []
    if not isinstance(rows, Sequence):
        return []
    results: list[SearchResult] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            continue
        provenance = row.get("provenance") if isinstance(row.get("provenance"), Mapping) else {}
        memory = row.get("memory") if isinstance(row.get("memory"), Mapping) else {}
        text = _first_str(row, "text", "content", "snippet", "matched_text")
        results.append(
            SearchResult(
                rank=int(row.get("rank") or index),
                artifact_id=_first_str(row, "artifact_id", "id", "node_id", "chunk_id"),
                content_digest=_first_str(row, "content_digest", "digest")
                or (provenance.get("content_digest") if provenance else None),
                score=_as_float(row.get("score") if "score" in row else row.get("relevance")),
                source_id=_first_str(row, "lab_source_id", "source_id")
                or (str(provenance.get("source_id")) if provenance.get("source_id") else None)
                or _metadata_source_id(row),
                locator=_first_str(row, "locator", "section", "path")
                or (str(provenance.get("path")) if provenance.get("path") else None),
                retrieval_arm=_first_str(row, "arm", "retrieval_arm", "matched_by"),
                contributing_arms=[
                    str(a) for a in (row.get("contributing_arms") or row.get("arms") or [])
                ],
                memory_ids=[str(memory.get("record_id"))] if memory.get("record_id") else [],
                steering_rules=[
                    str(r) for r in (row.get("steering") or row.get("steering_rules") or [])
                ],
                matched_text=text,
                matched_text_digest=digest_text(text) if text else None,
                snapshot_id=_first_str(row, "snapshot_id") or request.snapshot_id,
                index_version=_first_str(row, "index_version", "generation_id"),
                source_type=_first_str(row, "source_type")
                or (str(provenance.get("source_type")) if provenance.get("source_type") else None),
                title=_first_str(row, "title", "name"),
            )
        )
    return results


def _metadata_source_id(row: Mapping[str, Any]) -> str | None:
    metadata = row.get("metadata")
    if isinstance(metadata, Mapping):
        value = metadata.get("lab_source_id") or metadata.get("source_id")
        if value:
            return str(value)
    return None


def _first_str(row: Mapping[str, Any], *names: str) -> str | None:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return str(value)
    return None


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
