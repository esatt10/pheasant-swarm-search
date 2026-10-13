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
    #: The region's graph expansion, as ``replay.graph_expansion`` spells it.
    #: Sent only when the argument map names ``expand``, like the pin.
    expand: Any = None
    #: ``replay.min_score``: sent only when set and when the map names it.
    min_score: float | None = None

    @property
    def search_request_id(self) -> str:
        return ids.search_request_id(
            self.run_id, self.arm_id, self.question_id, self.query, self.round
        )

    def pin_sent(self, argument_map: Mapping[str, str]) -> bool:
        """Was this request actually pinned to a snapshot on the wire?"""

        return bool(self.snapshot_id) and "snapshot_id" in argument_map

    def expand_sent(self, argument_map: Mapping[str, str]) -> bool:
        """Did this request ask the region for graph neighbourhoods?"""

        return self.expand is not None and "expand" in argument_map

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
        if self.expand_sent(argument_map):
            arguments[argument_map["expand"]] = self.expand
        if self.min_score is not None and "min_score" in argument_map:
            arguments[argument_map["min_score"]] = self.min_score
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
    #: The region's own name for the source the hit came from - needed to
    #: fetch the document back, and never confused with the lab's source id.
    region_source: str | None = None
    contributing_arms: list[str] = field(default_factory=list)
    memory_ids: list[str] = field(default_factory=list)
    steering_rules: list[str] = field(default_factory=list)
    matched_text: str | None = None
    matched_text_digest: str | None = None
    snapshot_id: str | None = None
    index_version: str | None = None
    source_type: str | None = None
    title: str | None = None
    #: The hit's graph neighbourhood, when the search asked for one: each
    #: neighbour's node id, type, label, the artifact it belongs to, its depth,
    #: the edge types it was reached by and the node it was reached ``via``.
    #: Empty when nothing was asked or the region walked nothing.
    graph_neighbors: list[dict[str, Any]] = field(default_factory=list)
    #: The region's path for the artifact inside its source - what pheasant
    #: resolves a Markdown link against. ``locator`` may be a section heading
    #: under taxonomy, so it cannot stand in for this.
    relative_path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "artifact_id": self.artifact_id,
            "content_digest": self.content_digest,
            "score": self.score,
            "source_id": self.source_id,
            "locator": self.locator,
            "region_source": self.region_source,
            "retrieval_arm": self.retrieval_arm,
            "contributing_arms": list(self.contributing_arms),
            "memory_ids": list(self.memory_ids),
            "steering_rules": list(self.steering_rules),
            "matched_text_digest": self.matched_text_digest,
            "snapshot_id": self.snapshot_id,
            "index_version": self.index_version,
            "source_type": self.source_type,
            "title": self.title,
            "graph_neighbors": [dict(n) for n in self.graph_neighbors],
            "relative_path": self.relative_path,
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
    #: Whether graph expansion was asked for, and the region's own account of
    #: what it walked (pheasant's ``expansion`` block: the settings in force,
    #: ``seeds``, ``seeds_skipped`` and ``nodes``). ``None`` when not asked, or
    #: when the region answered without one - recorded as absent, not as zero.
    expand_sent: bool = False
    expansion: dict[str, Any] | None = None

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
            "expand_sent": self.expand_sent,
            "expansion": dict(self.expansion) if self.expansion is not None else None,
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
        artifact_sources: Mapping[str, str] | None = None,
    ) -> None:
        self.client = client
        self.capabilities = capabilities
        self.config = config
        self.store_text = store_text
        #: Region artifact id -> this lab's source id, from the ingest
        #: receipts. Set by whoever holds the run's receipts (`evaluate`).
        self.artifact_sources: dict[str, str] = dict(artifact_sources or {})
        #: Region artifact id -> the relative path this lab submitted it under,
        #: for hits that name an artifact without a path (a graph-arm
        #: relationship hit carries only the node id).
        self.artifact_paths: dict[str, str] = {}
        self._argument_map: dict[str, str] = dict(config.argument_map.get("search") or {})
        self._kb_field = str(config.argument_map.get("knowledge_base_field", "knowledge_base"))
        self._documents: dict[tuple[str, str], str | None] = {}

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

    @property
    def supports_graph_expansion(self) -> bool:
        return "expand" in self._argument_map

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
        results = normalise_results(body, request, artifact_sources=self.artifact_sources)
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
            expand_sent=request.expand_sent(self._argument_map),
            expansion=dict(body["expansion"])
            if isinstance(body.get("expansion"), Mapping)
            else None,
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

    def fetch(self, path: str, source_name: str | None = None) -> dict[str, Any] | None:
        """One document, by the path the region reported for it.

        Pheasant's ``get_file_summary`` takes the artifact's ``path`` (the
        relative path a hit reports, scoped by ``source_name``) and returns
        the whole document as ``content``.
        """

        if not self.capabilities.has("fetch") or not path:
            return None
        arguments: dict[str, Any] = {self._kb_field: self.config.knowledge_base, "path": path}
        if source_name:
            arguments["source_name"] = source_name
        outcome = self.client.call(
            self.capabilities.tool("fetch"),
            arguments,
            idempotent=True,
            stage="retrieval",
            allow_error=True,
        )
        if outcome.result is None or outcome.result.is_error:
            return None
        payload = outcome.result.payload()
        return dict(payload) if isinstance(payload, Mapping) else None

    def hydrate(self, passages: Sequence[dict[str, Any]]) -> None:
        """Replace each passage's preview with the document it came from.

        A search hit carries a preview the region caps (500 characters on
        pheasant), and this lab's documents open with front matter - so an
        answerer handed previews reads metadata and very little of the
        abstract, and its score measures the preview length rather than
        retrieval. Pheasant's own answering path reads whole documents for
        the same reason. A passage that cannot be fetched keeps its preview
        and says so in ``text_source``; nothing is ever invented.
        """

        if not self.capabilities.has("fetch"):
            for passage in passages:
                passage.setdefault("text_source", "preview")
            return
        for passage in passages:
            passage.setdefault("text_source", "preview")
            artifact_id = str(passage.get("artifact_id") or "")
            candidates: list[tuple[str, str]] = []
            # The path this lab submitted the artifact under comes first: it
            # is known rather than reported, and a hit's `locator` may be a
            # section heading (taxonomy) or absent (a graph relationship hit).
            if artifact_id in self.artifact_paths:
                candidates.append((self.config.source_name, self.artifact_paths[artifact_id]))
            if passage.get("locator"):
                candidates.append(
                    (str(passage.get("region_source") or ""), str(passage["locator"]))
                )
            for key in candidates:
                if key not in self._documents:
                    document = self.fetch(key[1], key[0] or None) or {}
                    content = document.get("content") or document.get("text")
                    self._documents[key] = str(content) if content else None
                text = self._documents[key]
                if text:
                    passage["text"] = text
                    passage["text_source"] = "document"
                    break


def normalise_results(
    body: Mapping[str, Any],
    request: SearchRequest,
    *,
    artifact_sources: Mapping[str, str] | None = None,
) -> list[SearchResult]:
    """Flatten whatever shape the region returned into ranked rows.

    ``artifact_sources`` maps the region's artifact ids onto this lab's source
    ids, built from the ingest receipts. It is the authoritative join: the
    region's own ``provenance.source_id`` names *its* source (one per
    submission namespace), not the lab's, and reading it as the lab's would
    collapse every hit onto one source id - which every source-level metric
    would then count as a single document.
    """

    rows = body.get("results") or body.get("hits") or body.get("chunks") or []
    if not isinstance(rows, Sequence):
        return []
    results: list[SearchResult] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            continue
        provenance = row.get("provenance") if isinstance(row.get("provenance"), Mapping) else {}
        memory = row.get("memory") if isinstance(row.get("memory"), Mapping) else {}
        text = _first_str(row, "text", "content", "snippet", "matched_text") or _preview(row)
        artifact_id = _first_str(row, "artifact_id", "id", "node_id", "chunk_id")
        results.append(
            SearchResult(
                rank=int(row.get("rank") or index),
                artifact_id=artifact_id,
                content_digest=_first_str(row, "content_digest", "digest")
                or (provenance.get("content_digest") if provenance else None),
                score=_as_float(row.get("score") if "score" in row else row.get("relevance")),
                source_id=_first_str(row, "lab_source_id")
                or _metadata_source_id(row)
                or (artifact_sources or {}).get(artifact_id or "")
                or _front_matter_source_id(text),
                locator=_first_str(row, "locator", "section", "relative_path", "path")
                or _first_str(provenance, "relative_path", "path"),
                # A graph-arm node hit carries the node's own provenance, which
                # may not name a source; the hit's top-level `source_id` does.
                # Both are the *region's* name, and only ever fill this field.
                region_source=_first_str(provenance, "source_id", "source_name")
                or _first_str(row, "source_name", "source_id"),
                retrieval_arm=_first_str(row, "arm", "retrieval_arm", "retrieved_by", "matched_by"),
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
                graph_neighbors=_neighbors(row),
                relative_path=_first_str(row, "relative_path")
                or _first_str(provenance, "relative_path"),
            )
        )
    return results


#: The neighbour fields a run keeps. Anything else the region attaches is
#: left out rather than copied, so a record's shape is this list.
_NEIGHBOR_FIELDS = (
    "node_id",
    "type",
    "label",
    "artifact_id",
    "relative_path",
    "depth",
    "edge_types",
    "via",
)


def _neighbors(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The hit's ``graph.neighbors``, projected to :data:`_NEIGHBOR_FIELDS`."""

    graph = row.get("graph")
    if not isinstance(graph, Mapping):
        return []
    neighbors = graph.get("neighbors")
    if not isinstance(neighbors, Sequence) or isinstance(neighbors, str):
        return []
    return [
        {name: neighbor[name] for name in _NEIGHBOR_FIELDS if name in neighbor}
        for neighbor in neighbors
        if isinstance(neighbor, Mapping) and neighbor.get("node_id")
    ]


def _preview(row: Mapping[str, Any]) -> str | None:
    """The passage text a region returned under a name other than ``text``.

    Pheasant's hits carry the matched chunks' ``text_preview`` (and a shorter
    ``summary``) rather than a ``text`` field. Both are previews, capped by the
    region; :meth:`Retriever.hydrate` replaces them with the document itself.
    """

    chunks = row.get("chunks")
    if isinstance(chunks, Sequence) and not isinstance(chunks, str):
        parts = [
            str(chunk.get("text_preview") or chunk.get("text") or "")
            for chunk in chunks
            if isinstance(chunk, Mapping)
        ]
        joined = "\n".join(part for part in parts if part)
        if joined:
            return joined
    return _first_str(row, "summary")


def _front_matter_source_id(text: str | None) -> str | None:
    """The ``lab_source_id`` this lab wrote into a document's front matter.

    The last resort, for a hit whose artifact the receipts do not name (a
    region populated by an earlier run). It reads only what the researcher
    itself wrote, never anything the region inferred.
    """

    if not text or not text.startswith("---"):
        return None
    for line in text.splitlines()[1:40]:
        if line.strip() == "---":
            return None
        key, _, value = line.partition(":")
        if key.strip() == "lab_source_id" and value.strip():
            return value.strip()
    return None


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
