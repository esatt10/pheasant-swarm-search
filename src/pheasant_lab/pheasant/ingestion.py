"""The ingest adapter.

The lab normalises every literature item into one internal shape before the
adapter maps it onto whatever this server's ingest tool is actually called and
whatever its parameters are actually named. That indirection is the point: a
server rename is a config edit here, not a code change, and the run manifest
records which spelling was used.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .. import ids
from ..hashing import digest, digest_text
from ..settings import PheasantFile
from .capabilities import CapabilityMap
from .client import PheasantClient
from .receipts import IngestReceipt, ReceiptLedger, parse_receipts

MEDIA_TYPES = ("text/plain", "text/markdown", "application/pdf", "metadata-only")


@dataclass
class IngestSource:
    title: str
    stable_identifier: str | None = None
    canonical_url: str | None = None
    authors: list[str] = field(default_factory=list)
    published_at: str | None = None
    license: str = "unknown"


@dataclass
class IngestPayload:
    media_type: str = "text/markdown"
    text: str = ""
    bytes: int | None = None
    local_artifact_ref: str | None = None


@dataclass
class IngestProvenance:
    discovery_event_id: str | None = None
    acquisition_event_id: str | None = None
    researcher_agent_id: str | None = None


@dataclass
class IngestRequest:
    """The lab's stable internal shape (spec section 7.4)."""

    run_id: str
    topic_id: str
    source_id: str
    namespace: str
    source: IngestSource
    payload: IngestPayload
    provenance: IngestProvenance = field(default_factory=IngestProvenance)
    operation: str = "upload"
    relative_path: str | None = None

    @property
    def content_digest(self) -> str:
        return (
            digest_text(self.payload.text or "")
            if self.payload.text
            else digest(
                {
                    "metadata_only": self.source.stable_identifier
                    or self.source.canonical_url
                    or self.source.title
                }
            )
        )

    @property
    def idempotency_key(self) -> str:
        return ids.idempotency_key("pheasant-lab", self.source_id, self.content_digest)

    @property
    def ingest_request_id(self) -> str:
        return ids.ingest_request_id(self.source_id, self.content_digest)

    def path(self) -> str:
        if self.relative_path:
            return self.relative_path
        slug = (self.source.stable_identifier or self.source_id).replace("/", "_").replace(":", "_")
        return f"{self.topic_id}/{slug}.md"

    def as_document(self, argument_map: Mapping[str, str]) -> dict[str, Any]:
        return {
            argument_map.get("document_path", "relative_path"): self.path(),
            argument_map.get("document_text", "text"): self.payload.text,
            argument_map.get("document_key", "idempotency_key"): self.idempotency_key,
            argument_map.get("document_metadata", "metadata"): self.metadata(),
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "lab_run_id": self.run_id,
            "lab_source_id": self.source_id,
            "lab_topic_id": self.topic_id,
            "lab_ingest_request_id": self.ingest_request_id,
            "title": self.source.title,
            "stable_identifier": self.source.stable_identifier,
            "canonical_url": self.source.canonical_url,
            "authors": list(self.source.authors),
            "published_at": self.source.published_at,
            "license": self.source.license,
            "media_type": self.payload.media_type,
            "content_digest": self.content_digest,
            "discovery_event_id": self.provenance.discovery_event_id,
            "acquisition_event_id": self.provenance.acquisition_event_id,
            "researcher_agent_id": self.provenance.researcher_agent_id,
        }

    def as_record(self) -> dict[str, Any]:
        return {
            "ingest_request_id": self.ingest_request_id,
            "run_id": self.run_id,
            "topic_id": self.topic_id,
            "source_id": self.source_id,
            "content_digest": self.content_digest,
            "idempotency_key": self.idempotency_key,
            "namespace": self.namespace,
            "operation": self.operation,
            "relative_path": self.path(),
            "media_type": self.payload.media_type,
            "bytes": self.payload.bytes,
            "license": self.source.license,
        }


class Ingestor:
    """Submits, syncs, acknowledges and reconciles."""

    def __init__(
        self,
        client: PheasantClient,
        capabilities: CapabilityMap,
        config: PheasantFile,
        *,
        run_id: str,
        ledger: ReceiptLedger | None = None,
        tracer: Any = None,
    ) -> None:
        self.client = client
        self.capabilities = capabilities
        self.config = config
        self.run_id = run_id
        self.ledger = ledger or ReceiptLedger()
        self.tracer = tracer
        self._submission_directory: str | None = None
        self._argument_map: dict[str, str] = dict(config.argument_map.get("ingest") or {})
        self._kb_field = str(config.argument_map.get("knowledge_base_field", "knowledge_base"))

    # -- submission --------------------------------------------------------
    def submit(
        self, requests: Sequence[IngestRequest], *, batch_size: int = 20
    ) -> list[IngestReceipt]:
        """Submit in batches, one receipt per item.

        Batching is bounded rather than unbounded because one malformed item
        inside a single transaction takes every good item beside it down; a
        bounded batch keeps that blast radius to something a retry can afford.
        """

        tool = self.capabilities.tool("ingest")
        receipts: list[IngestReceipt] = []
        for start in range(0, len(requests), batch_size):
            chunk = list(requests[start : start + batch_size])
            receipts.extend(self._submit_batch(tool, chunk))
        return receipts

    def _submit_batch(self, tool: str, chunk: Sequence[IngestRequest]) -> list[IngestReceipt]:
        submission_id = (
            "submission-" + digest([r.idempotency_key for r in chunk]).removeprefix("sha256:")[:16]
        )
        documents = [request.as_document(self._argument_map) for request in chunk]
        key_to_source = {request.idempotency_key: request.source_id for request in chunk}
        digests = {request.idempotency_key: request.content_digest for request in chunk}
        for request in chunk:
            self.ledger.note_submission(request.idempotency_key)
            if self.tracer is not None:
                self.tracer.append("ingest-requests.jsonl", request.as_record())

        arguments = {
            self._kb_field: self.config.knowledge_base,
            self._argument_map.get("documents", "documents"): documents,
            self._argument_map.get("source_name", "source_name"): self.config.source_name,
            self._argument_map.get("submission_id", "submission_id"): submission_id,
        }
        outcome = self.client.call(
            tool,
            arguments,
            idempotency_key=submission_id,
            stage="ingest",
        )
        payload = outcome.result.payload() if outcome.result else {}
        if isinstance(payload, Mapping) and payload.get("directory"):
            self._submission_directory = str(payload["directory"])
        receipts = parse_receipts(
            payload if isinstance(payload, Mapping | list) else [],
            run_id=self.run_id,
            key_to_source=key_to_source,
            submission_id=submission_id,
            requested_digests=digests,
        )
        # A region that answered without receipts has not ingested; record the
        # absence rather than inferring success from a 200.
        if not receipts:
            for request in chunk:
                receipts.append(
                    IngestReceipt(
                        receipt_id=f"receipt-missing-{request.idempotency_key[-12:]}",
                        run_id=self.run_id,
                        source_id=request.source_id,
                        idempotency_key=request.idempotency_key,
                        submission_id=submission_id,
                        status="no_receipt",
                        content_digest=request.content_digest,
                        error_message="transport succeeded and the region returned no receipt",
                    )
                )
        for receipt in receipts:
            self.ledger.record(receipt)
            if self.tracer is not None:
                self.tracer.append("ingest-receipts.jsonl", receipt.as_dict())
        return receipts

    # -- the index barrier -------------------------------------------------
    def sync(self, *, mode: str = "incremental") -> dict[str, Any]:
        if not self.capabilities.has("sync"):
            return {"skipped": "no sync capability configured"}
        self._register_submission_source()
        outcome = self.client.call(
            self.capabilities.tool("sync"),
            {
                self._kb_field: self.config.knowledge_base,
                "source_name": self.config.source_name,
                "mode": mode,
            },
            idempotent=False,
            stage="index",
        )
        return _as_mapping(outcome.result.payload() if outcome.result else {})

    def _register_submission_source(self) -> None:
        """Register the server's intake directory without replacing another source."""

        if not self._submission_directory:
            return  # Some regions, including the mock, index submissions directly.
        if not self.capabilities.has("list_sources") or not self.capabilities.has(
            "register_source"
        ):
            raise RuntimeError(
                "This region requires an indexed submission source. Configure the "
                "list_sources and register_source capabilities before collecting."
            )
        offset = 0
        while True:
            outcome = self.client.call(
                self.capabilities.tool("list_sources"),
                {self._kb_field: self.config.knowledge_base, "limit": 100, "offset": offset},
                idempotent=True,
                stage="index",
            )
            payload = _as_mapping(outcome.result.payload() if outcome.result else {})
            sources = payload.get("sources")
            if not isinstance(sources, list):
                raise RuntimeError("Cannot verify existing sources: list_sources returned no list.")
            for source in sources:
                if source.get("name") != self.config.source_name:
                    continue
                if source.get("path") != self._submission_directory:
                    raise RuntimeError(
                        f"Source {self.config.source_name!r} already indexes another path; "
                        "choose a new dedicated source_name for the lab."
                    )
                return
            if len(sources) < 100:
                break
            offset += len(sources)
        self.client.call(
            self.capabilities.tool("register_source"),
            {
                self._kb_field: self.config.knowledge_base,
                "name": self.config.source_name,
                "source_type": "document_folder",
                "path": self._submission_directory,
            },
            idempotent=False,
            stage="index",
        )

    def acknowledge(self, submission_id: str | None = None) -> dict[str, Any]:
        """Cross the index barrier from what the region *holds*.

        Not from what a sync reported: a sync's summary is a claim about what
        it did, and this is a question about what the region has.
        """

        if not self.capabilities.has("ingest_acknowledge"):
            return {"skipped": "no acknowledge capability configured"}
        arguments: dict[str, Any] = {self._kb_field: self.config.knowledge_base}
        if submission_id:
            arguments["submission_id"] = submission_id
        outcome = self.client.call(
            self.capabilities.tool("ingest_acknowledge"), arguments, idempotent=True, stage="index"
        )
        payload = _as_mapping(outcome.result.payload() if outcome.result else {})
        rows = payload.get("receipts", payload.get("acknowledged", []))
        if not isinstance(rows, list):
            # Pheasant 0.12.5 acknowledges a count; query the actual dispositions.
            # Fetch by key so a server's listing limit cannot hide this run's items.
            rows = [
                row
                for receipt in self.ledger.receipts
                if submission_id is None or receipt.submission_id == submission_id
                for row in self.ingest_status(idempotency_key=receipt.idempotency_key).get(
                    "receipts", []
                )
            ]
        for row in rows:
            if not isinstance(row, Mapping) or not row.get("idempotency_key"):
                continue
            key = str(row["idempotency_key"])
            self.ledger.acknowledge(
                key,
                status=str(row.get("status") or row.get("disposition") or "unknown"),
                artifact_id=row.get("artifact_id"),
            )
            receipt = self.ledger.get(key)
            if receipt is not None and self.tracer is not None:
                # Appended, not rewritten: the trace records that the barrier
                # was crossed, and a reader folds the file by key.
                self.tracer.append("ingest-receipts.jsonl", receipt.as_dict())
        return payload

    def reconcile(self, submission_id: str | None = None) -> dict[str, Any]:
        """Submitted against held. ``silent_loss`` is the number to read."""

        if not self.capabilities.has("ingest_reconcile"):
            local = self.ledger.submitted_without_receipt()
            return {
                "source": "local",
                "silent_loss": len(local),
                "keys": local,
                "limitation": "the region offers no reconcile tool; this compares submissions "
                "against receipts this lab holds, which cannot see an artifact lost after a receipt",
            }
        arguments: dict[str, Any] = {self._kb_field: self.config.knowledge_base}
        if submission_id:
            arguments["submission_id"] = submission_id
        outcome = self.client.call(
            self.capabilities.tool("ingest_reconcile"), arguments, idempotent=True, stage="index"
        )
        return _as_mapping(outcome.result.payload() if outcome.result else {})

    def ingest_status(
        self, *, idempotency_key: str | None = None, submission_id: str | None = None
    ) -> dict[str, Any]:
        arguments: dict[str, Any] = {self._kb_field: self.config.knowledge_base}
        if idempotency_key:
            arguments["idempotency_key"] = idempotency_key
        if submission_id:
            arguments["submission_id"] = submission_id
        outcome = self.client.call(
            self.capabilities.tool("ingest_status"), arguments, idempotent=True, stage="ingest"
        )
        return _as_mapping(outcome.result.payload() if outcome.result else {})

    # -- snapshots ---------------------------------------------------------
    def seal_snapshot(self, *, label: str, note: str | None = None) -> dict[str, Any]:
        if not self.capabilities.has("snapshot"):
            return {"skipped": "no snapshot capability configured"}
        outcome = self.client.call(
            self.capabilities.tool("snapshot"),
            {
                self._kb_field: self.config.knowledge_base,
                "label": label,
                "sealed_by": "pheasant-swarm-lab",
                "note": note,
            },
            idempotent=True,
            stage="index",
        )
        return _as_mapping(outcome.result.payload() if outcome.result else {})

    def get_snapshot(self, snapshot_id: str) -> dict[str, Any]:
        if not self.capabilities.has("snapshot_get"):
            return {"skipped": "no snapshot_get capability configured"}
        outcome = self.client.call(
            self.capabilities.tool("snapshot_get"),
            {self._kb_field: self.config.knowledge_base, "snapshot_id": snapshot_id},
            idempotent=True,
            stage="index",
        )
        return _as_mapping(outcome.result.payload() if outcome.result else {})


def _as_mapping(payload: Any) -> dict[str, Any]:
    return dict(payload) if isinstance(payload, Mapping) else {"raw": payload}


def build_requests(
    items: Iterable[Mapping[str, Any]],
    *,
    run_id: str,
    topic_id: str,
    namespace: str,
) -> list[IngestRequest]:
    """Build ingest requests from normalised source records."""

    requests: list[IngestRequest] = []
    for item in items:
        requests.append(
            IngestRequest(
                run_id=run_id,
                topic_id=topic_id,
                source_id=str(item["source_id"]),
                namespace=namespace,
                source=IngestSource(
                    title=str(item.get("title", "")),
                    stable_identifier=item.get("stable_identifier"),
                    canonical_url=item.get("canonical_url"),
                    authors=list(item.get("authors", []) or []),
                    published_at=item.get("published_at"),
                    license=str(item.get("license", "unknown")),
                ),
                payload=IngestPayload(
                    media_type=str(item.get("media_type", "text/markdown")),
                    text=str(item.get("text", "")),
                    bytes=len(str(item.get("text", "")).encode("utf-8")) or None,
                    local_artifact_ref=item.get("local_artifact_ref"),
                ),
                provenance=IngestProvenance(
                    discovery_event_id=item.get("discovery_event_id"),
                    acquisition_event_id=item.get("acquisition_event_id"),
                    researcher_agent_id=item.get("researcher_agent_id"),
                ),
                operation=str(item.get("operation", "upload")),
            )
        )
    return requests
