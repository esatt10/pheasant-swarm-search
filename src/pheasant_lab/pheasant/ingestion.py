"""The ingest adapter.

The lab normalises every literature item into one internal shape before the
adapter maps it onto whatever this server's ingest tool is actually called and
whatever its parameters are actually named. That indirection is the point: a
server rename is a config edit here, not a code change, and the run manifest
records which spelling was used.
"""

from __future__ import annotations

import base64
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import ids
from ..hashing import digest, digest_file, digest_text
from ..settings import PheasantFile
from . import protocol
from .capabilities import CapabilityMap
from .client import PheasantClient
from .protocol import McpToolError
from .receipts import IngestReceipt, ReceiptLedger, parse_receipts

MEDIA_TYPES = ("text/plain", "text/markdown", "application/pdf", "metadata-only")


class RegistrationRefused(RuntimeError):
    """The region stored a submission and will not index where it put it."""


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
        if self.payload.local_artifact_ref:
            return digest_file(Path(self.payload.local_artifact_ref))
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
        extension = ".pdf" if self.payload.media_type == "application/pdf" else ".md"
        return f"{self.topic_id}/{slug}{extension}"

    def as_document(self, argument_map: Mapping[str, str]) -> dict[str, Any]:
        text = self.payload.text
        if argument_map.get("content_encoding"):
            content = (
                Path(self.payload.local_artifact_ref).read_bytes()
                if self.payload.local_artifact_ref
                else text.encode("utf-8")
            )
            text = base64.b64encode(content).decode("ascii")
        elif self.payload.local_artifact_ref:
            raise ValueError("original files require a configured ingest content_encoding argument")
        return self._document(text, argument_map)

    def encoded_document_size(self, argument_map: Mapping[str, str]) -> int:
        """Exact base64 JSON size, without reading and allocating the file payload."""
        if not argument_map.get("content_encoding"):
            return protocol.encoded_json_size(self.as_document(argument_map))
        raw_bytes = (
            Path(self.payload.local_artifact_ref).stat().st_size
            if self.payload.local_artifact_ref
            else len(self.payload.text.encode("utf-8"))
        )
        # Base64 is ASCII and needs no JSON escaping. The empty string's quotes
        # are already included in the document skeleton's encoded size.
        return protocol.encoded_json_size(self._document("", argument_map)) + 4 * (
            (raw_bytes + 2) // 3
        )

    def _document(self, text: str, argument_map: Mapping[str, str]) -> dict[str, Any]:
        return {
            argument_map.get("document_path", "relative_path"): self.path(),
            argument_map.get("document_text", "text"): text,
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
        self._argument_map: dict[str, str] = dict(config.argument_map.get("ingest") or {})
        self._kb_field = str(config.argument_map.get("knowledge_base_field", "knowledge_base"))
        #: Where the region says it put the submitted bytes. Pheasant lands a
        #: submission in a directory and leaves registering that directory as
        #: a source to the caller, so this is what `register` points it at.
        self.landing_directory: str | None = None
        self._registered = False
        self._clock = time

    # -- submission --------------------------------------------------------
    def submit(
        self, requests: Sequence[IngestRequest], *, batch_size: int = 20
    ) -> list[IngestReceipt]:
        """Bound both document count and the entire encoded request's bytes."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        tool = self.capabilities.tool("ingest")
        receipts: list[IngestReceipt] = []
        limit = self.config.ingest_max_request_bytes
        # Reserve 20 digits for the client's monotonic JSON-RPC request id.
        # The batch's submission id has a fixed length regardless of its keys.
        envelope_bytes = protocol.encoded_json_size(
            protocol.request(
                protocol.METHOD_TOOLS_CALL,
                {
                    "name": tool,
                    "arguments": self._submission_arguments([], "submission-" + "0" * 16),
                },
                2**64 - 1,
            )
        )
        chunk: list[IngestRequest] = []
        documents: list[dict[str, Any]] = []
        request_bytes = envelope_bytes
        for request in requests:
            document_bytes = request.encoded_document_size(self._argument_map)
            added_bytes = document_bytes + (2 if chunk else 0)  # JSON list separator: ", "
            if chunk and (len(chunk) >= batch_size or request_bytes + added_bytes > limit):
                receipts.extend(self._submit_batch(tool, chunk, documents, request_bytes))
                chunk, documents = [], []
                request_bytes = envelope_bytes
            if envelope_bytes + document_bytes > limit:
                receipts.append(self._oversized(request, envelope_bytes + document_bytes))
                continue
            # Encode each original file once, only after the previous batch
            # has been flushed, and never allocate an oversized base64 payload.
            document = request.as_document(self._argument_map)
            request_bytes += document_bytes + (2 if chunk else 0)
            chunk.append(request)
            documents.append(document)
        if chunk:
            receipts.extend(self._submit_batch(tool, chunk, documents, request_bytes))
        return receipts

    def _submission_arguments(
        self, documents: list[dict[str, Any]], submission_id: str
    ) -> dict[str, Any]:
        arguments = {
            self._kb_field: self.config.knowledge_base,
            self._argument_map.get("documents", "documents"): documents,
            self._argument_map.get("source_name", "source_name"): self.config.source_name,
            self._argument_map.get("submission_id", "submission_id"): submission_id,
        }
        if self._argument_map.get("content_encoding"):
            arguments[self._argument_map["content_encoding"]] = "base64"
        return arguments

    def _note_request(self, request: IngestRequest) -> None:
        self.ledger.note_submission(request.idempotency_key)
        if self.tracer is not None:
            self.tracer.append("ingest-requests.jsonl", request.as_record())

    def _oversized(self, request: IngestRequest, request_bytes: int) -> IngestReceipt:
        self._note_request(request)
        receipt = IngestReceipt(
            receipt_id=f"receipt-local-size-{request.idempotency_key[-12:]}",
            run_id=self.run_id,
            source_id=request.source_id,
            idempotency_key=request.idempotency_key,
            submission_id=None,  # No request reached the region.
            status="failed",
            content_digest=request.content_digest,
            error_code="REQUEST_TOO_LARGE",
            error_message=(
                f"Document {request.path()} needs {request_bytes} encoded MCP request bytes, "
                f"exceeding ingest_max_request_bytes={self.config.ingest_max_request_bytes}; "
                "not sent. Increase the setting only within the region's whole-request limit."
            ),
            retryable=False,
        )
        self.ledger.record(receipt)
        if self.tracer is not None:
            self.tracer.append("ingest-receipts.jsonl", receipt.as_dict())
        self._emit(
            "ingest.batch",
            "failed",
            source_id=request.source_id,
            request_bytes=request_bytes,
            max_request_bytes=self.config.ingest_max_request_bytes,
            error_code=receipt.error_code,
        )
        return receipt

    def _submit_batch(
        self,
        tool: str,
        chunk: Sequence[IngestRequest],
        documents: list[dict[str, Any]],
        request_bytes: int,
    ) -> list[IngestReceipt]:
        submission_id = (
            "submission-" + digest([r.idempotency_key for r in chunk]).removeprefix("sha256:")[:16]
        )
        key_to_source = {request.idempotency_key: request.source_id for request in chunk}
        digests = {request.idempotency_key: request.content_digest for request in chunk}
        for request in chunk:
            self._note_request(request)
        arguments = self._submission_arguments(documents, submission_id)
        self._emit(
            "ingest.batch",
            "submitted",
            submission_id=submission_id,
            document_count=len(chunk),
            request_bytes=request_bytes,
            max_request_bytes=self.config.ingest_max_request_bytes,
        )
        outcome = self.client.call(
            tool,
            arguments,
            idempotency_key=submission_id,
            stage="ingest",
        )
        payload = outcome.result.payload() if outcome.result else {}
        if isinstance(payload, Mapping) and payload.get("directory"):
            self.landing_directory = str(payload["directory"])
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
    def register(self) -> dict[str, Any]:
        """Make the landing directory a source the region will index.

        Pheasant's `submit_documents` persists bytes and receipts them, and
        stops there: its landing directory becomes searchable only once
        something registers it as an ordinary ``document_folder`` source,
        which is what its own upload route and readiness probe do. Without
        this the first sync is refused with "Unknown source" and nothing the
        swarm collected is ever indexed.

        Registration is an upsert of the same definition, so repeating it is
        harmless; it is done once per process because a resumed run that
        submitted nothing has no directory to name and was registered by the
        run that did.
        """

        if self._registered:
            return {"skipped": "already registered by this process"}
        if not self.capabilities.has("register_source"):
            return {"skipped": "no register_source capability configured"}
        if not self.landing_directory:
            return {"skipped": "nothing was submitted by this process"}
        tool = self.capabilities.tool("register_source")
        try:
            outcome = self.client.call(
                tool,
                {
                    self._kb_field: self.config.knowledge_base,
                    "name": self.config.source_name,
                    "source_type": "document_folder",
                    "path": self.landing_directory,
                    "description": "Documents submitted by pheasant-swarm-lab",
                    # The submitted documents are the corpus. The region's
                    # default include list is code-shaped, and an include that
                    # quietly drops a submission is a silent loss nothing
                    # downstream could attribute.
                    "include": ["**/*"],
                },
                idempotent=True,
                stage="index",
            )
        except McpToolError as exc:
            raise RegistrationRefused(
                f"the region accepted the submission into {self.landing_directory} but refused "
                f"to register that directory as source '{self.config.source_name}': "
                f"{exc.message}\nPheasant only registers paths under an allow-listed root; add "
                "the region's `pheasant.state_path` (e.g. /state) to "
                "`security.allow_workspace_roots` in its pheasant.yaml."
            ) from exc
        self._registered = True
        return _as_mapping(outcome.result.payload() if outcome.result else {})

    def sync(self, *, mode: str = "incremental") -> dict[str, Any]:
        if not self.capabilities.has("sync"):
            return {"skipped": "no sync capability configured"}
        self.register()
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
        payload = _as_mapping(outcome.result.payload() if outcome.result else {})
        self._emit("ingest.sync", sync_disposition(payload), mode=mode, region=payload)
        return payload

    def acknowledge(
        self, submission_id: str | None = None, *, wait_seconds: float | None = None
    ) -> dict[str, Any]:
        """Cross the index barrier from what the region *holds*.

        Not from what a sync reported: a sync's summary is a claim about what
        it did, and this is a question about what the region has.

        Pheasant answers with *counts* (``acknowledged``, ``still_accepted``)
        and keeps the receipts behind ``get_ingest_status``, so the barrier is
        crossed in two steps: ask the region to promote what it now holds,
        then read each submission's receipts back. On a fleet the sync is
        published to an indexer rather than run in the call, so
        ``still_accepted`` can stay above zero for a while; the call polls up
        to ``wait_seconds`` (default: the configured call timeout) and then
        reports what is still outstanding rather than treating it as indexed.
        """

        if not self.capabilities.has("ingest_acknowledge"):
            return {"skipped": "no acknowledge capability configured"}
        arguments: dict[str, Any] = {self._kb_field: self.config.knowledge_base}
        submissions = self._submissions(submission_id)
        if not submissions:
            return {"skipped": "no recorded submission to acknowledge"}
        deadline = self._clock.monotonic() + (
            self.config.timeout_seconds if wait_seconds is None else wait_seconds
        )
        backoff = max(0.05, self.config.retry_backoff_seconds)
        started = self._clock.monotonic()
        polls = 0
        while True:
            polls += 1
            reports = []
            for submission in submissions:
                outcome = self.client.call(
                    self.capabilities.tool("ingest_acknowledge"),
                    {**arguments, "submission_id": submission},
                    idempotent=True,
                    stage="index",
                )
                report = _as_mapping(outcome.result.payload() if outcome.result else {})
                if not isinstance(report.get("still_accepted"), int):
                    raise RuntimeError(
                        "region returned no pending receipt count for this submission"
                    )
                for name in ("receipts", "acknowledged"):
                    if isinstance(report.get(name), list):
                        for row in report[name]:
                            self._fold(row)
                self._refresh(submission)
                reports.append(report)
            # No knowledge-base-wide count belongs to this run's barrier.
            payload = {
                "submission_ids": submissions,
                "acknowledged": sum(
                    r["acknowledged"] for r in reports if isinstance(r.get("acknowledged"), int)
                ),
                "still_accepted": sum(r["still_accepted"] for r in reports),
            }
            pending = payload.get("still_accepted")
            waited = round(self._clock.monotonic() - started, 3)
            if not isinstance(pending, int) or pending <= 0:
                self._emit(
                    "ingest.barrier",
                    "crossed",
                    poll=polls,
                    waited_seconds=waited,
                    acknowledged=payload.get("acknowledged"),
                    still_accepted=pending if isinstance(pending, int) else 0,
                )
                break
            queue = self.index_queue()
            if self._clock.monotonic() >= deadline:
                payload["barrier"] = "timed_out"
                self._emit(
                    "ingest.barrier",
                    "timed_out",
                    poll=polls,
                    waited_seconds=waited,
                    acknowledged=payload.get("acknowledged"),
                    still_accepted=pending,
                    queue=queue,
                )
                break
            self._emit(
                "ingest.barrier",
                "waiting",
                poll=polls,
                waited_seconds=waited,
                acknowledged=payload.get("acknowledged"),
                still_accepted=pending,
                queue=queue,
            )
            self._clock.sleep(min(backoff, self.config.retry_backoff_max_seconds))
            backoff *= 2
        return payload

    def index_queue(self) -> list[dict[str, Any]] | None:
        """This source's outstanding index tasks, or ``None`` when unknowable.

        Only read while the barrier waits, and only when the region offers
        it: it is what turns "still_accepted is 4" into "no indexer has
        claimed the sync yet" — two waits that call for different responses.
        A failure here is an unknown, never a reason to fail the barrier.
        """

        return read_index_queue(
            self.client,
            self.capabilities,
            {self._kb_field: self.config.knowledge_base},
            source=self.config.source_name,
        )

    def _emit(self, event_type: str, disposition: str, **payload: Any) -> None:
        if self.tracer is None:
            return
        status = {
            "failed": "failed",
            "crossed": "succeeded",
            "completed": "succeeded",
            "timed_out": "failed",
            "refused": "failed",
            "consistent": "succeeded",
            "mismatch": "partial",
        }.get(disposition, "partial")
        self.tracer.emit(
            event_type,
            status=status,
            payload={
                "disposition": disposition,
                "source_name": self.config.source_name,
                **{k: v for k, v in payload.items() if v is not None},
            },
        )

    def _refresh(self, submission_id: str | None) -> None:
        """Read this process's receipts back from the region and fold them."""

        if not self.capabilities.has("ingest_status"):
            return
        submissions = self._submissions(submission_id)
        for submission in submissions:
            payload = self.ingest_status(submission_id=submission)
            for row in payload.get("receipts") or []:
                self._fold(row)

    def _submissions(self, submission_id: str | None) -> list[str]:
        return (
            [submission_id]
            if submission_id
            else sorted({r.submission_id for r in self.ledger.receipts if r.submission_id})
        )

    def _fold(self, row: Any) -> None:
        if not isinstance(row, Mapping) or not row.get("idempotency_key"):
            return
        key = str(row["idempotency_key"])
        receipt = self.ledger.get(key)
        if receipt is None:
            return
        status = str(row.get("status") or row.get("disposition") or "indexed")
        if status == receipt.status and (row.get("artifact_id") or None) in (
            None,
            receipt.artifact_id,
        ):
            return
        self.ledger.acknowledge(key, status=status, artifact_id=row.get("artifact_id"))
        if self.tracer is not None:
            # Appended, not rewritten: the trace records that the barrier
            # was crossed, and a reader folds the file by key.
            self.tracer.append("ingest-receipts.jsonl", receipt.as_dict())

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
        submissions = self._submissions(submission_id)
        if not submissions:
            return {"skipped": "no recorded submission to reconcile"}
        reports = []
        for submission in submissions:
            outcome = self.client.call(
                self.capabilities.tool("ingest_reconcile"),
                {**arguments, "submission_id": submission},
                idempotent=True,
                stage="index",
            )
            reports.append(_as_mapping(outcome.result.payload() if outcome.result else {}))
        payload: dict[str, Any] = {"submission_ids": submissions, "scope": "run"}
        for name in (
            "submitted",
            "indexed",
            "rejected",
            "failed",
            "accepted_not_indexed",
            "resubmitted",
            "unaccounted",
            "silent_loss",
        ):
            if all(isinstance(r.get(name), int) for r in reports):
                payload[name] = sum(r[name] for r in reports)
        payload["by_disposition"] = {}
        for report in reports:
            for name, count in (report.get("by_disposition") or {}).items():
                payload["by_disposition"][name] = payload["by_disposition"].get(name, 0) + count
        payload["unaccounted_receipts"] = [
            key for r in reports for key in r.get("unaccounted_receipts", [])
        ]
        from collections import Counter

        artifacts = Counter(r.artifact_id for r in self.ledger.receipts if r.artifact_id)
        payload["duplicated_artifacts"] = sorted(
            key for key, count in artifacts.items() if count > 1
        )
        payload["duplicated"] = len(payload["duplicated_artifacts"])
        payload["reconciled"] = (
            all(r.get("reconciled") is True for r in reports) and not payload["duplicated"]
        )
        return payload

    def inventory(self) -> dict[str, Any] | None:
        """What the region says it holds for this lab's source, beside the receipts.

        pheasant >= 0.13.3 describes a source from its own index
        (``describe_source``): a document count read off the artifacts, not
        off the receipts. Reconcile asks "is every receipt's artifact there";
        this asks the converse - "is everything there something a receipt
        accounts for" - which is how a duplicate or a stray file in the
        landing directory shows up. Observational: ``None`` when the region
        has no such tool or the call fails, an unknown and never a finding.
        """

        if not self.capabilities.has("source_inventory"):
            return None
        try:
            outcome = self.client.call(
                self.capabilities.tool("source_inventory"),
                {
                    self._kb_field: self.config.knowledge_base,
                    "source_name": self.config.source_name,
                },
                idempotent=True,
                stage="index",
            )
        except Exception:
            return None
        if outcome.result is None or outcome.result.is_error:
            return None
        payload = _as_mapping(outcome.result.payload())
        totals = payload.get("totals") if isinstance(payload.get("totals"), Mapping) else {}
        held = totals.get("documents")
        indexed, receipts = self.ledger.index_rate()
        if not isinstance(held, int):
            return None
        disposition = "consistent" if held == indexed else "mismatch"
        summary = {
            "region_documents": held,
            "region_bytes": totals.get("size_bytes"),
            "receipts_indexed": indexed,
            "receipts": receipts,
            "last_indexed_at": (payload.get("source") or {}).get("last_indexed_at"),
            "disposition": disposition,
        }
        self._emit(
            "ingest.inventory",
            disposition,
            **{k: v for k, v in summary.items() if k != "disposition"},
        )
        return summary

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


def sync_disposition(payload: Mapping[str, Any]) -> str:
    """What a sync call actually did, in pheasant's own words.

    Only a result carrying counts ran here. On a fleet the call *publishes*
    (``queued``) and an indexer claims it later; ``already_syncing`` means
    another job holds the source. Treating either as "indexed" is how a
    barrier gets blamed for a wait the region announced up front.
    """

    status = str(payload.get("status") or "").lower()
    if status in {"queued", "already_syncing", "syncing"}:
        return status
    if "skipped" in payload:
        return "skipped"
    return "completed"


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
                    bytes=(
                        Path(item["local_artifact_ref"]).stat().st_size
                        if item.get("local_artifact_ref")
                        else len(str(item.get("text", "")).encode("utf-8"))
                    )
                    or None,
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


def read_index_queue(
    client: Any,
    capabilities: Any,
    arguments: Mapping[str, Any],
    *,
    source: str,
) -> list[dict[str, Any]] | None:
    """``source``'s outstanding index tasks, or ``None`` when unknowable.

    ``None`` covers a region without the tool, a failed call and a backend
    that can count but not list (``listing: "unavailable"``): each is an
    unknown, and an empty list would read as "nothing outstanding".
    """

    if not capabilities.has("index_queue"):
        return None
    try:
        outcome = client.call(
            capabilities.tool("index_queue"), dict(arguments), idempotent=True, stage="index"
        )
    except Exception:
        return None
    payload = _as_mapping(outcome.result.payload() if outcome.result else {})
    if payload.get("listing") != "complete":
        return None
    return [
        {
            key: task.get(key)
            for key in ("task_id", "state", "waiting_seconds", "claimed_by", "position")
        }
        for task in payload.get("tasks") or []
        if isinstance(task, dict) and task.get("source") == source
    ]
