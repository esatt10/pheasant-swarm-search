"""An in-process Pheasant-shaped MCP server.

This exists so the whole lab - handshake, capability resolution, ingestion,
the index barrier, retrieval, memory, snapshots and every metric downstream -
can be exercised offline and deterministically. It implements the MCP methods
the client uses and the Pheasant tools the capability map names, with a real
BM25 index behind ``search_context`` rather than a canned result list.

It is **not** a Pheasant substitute and nothing here should be read as one:
its ranking is a plain BM25 with a memory-steering overlay, it has no graph
arm and no embeddings, and a lab result produced against it measures this
file. Its job is to make the *plumbing* testable; the science needs the real
region, which is why ``doctor`` refuses to treat ``mock`` as a live target.

**Its wire shapes follow pheasant's, not this lab's wishes.** Every response
here was checked against a running pheasant (0.12.16, 0.13.0, 0.13.1, which
added ``expand``, 0.13.2, which added ``get_index_queue``, and 0.13.4, whose
``describe_source`` arrived in 0.13.3): receipts split into
``accepted``/``rejected`` lists with a ``disposition``, acknowledgement as
counts, a submission landing in a directory that must be registered as a
source before ``sync_source`` will index it, hits carrying a capped preview
rather than the passage, and ``get_file_summary`` keyed by path. A mock that
is kinder than the server hides exactly the adapter bugs it exists to catch,
which is how every one of those shapes once went unnoticed here. With
``claim_seconds`` above zero it is a role-split fleet: a sync - the memory
source's included - is published, and nothing it covers is searchable until
the simulated indexer has run it.

Failure injection is first-class (``FaultPlan``) because the error contract is
a thing this repository must test, and an error path nothing exercises is an
error path nobody has.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
import uuid
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..hashing import digest, digest_text
from ..lifecycle import isonow
from . import protocol

TOKEN = re.compile(r"[a-z0-9]+")
K1 = 1.5
B = 0.75
PREFER_BONUS = 0.4
PREFERENCE = re.compile(
    r"\bwhen\s*:\s*(?P<when>[^\n]*?)\s*(?:->|→|=>)\s*prefer\s*:\s*(?P<prefer>[^\n]+)", re.I
)


def tokenize(text: str) -> list[str]:
    return TOKEN.findall((text or "").lower())


@dataclass
class StoredDocument:
    artifact_id: str
    relative_path: str
    text: str
    metadata: dict[str, Any]
    content_digest: str
    indexed: bool = False
    source_name: str = "submissions"

    @property
    def tokens(self) -> list[str]:
        return tokenize(self.text)


MEMORY_SOURCE = "agent-memory"


@dataclass
class MemoryRecord:
    record_id: str
    text: str
    scope: str
    kind: str
    subject: str | None
    principal: str | None
    asserted_at: str
    supersedes: str | None = None
    superseded_by: str | None = None
    valid_until: str | None = None
    #: A fleet publishes a memory record's sync rather than running it, so the
    #: record exists before it is searchable (see `_tool_memory_write`).
    indexed: bool = True

    @property
    def current(self) -> bool:
        return self.superseded_by is None


@dataclass
class FaultPlan:
    """Deterministic failure injection, keyed by tool name.

    ``fail_first`` fails a tool's first *n* calls and then succeeds, which is
    the shape a retry policy has to get right: retrying is only correct when
    the operation is idempotent or carries a key.
    """

    fail_first: dict[str, int] = field(default_factory=dict)
    fail_with: dict[str, str] = field(default_factory=dict)
    always_fail: set[str] = field(default_factory=set)
    partial: set[str] = field(default_factory=set)
    malformed: set[str] = field(default_factory=set)
    latency_ms: dict[str, float] = field(default_factory=dict)
    _seen: Counter = field(default_factory=Counter)

    def check(self, tool: str) -> str | None:
        self._seen[tool] += 1
        if tool in self.always_fail:
            return self.fail_with.get(tool, "internal")
        budget = self.fail_first.get(tool, 0)
        if self._seen[tool] <= budget:
            return self.fail_with.get(tool, "timeout")
        return None


class MockPheasantServer:
    """A Pheasant-shaped MCP endpoint, in this process."""

    SERVER_NAME = "pheasant-mock"
    SERVER_VERSION = "0.12-mock"

    def __init__(
        self,
        *,
        knowledge_base: str = "pheasant-lab",
        protocol_version: str = "2026-07-28",
        faults: FaultPlan | None = None,
        auto_index: bool = False,
        tool_names: Mapping[str, str] | None = None,
        state_path: str | Path | None = None,
        claim_seconds: float = 0.0,
        keep_indexed_on_move: bool = False,
    ) -> None:
        self.knowledge_base = knowledge_base
        #: How a resubmitted key that lands somewhere new folds onto its
        #: receipt. pheasant >= 0.14.0 sends it back to `accepted`; before
        #: that the receipt stayed `indexed` with nothing indexed behind it,
        #: and the lab still supports those regions.
        self.keep_indexed_on_move = keep_indexed_on_move
        self.protocol_version = protocol_version
        self.faults = faults or FaultPlan()
        self.auto_index = auto_index
        self.documents: dict[str, StoredDocument] = {}
        self.receipts: dict[str, dict[str, Any]] = {}
        self.submissions: dict[str, list[str]] = {}
        self.memory: dict[str, MemoryRecord] = {}
        self.snapshots: dict[str, dict[str, Any]] = {}
        #: Registered sources, name -> path. A submission is not searchable
        #: until its landing directory is one of these and has been synced.
        self.sources: dict[str, str] = {}
        self.evidence: list[dict[str, Any]] = []
        self.call_log: list[tuple[str, dict[str, Any]]] = []
        self.initialized = False
        self._tool_names = dict(tool_names or {})
        self._generation = 0
        #: Above zero, behave like a role-split fleet: `sync_source` publishes
        #: a task instead of indexing, an indexer claims it after this many
        #: seconds and finishes it after as many again. Wall-clock times, so
        #: the interval survives the per-command reconnect a run makes.
        self.claim_seconds = max(0.0, float(claim_seconds))
        self.queue: dict[str, dict[str, Any]] = {}
        # A region outlives the process that talks to it. Without this, each
        # CLI command would connect to an empty region and the index barrier
        # - the thing this lab exists to measure - would never be crossed.
        self.state_path = Path(state_path) if state_path else None
        # Research branches run concurrently against one region, so this is
        # shared mutable state like any other server's.
        self._lock = threading.RLock()
        self._load()

    # -- persistence -------------------------------------------------------
    def _load(self) -> None:
        if self.state_path is None or not self.state_path.is_file():
            return
        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.knowledge_base = str(payload.get("knowledge_base", self.knowledge_base))
        self._generation = int(payload.get("generation", 0))
        self.documents = {
            key: StoredDocument(
                artifact_id=row["artifact_id"],
                relative_path=row["relative_path"],
                text=row["text"],
                metadata=dict(row.get("metadata") or {}),
                content_digest=row["content_digest"],
                indexed=bool(row.get("indexed")),
                source_name=str(row.get("source_name") or "submissions"),
            )
            for key, row in (payload.get("documents") or {}).items()
        }
        self.receipts = dict(payload.get("receipts") or {})
        self.submissions = {k: list(v) for k, v in (payload.get("submissions") or {}).items()}
        self.snapshots = dict(payload.get("snapshots") or {})
        self.sources = dict(payload.get("sources") or {})
        self.queue = {str(k): dict(v) for k, v in (payload.get("queue") or {}).items()}
        self.evidence = list(payload.get("evidence") or [])
        self.memory = {
            key: MemoryRecord(**row) for key, row in (payload.get("memory") or {}).items()
        }

    def _save(self) -> None:
        if self.state_path is None:
            return
        payload = {
            "knowledge_base": self.knowledge_base,
            "generation": self._generation,
            "documents": {
                key: {
                    "artifact_id": doc.artifact_id,
                    "relative_path": doc.relative_path,
                    "text": doc.text,
                    "metadata": doc.metadata,
                    "content_digest": doc.content_digest,
                    "indexed": doc.indexed,
                    "source_name": doc.source_name,
                }
                for key, doc in self.documents.items()
            },
            "receipts": self.receipts,
            "submissions": self.submissions,
            "snapshots": self.snapshots,
            "sources": self.sources,
            "queue": self.queue,
            "evidence": self.evidence,
            "memory": {key: record.__dict__ for key, record in self.memory.items()},
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        # Unique per write, not per process: a fixed temp name is a collision
        # waiting for a second writer, and two threads racing on one would see
        # the first rename take the file out from under the second.
        temp = self.state_path.with_suffix(f".{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
        try:
            temp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            os.replace(temp, self.state_path)
        finally:
            if temp.exists():
                temp.unlink(missing_ok=True)

    # -- MCP surface -------------------------------------------------------
    def handle(self, message: Mapping[str, Any]) -> dict[str, Any] | None:
        with self._lock:
            return self._handle(message)

    def _handle(self, message: Mapping[str, Any]) -> dict[str, Any] | None:
        method = message.get("method")
        if method is None:
            raise ValueError("not a request or notification")
        if "id" not in message:
            if method == protocol.NOTIFICATION_INITIALIZED:
                self.initialized = True
            return None
        request_id = message["id"]
        params = message.get("params") or {}
        try:
            if method == protocol.METHOD_INITIALIZE:
                return self._ok(request_id, self._initialize(params))
            if method == protocol.METHOD_PING:
                return self._ok(request_id, {})
            if method == protocol.METHOD_TOOLS_LIST:
                return self._ok(request_id, {"tools": self._tools()})
            if method == protocol.METHOD_TOOLS_CALL:
                return self._ok(request_id, self._call(params))
            return self._error(request_id, protocol.METHOD_NOT_FOUND, f"unknown method '{method}'")
        except _InjectedFailure as failure:
            raise failure.to_exception() from None

    def _initialize(self, params: Mapping[str, Any]) -> dict[str, Any]:
        requested = str(params.get("protocolVersion") or self.protocol_version)
        # Negotiate down, never up: answering with a newer revision than the
        # client asked for is how a client ends up sent a shape it cannot read.
        negotiated = requested if requested <= self.protocol_version else self.protocol_version
        return {
            "protocolVersion": negotiated,
            "capabilities": {"tools": {"listChanged": False}, "resources": {}},
            "serverInfo": {"name": self.SERVER_NAME, "version": self.SERVER_VERSION},
            "instructions": "Mock region. Search is BM25 over submitted documents.",
        }

    def name_for(self, canonical: str) -> str:
        return self._tool_names.get(canonical, canonical)

    def _tools(self) -> list[dict[str, Any]]:
        def tool(
            name: str, description: str, properties: dict[str, Any], required: list[str]
        ) -> dict[str, Any]:
            return {
                "name": self.name_for(name),
                "description": description,
                "inputSchema": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            }

        kb = {"knowledge_base": {"type": "string"}}
        return [
            tool("list_knowledge_bases", "Registered knowledge bases and status.", {}, []),
            tool(
                "submit_documents",
                "Persist documents with an idempotency key and one receipt per item.",
                {
                    **kb,
                    "documents": {"type": "array", "items": {"type": "object"}},
                    "source_name": {"type": "string"},
                    "submission_id": {"type": "string"},
                    "agent_id": {"type": "string"},
                },
                ["knowledge_base", "documents"],
            ),
            tool(
                "get_ingest_status",
                "Receipts per submitted item.",
                {**kb, "idempotency_key": {"type": "string"}, "submission_id": {"type": "string"}},
                ["knowledge_base"],
            ),
            tool(
                "acknowledge_ingest",
                "Cross the index barrier for receipts whose artifacts now exist.",
                {**kb, "submission_id": {"type": "string"}},
                ["knowledge_base"],
            ),
            tool(
                "reconcile_ingest",
                "Submitted against held; silent_loss is the number to read.",
                {**kb, "submission_id": {"type": "string"}},
                ["knowledge_base"],
            ),
            tool(
                "register_source",
                "Register a source: a folder/file path, or web pages by URL.",
                {
                    **kb,
                    "name": {"type": "string"},
                    "source_type": {"type": "string"},
                    "path": {"type": "string"},
                    "description": {"type": "string"},
                    "enabled": {"type": "boolean"},
                    "include": {"type": "array", "items": {"type": "string"}},
                    "exclude": {"type": "array", "items": {"type": "string"}},
                    "sync_now": {"type": "boolean"},
                },
                ["knowledge_base", "name", "source_type"],
            ),
            tool(
                "sync_source",
                "Trigger one source sync.",
                {**kb, "source_name": {"type": "string"}, "mode": {"type": "string"}},
                ["knowledge_base", "source_name"],
            ),
            tool(
                "search_context",
                "Search indexed context and return compact results with provenance.",
                {
                    **kb,
                    "query": {"type": "string"},
                    "mode": {"type": "string"},
                    "max_results": {"type": "integer"},
                    "memory": {"type": ["object", "string"]},
                    "session": {"type": "string"},
                    "principal": {"type": "string"},
                    "source_name": {"type": "string"},
                    "exclude_sources": {"type": "array", "items": {"type": "string"}},
                    "node_types": {"type": "array", "items": {"type": "string"}},
                    "min_score": {"type": "number"},
                    "as_of": {"type": "string"},
                    "snapshot_id": {"type": "string"},
                    "expand": {"type": ["object", "integer", "boolean", "null"]},
                },
                ["knowledge_base", "query"],
            ),
            tool(
                "get_file_summary",
                "Return summary and provenance for one indexed file.",
                {**kb, "path": {"type": "string"}, "source_name": {"type": "string"}},
                ["knowledge_base", "path"],
            ),
            tool(
                "memory_write",
                "Append one agent-memory record and index it.",
                {
                    **kb,
                    "text": {"type": "string"},
                    "scope": {"type": "string"},
                    "kind": {"type": "string"},
                    "subject": {"type": "string"},
                    "supersedes": {"type": "string"},
                    "principal": {"type": "string"},
                    "session": {"type": "string"},
                    "sync": {"type": "boolean"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "valid_until": {"type": "string"},
                },
                ["knowledge_base", "text"],
            ),
            tool(
                "seal_snapshot",
                "Seal the current state as a run's reference snapshot.",
                {
                    **kb,
                    "label": {"type": "string"},
                    "sealed_by": {"type": "string"},
                    "note": {"type": "string"},
                },
                ["knowledge_base"],
            ),
            tool(
                "get_snapshot",
                "A snapshot's manifest, and whether the region still stands there.",
                {**kb, "snapshot_id": {"type": "string"}},
                ["knowledge_base", "snapshot_id"],
            ),
            tool("describe_retrieval", "How this region retrieves.", kb, ["knowledge_base"]),
            tool(
                "get_index_queue",
                "Outstanding index tasks and whether an indexer has claimed each one.",
                {**kb, "limit": {"type": "integer"}},
                ["knowledge_base"],
            ),
            tool(
                "describe_source",
                "One source in detail: what it holds and what it links to.",
                {
                    **kb,
                    "source_name": {"type": "string"},
                    "principal": {"type": "string"},
                    "principal_groups": {"type": "array", "items": {"type": "string"}},
                },
                ["knowledge_base", "source_name"],
            ),
            tool(
                "record_evidence",
                "Record what came of a result this region returned.",
                {
                    **kb,
                    "query": {"type": "string"},
                    "target_id": {"type": "string"},
                    "event_type": {"type": "string"},
                    "target_type": {"type": "string"},
                    "principal": {"type": "string"},
                    "session_id": {"type": "string"},
                    "position": {"type": "integer"},
                    "outcome_reference": {"type": "string"},
                },
                ["knowledge_base", "query", "target_id", "event_type"],
            ),
        ]

    # -- dispatch ----------------------------------------------------------
    def _call(self, params: Mapping[str, Any]) -> dict[str, Any]:
        raw_name = str(params.get("name") or "")
        canonical = next((c for c, n in self._tool_names.items() if n == raw_name), raw_name)
        arguments = dict(params.get("arguments") or {})
        self.call_log.append((canonical, arguments))

        delay = self.faults.latency_ms.get(canonical)
        if delay:
            time.sleep(delay / 1000.0)

        failure = self.faults.check(canonical)
        if failure:
            raise _InjectedFailure(failure, canonical)
        if canonical in self.faults.malformed:
            return {"content": [{"type": "text", "text": "{not json"}], "isError": False}

        self._advance_queue()
        handler = getattr(self, f"_tool_{canonical}", None)
        if handler is None:
            return self._tool_error(f"Unknown tool: {raw_name}")
        kb = arguments.get("knowledge_base")
        if kb is not None and kb != self.knowledge_base:
            return self._tool_error(f"Unknown knowledge base: {kb}")
        try:
            payload = handler(arguments)
        except _Refusal as refusal:
            return self._tool_error(str(refusal))
        if canonical in {
            "submit_documents",
            "acknowledge_ingest",
            "sync_source",
            "register_source",
            "memory_write",
            "seal_snapshot",
            "record_evidence",
        }:
            self._save()
        partial = canonical in self.faults.partial
        if partial and isinstance(payload, dict):
            payload = {
                **payload,
                "partial": True,
                "warnings": ["result set truncated by the region"],
            }
        return {"content": [{"type": "text", "text": json.dumps(payload)}], "isError": False}

    # -- tools -------------------------------------------------------------
    def _tool_list_knowledge_bases(self, _: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "knowledge_bases": [
                {
                    "kb_id": self.knowledge_base,
                    "status": "ready",
                    "artifacts": len(self.documents),
                    "indexed": sum(1 for d in self.documents.values() if d.indexed),
                    "generation_id": self._generation_id(),
                }
            ]
        }

    def _tool_submit_documents(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        submission_id = str(
            arguments.get("submission_id") or "submission-" + digest(arguments)[7:23]
        )
        source_name = str(arguments.get("source_name") or "submissions")
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        keys: list[str] = []
        for entry in arguments.get("documents") or []:
            if not isinstance(entry, Mapping):
                continue
            path = str(entry.get("relative_path") or "")
            text = str(entry.get("text") or "")
            key = str(entry.get("idempotency_key") or digest_text(path + text))
            metadata = dict(entry.get("metadata") or {})
            keys.append(key)

            existing = self.receipts.get(key)
            if existing is not None:
                # One key, one receipt: the retry folds onto it and bumps the
                # region's own count of submissions. Like pheasant, the bytes
                # are written again at the recorded path under *this* call's
                # source and the receipt moves there. A receipt that now names
                # another artifact or other bytes is `accepted` again
                # (pheasant >= 0.14.0), unless this mock plays an older region.
                existing["submissions"] = int(existing.get("submissions", 1)) + 1
                if existing["disposition"] != "rejected":
                    path = str((existing.get("detail") or {}).get("relative_path") or path)
                    artifact_id = _artifact_id(source_name, path)
                    sha = digest_text(text).removeprefix("sha256:")
                    moved = (artifact_id, sha) != (
                        existing.get("artifact_id"),
                        existing.get("content_sha256"),
                    )
                    if moved and not self.keep_indexed_on_move:
                        existing["disposition"] = "accepted"
                        existing["indexed_at"] = None
                        existing["content_sha256"] = sha
                    held = self.documents.get(artifact_id)
                    self.documents[artifact_id] = StoredDocument(
                        artifact_id=artifact_id,
                        relative_path=path,
                        text=text,
                        metadata=metadata,
                        content_digest=digest_text(text),
                        indexed=held.indexed if held is not None else self.auto_index,
                        source_name=source_name,
                    )
                    existing["source_name"] = source_name
                    existing["artifact_id"] = artifact_id
                (rejected if existing["disposition"] == "rejected" else accepted).append(
                    dict(existing)
                )
                continue
            sha = digest_text(text).removeprefix("sha256:")
            if not path or not text:
                receipt = self._receipt(
                    key,
                    submission_id,
                    source_name,
                    disposition="rejected",
                    content_sha256=sha,
                    error_code="INVALID_REQUEST",
                    detail={"reason": "a document needs a relative_path and text", **metadata},
                )
                self.receipts[key] = receipt
                rejected.append(dict(receipt))
                continue

            artifact_id = _artifact_id(source_name, path)
            self.documents[artifact_id] = StoredDocument(
                artifact_id=artifact_id,
                relative_path=path,
                text=text,
                metadata=metadata,
                content_digest=digest_text(text),
                indexed=self.auto_index,
                source_name=source_name,
            )
            receipt = self._receipt(
                key,
                submission_id,
                source_name,
                disposition="accepted",
                content_sha256=sha,
                artifact_id=artifact_id,
                detail={"relative_path": path, **metadata},
            )
            self.receipts[key] = receipt
            accepted.append(dict(receipt))
        self.submissions[submission_id] = keys
        self._generation += 1
        return {
            "knowledge_base": self.knowledge_base,
            "submission_id": submission_id,
            "source_name": source_name,
            "directory": self._directory(source_name),
            "submitted": len(keys),
            "accepted": accepted,
            "rejected": rejected,
            "indexed": 0,
            "next": "sync the source, then call ingest_acknowledge to cross the index barrier",
        }

    def _receipt(
        self,
        key: str,
        submission_id: str,
        source_name: str,
        *,
        disposition: str,
        content_sha256: str,
        artifact_id: str | None = None,
        error_code: str | None = None,
        detail: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = isonow()
        return {
            "receipt_id": digest({"k": key, "s": submission_id})[7:39],
            "idempotency_key": key,
            "submission_id": submission_id,
            "source_name": source_name,
            "disposition": disposition,
            "submitted_at": now,
            "updated_at": now,
            "indexed_at": None,
            "artifact_id": artifact_id,
            "content_sha256": content_sha256,
            "chunk_count": None,
            "submissions": 1,
            "error_code": error_code,
            "retryable": False,
            "detail": dict(detail or {}),
        }

    def _directory(self, source_name: str) -> str:
        return f"/state/uploads/{source_name}"

    def _keys_for(self, submission: Any) -> list[str]:
        if submission:
            return list(self.submissions.get(str(submission), []))
        return list(self.receipts)

    def _tool_get_ingest_status(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        key = arguments.get("idempotency_key")
        if key:
            receipt = self.receipts.get(str(key))
            return {
                "knowledge_base": self.knowledge_base,
                "receipts": [dict(receipt)] if receipt else [],
                "found": receipt is not None,
            }
        keys = self._keys_for(arguments.get("submission_id"))
        receipts = [dict(self.receipts[k]) for k in keys if k in self.receipts]
        return {
            "knowledge_base": self.knowledge_base,
            "submission_id": arguments.get("submission_id"),
            "receipts": receipts,
            "found": bool(receipts),
        }

    def _tool_acknowledge_ingest(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """Counts, not receipts - the receipts are behind get_ingest_status."""

        pending = [
            self.receipts[k]
            for k in self._keys_for(arguments.get("submission_id"))
            if k in self.receipts and self.receipts[k]["disposition"] == "accepted"
        ]
        crossed = 0
        for receipt in pending:
            artifact = self.documents.get(str(receipt.get("artifact_id")))
            if artifact is None or not artifact.indexed:
                continue
            receipt["disposition"] = "indexed"
            receipt["indexed_at"] = receipt["updated_at"] = isonow()
            receipt["chunk_count"] = 1
            crossed += 1
        return {
            "knowledge_base": self.knowledge_base,
            "submission_id": arguments.get("submission_id"),
            "acknowledged": crossed,
            "still_accepted": len(pending) - crossed,
        }

    def _tool_reconcile_ingest(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        keys = [k for k in self._keys_for(arguments.get("submission_id")) if k in self.receipts]
        by_disposition = Counter(self.receipts[k]["disposition"] for k in keys)
        # Not a difference between two totals: two totals can agree while one
        # item was lost and another double-written. A receipt that says
        # `indexed` and names an artifact the region does not hold is the loss.
        unaccounted = [
            self.receipts[k]["receipt_id"]
            for k in keys
            if self.receipts[k]["disposition"] == "indexed"
            and self.receipts[k].get("artifact_id")
            and str(self.receipts[k]["artifact_id"]) not in self.documents
        ]
        return {
            "kb_id": self.knowledge_base,
            "submission_id": arguments.get("submission_id"),
            "submitted": len(keys),
            "by_disposition": dict(by_disposition),
            "indexed": by_disposition.get("indexed", 0),
            "rejected": by_disposition.get("rejected", 0),
            "failed": by_disposition.get("failed", 0),
            "accepted_not_indexed": by_disposition.get("accepted", 0),
            "unaccounted": len(unaccounted),
            "unaccounted_receipts": unaccounted,
            "silent_loss": len(unaccounted),
            "duplicated": 0,
            "reconciled": not unaccounted,
        }

    def _tool_register_source(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        name = str(arguments.get("name") or "")
        if not name:
            raise _Refusal("register_source needs a name")
        path = str(arguments.get("path") or "")
        self.sources[name] = path
        return {
            "status": "registered",
            "knowledge_base": self.knowledge_base,
            "source": {"name": name, "type": arguments.get("source_type"), "path": path},
        }

    def _tool_sync_source(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        name = str(arguments.get("source_name") or "")
        if name not in self.sources:
            # Pheasant's own refusal: submitting documents does not register
            # the directory they land in.
            raise _Refusal(f"Unknown source: {name}")
        mode = str(arguments.get("mode") or "incremental")
        if self.claim_seconds > 0:
            return self._publish(name, mode)
        newly = 0
        for document in self.documents.values():
            if document.source_name == name and not document.indexed:
                document.indexed = True
                newly += 1
        self._generation += 1
        return {
            "source_id": name,
            "mode": str(arguments.get("mode") or "incremental"),
            "indexed_artifacts": newly,
            "skipped_artifacts": sum(1 for d in self.documents.values() if d.source_name == name)
            - newly,
            "status": "healthy",
        }

    def _publish(self, name: str, mode: str = "incremental") -> dict[str, Any]:
        """pheasant's fleet answer (`PheasantTools._publish_sync`).

        The task id is content-addressed, so a repeat while one is outstanding
        is the same task rather than a second one.
        """

        task_id = (
            "idx-"
            + hashlib.sha256(f"{self.knowledge_base}\0{name}\0{mode}".encode()).hexdigest()[:24]
        )
        if task_id not in self.queue:
            now = time.time()
            self.queue[task_id] = {
                "source": name,
                "mode": mode,
                "enqueued": now,
                "claim_at": now + self.claim_seconds,
                "done_at": now + 2 * self.claim_seconds,
            }
        return {"source_id": name, "status": "queued", "task_id": task_id}

    def _advance_queue(self) -> None:
        """Let simulated indexers finish what they claimed, by the clock."""

        if not self.queue:
            return
        now = time.time()
        with self._lock:
            for task_id, task in list(self.queue.items()):
                if now < float(task["done_at"]):
                    continue
                for document in self.documents.values():
                    if document.source_name == task["source"]:
                        document.indexed = True
                if task["source"] == MEMORY_SOURCE:
                    for record in self.memory.values():
                        record.indexed = True
                self._generation += 1
                del self.queue[task_id]

    def _tool_get_index_queue(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """pheasant's `get_index_queue` shape (pheasant `services/index_queue.py`)."""

        now = time.time()
        tasks: list[dict[str, Any]] = []
        position = 0
        for task_id, task in sorted(self.queue.items(), key=lambda kv: kv[1]["enqueued"]):
            claimed = now >= float(task["claim_at"])
            if not claimed:
                position += 1
            enqueued = datetime.fromtimestamp(float(task["enqueued"]), tz=UTC).isoformat()
            tasks.append(
                {
                    "task_id": task_id,
                    "enqueued_at": enqueued,
                    "visible_at": enqueued,
                    "source": task["source"],
                    "mode": task["mode"],
                    "state": "claimed" if claimed else "awaiting_claim",
                    "waiting_seconds": round(now - float(task["enqueued"]), 1),
                    "claimed_by": "mock-indexer" if claimed else None,
                    "attempts": 0,
                    "max_attempts": 3,
                    "last_error": None,
                    "position": None if claimed else position,
                }
            )
        counts = dict.fromkeys(
            ("awaiting_claim", "retry_scheduled", "claimed", "claim_lapsed", "dead"), 0
        )
        for row in tasks:
            counts[row["state"]] += 1
        return {
            "knowledge_base": self.knowledge_base,
            "enabled": self.claim_seconds > 0,
            "backend": "local" if self.claim_seconds > 0 else None,
            "listing": "complete" if self.claim_seconds > 0 else "not_applicable",
            "depth": None,
            "tasks": tasks,
            "counts": counts,
        }

    def _tool_describe_source(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """pheasant's `describe_source` shape (0.13.3, `services/inventory_detail.py`).

        Counts what is *indexed*, like the region: a document still waiting
        for an indexer is not yet something the source holds.
        """

        name = str(arguments.get("source_name") or "")
        if name not in self.sources:
            raise _Refusal(f"Unknown source: {name}")
        with self._lock:
            held = [d for d in self.documents.values() if d.source_name == name and d.indexed]
        size = sum(len(d.text.encode("utf-8")) for d in held)
        directories: dict[str, int] = {}
        extensions: dict[str, int] = {}
        for document in held:
            head, _, _ = document.relative_path.rpartition("/")
            directories[f"{head}/" if head else ""] = (
                directories.get(f"{head}/" if head else "", 0) + 1
            )
            suffix = os.path.splitext(document.relative_path)[1] or ""
            extensions[suffix] = extensions.get(suffix, 0) + 1
        return {
            "knowledge_base": self.knowledge_base,
            "source": {
                "name": name,
                "type": "document_folder",
                "enabled": True,
                "status": "healthy",
                "location": self.sources[name],
                "description": None,
                "last_indexed_at": None,
            },
            "totals": {"documents": len(held), "size_bytes": size},
            "by_extension": [
                {"extension": k, "documents": v} for k, v in sorted(extensions.items())
            ],
            "by_directory": [
                {"directory": k, "documents": v} for k, v in sorted(directories.items())
            ],
            "recent": [],
            "links": {"outgoing": [], "incoming": []},
        }

    def _tool_search_context(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        query = str(arguments.get("query") or "")
        if not query.strip():
            raise _Refusal("Empty query: a search needs something to search for")
        limit = int(arguments.get("max_results") or 10)
        snapshot_id = arguments.get("snapshot_id")
        if snapshot_id:
            snapshot = self.snapshots.get(str(snapshot_id))
            if snapshot is None:
                raise _Refusal(f"Unknown snapshot: {snapshot_id}")
            if snapshot["state_digest"] != self._state_digest():
                raise _Refusal(
                    f"Snapshot {snapshot_id} no longer describes this region: "
                    + ", ".join(self._drifted_sections(snapshot))
                    + " changed since it was sealed."
                )

        expansion = _expansion(arguments.get("expand"))

        memory_mode = arguments.get("memory", "auto")
        memory_enabled = memory_mode not in ("off", None, False)
        steering = self._steering_terms() if memory_enabled else {}
        as_of = arguments.get("as_of")

        indexed = [d for d in self.documents.values() if d.indexed]
        scored = self._bm25(query, indexed, steering)
        results: list[dict[str, Any]] = []
        for rank, (document, score) in enumerate(scored[:limit], start=1):
            # Pheasant's hit shape: the artifact as `node_id`, the matched
            # chunk's capped preview rather than the passage, and the
            # region's own source name in provenance - not the lab's.
            preview = document.text[:500]
            results.append(
                {
                    "rank": rank,
                    "node_id": document.artifact_id,
                    "chunk_id": f"chunk:{document.artifact_id}:chunk=0000",
                    "type": "chunk",
                    "title": document.relative_path,
                    "path": f"{self._directory(document.source_name)}/{document.relative_path}",
                    "relative_path": document.relative_path,
                    "score": round(score, 6),
                    "reason": "mock BM25",
                    "summary": document.text[:240],
                    "chunks": [
                        {
                            "chunk_id": f"chunk:{document.artifact_id}:chunk=0000",
                            "text_preview": preview,
                        }
                    ],
                    "provenance": {
                        "source_id": document.source_name,
                        "path": f"{self._directory(document.source_name)}/{document.relative_path}",
                        "relative_path": document.relative_path,
                        "source_type": "document_folder",
                    },
                    "retrieved_by": "text",
                }
            )
        if memory_enabled:
            for record in self._memory_hits(
                query, as_of=as_of, include_rules=_include_rules(memory_mode)
            ):
                relative = f"{record.scope}/{record.record_id}.md"
                results.append(
                    {
                        "rank": len(results) + 1,
                        "node_id": _artifact_id("agent-memory", relative),
                        "type": "chunk",
                        "title": relative,
                        "relative_path": relative,
                        "score": 0.5,
                        "summary": record.text[:240],
                        "chunks": [{"text_preview": record.text[:500]}],
                        "retrieved_by": "text",
                        "memory": {
                            "record_id": record.record_id,
                            "scope": record.scope,
                            "subject": record.subject,
                            "kind": record.kind,
                            "asserted_at": record.asserted_at,
                            "tier": "hot",
                        },
                        "provenance": {
                            "source_id": "agent-memory",
                            "relative_path": relative,
                            "source_type": "memory",
                            "memory": True,
                        },
                    }
                )
        results = results[:limit] if not memory_enabled else results
        payload: dict[str, Any] = {
            "results": results,
            "query": query,
            "knowledge_base": self.knowledge_base,
            "mode": arguments.get("mode", "hybrid"),
            "graph_generation": self._generation_id(),
            "lineage": {
                "state": {
                    "snapshot_id": snapshot_id,
                    "graph_generation": self._generation_id(),
                    "memory": {"enabled": memory_enabled, "steering": sorted(steering)},
                }
            },
        }
        if expansion is not None:
            # Pheasant's shape, walked over no graph: this region has none, so
            # every neighbourhood is empty and says so rather than inventing
            # neighbours a real region would not have.
            seeds = list(dict.fromkeys(str(hit["node_id"]) for hit in results))
            walked = seeds[:25]
            for hit in results:
                if hit["node_id"] in walked:
                    hit["graph"] = {"seed": hit["node_id"], "neighbors": [], "truncated": False}
            payload["expansion"] = {
                **expansion,
                "seeds": len(walked),
                "seeds_skipped": len(seeds) - len(walked),
                "nodes": 0,
            }
        return payload

    def _tool_get_file_summary(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        path = str(arguments.get("path") or "")
        source = arguments.get("source_name")
        document = next(
            (
                d
                for d in self.documents.values()
                if d.relative_path == path and (not source or d.source_name == source)
            ),
            None,
        )
        if document is None:
            raise _Refusal(f"Unknown file: {path}")
        return {
            "id": document.artifact_id,
            "source_id": document.source_name,
            "relative_path": document.relative_path,
            "sha256": document.content_digest.removeprefix("sha256:"),
            "status": "indexed" if document.indexed else "pending",
            "summary": document.text[:240],
            "content": document.text,
        }

    def _tool_memory_write(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        text = str(arguments.get("text") or "")
        if not text.strip():
            raise _Refusal("Empty memory record")
        scope = str(arguments.get("scope") or "user")
        kind = str(arguments.get("kind") or "fact")
        subject = arguments.get("subject")
        principal = arguments.get("principal")
        supersedes = arguments.get("supersedes")
        record_id = (
            "mem-" + digest({"t": text.lower().strip(), "s": scope, "k": kind, "j": subject})[7:23]
        )
        if record_id in self.memory:
            return {
                "record": self._record_payload(self.memory[record_id]),
                "created": False,
                "source": MEMORY_SOURCE,
                "outcome": "duplicate",
            }
        record = MemoryRecord(
            record_id=record_id,
            text=text,
            scope=scope,
            kind=kind,
            subject=str(subject) if subject else None,
            principal=str(principal) if principal else None,
            asserted_at=isonow(),
            supersedes=str(supersedes) if supersedes else None,
            valid_until=arguments.get("valid_until"),
        )
        if record.supersedes and record.supersedes in self.memory:
            self.memory[record.supersedes].superseded_by = record_id
        self.memory[record_id] = record
        self._generation += 1
        # Pheasant nests the stored record rather than flattening it.
        payload: dict[str, Any] = {
            "record": self._record_payload(record),
            "created": True,
            "source": MEMORY_SOURCE,
            "outcome": "created",
        }
        if arguments.get("sync"):
            if self.claim_seconds > 0:
                # A fleet publishes the memory source's sync, as it does any
                # other (captured from pheasant 0.13.2): the record is stored
                # and is not searchable until an indexer has run the task.
                record.indexed = False
                payload["sync"] = self._publish(MEMORY_SOURCE)
            else:
                payload["sync"] = {
                    "source_id": MEMORY_SOURCE,
                    "indexed_artifacts": 1,
                    "skipped_artifacts": 0,
                    "status": "healthy",
                }
        return payload

    @staticmethod
    def _record_payload(record: MemoryRecord) -> dict[str, Any]:
        return {
            "record_id": record.record_id,
            "scope": record.scope,
            "subject": record.subject,
            "text": record.text,
            "kind": record.kind,
            "asserted_at": record.asserted_at,
            "supersedes": record.supersedes,
            "written_by": record.principal,
            "valid_until": record.valid_until,
        }

    def _tool_seal_snapshot(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        state = self._state_digest()
        snapshot_id = "snap-" + state.removeprefix("sha256:")[:16]
        snapshot = self.snapshots.get(snapshot_id)
        if snapshot is None:
            snapshot = {
                "snapshot_id": snapshot_id,
                "label": arguments.get("label"),
                "sealed_by": arguments.get("sealed_by"),
                "note": arguments.get("note"),
                "sealed_at": isonow(),
                "state_digest": state,
                "sections": self._sections(),
                "sealed": True,
            }
            self.snapshots[snapshot_id] = snapshot
        return dict(snapshot)

    def _tool_get_snapshot(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        snapshot = self.snapshots.get(str(arguments.get("snapshot_id")))
        if snapshot is None:
            raise _Refusal(f"Unknown snapshot: {arguments.get('snapshot_id')}")
        drifted = self._drifted_sections(snapshot)
        return {
            **snapshot,
            "verification": {"current": not drifted, "drifted_sections": drifted},
        }

    def _tool_describe_retrieval(self, _: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "default_mode": "hybrid",
            "modes_available": ["text"],
            "max_results": 10,
            "sources": sorted(self.sources),
            "source_types": ["document_folder"] if self.sources else [],
            "memory": {"records": len(self.memory), "steering": len(self._steering_terms())},
            "limitation": "mock region: BM25 only, no vector or graph arm",
        }

    def _tool_record_evidence(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        self.evidence.append(dict(arguments))
        return {"recorded": True, "event_type": arguments.get("event_type")}

    # -- index -------------------------------------------------------------
    def _bm25(
        self, query: str, documents: list[StoredDocument], steering: Mapping[str, Any]
    ) -> list[tuple[StoredDocument, float]]:
        if not documents:
            return []
        terms = tokenize(query)
        for alias, _ in steering.items():
            if alias in terms:
                terms.extend(
                    tokenize(str(steering[alias])) if isinstance(steering[alias], str) else []
                )
        frequencies = [Counter(d.tokens) for d in documents]
        lengths = [max(1, sum(f.values())) for f in frequencies]
        average = sum(lengths) / len(lengths)
        containing = Counter()
        for counts in frequencies:
            for term in set(counts):
                containing[term] += 1
        total = len(documents)

        scored: list[tuple[StoredDocument, float]] = []
        for document, counts, length in zip(documents, frequencies, lengths, strict=True):
            score = 0.0
            for term in terms:
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                idf = math.log(1 + (total - containing[term] + 0.5) / (containing[term] + 0.5))
                score += (
                    idf * (frequency * (K1 + 1)) / (frequency + K1 * (1 - B + B * length / average))
                )
            title = str(document.metadata.get("title") or "")
            title_tokens = set(tokenize(title))
            score += 0.6 * len(title_tokens & set(terms))
            for rule, triggers in steering.items():
                # `prefer:<path>` -> its trigger terms, as pheasant applies a
                # preference: a relative-path prefix, only when a trigger is
                # in the query.
                if (
                    rule.startswith("prefer:")
                    and isinstance(triggers, tuple)
                    and set(triggers) & set(terms)
                    and document.relative_path.lower().startswith(rule.removeprefix("prefer:"))
                ):
                    score += PREFER_BONUS
            if score > 0:
                scored.append((document, score))
        scored.sort(key=lambda pair: (-pair[1], pair[0].artifact_id))
        return scored

    def _steering_terms(self) -> dict[str, Any]:
        terms: dict[str, Any] = {}
        for record in self.memory.values():
            if not record.indexed:
                continue
            if not record.current or record.kind not in {"alias", "preference", "exclusion"}:
                continue
            # pheasant reads an alias from the record's first line only; the
            # lines after it are prose, and expanding a query with them would
            # be a rule the region never applies.
            rule = record.text.strip().partition("\n")[0]
            if record.kind == "alias" and "->" in rule:
                left, _, right = rule.partition("->")
                terms[left.strip().lower()] = right.strip()
            elif record.kind == "preference":
                # Pheasant's grammar, and only pheasant's: a record it cannot
                # parse is ignored there, so it is ignored here too.
                match = PREFERENCE.search(record.text)
                if not match:
                    continue
                triggers = tuple(
                    t.strip().lower() for t in match.group("when").split(",") if t.strip()
                )
                for path in match.group("prefer").split(","):
                    if path.strip():
                        terms["prefer:" + path.strip().lower().rstrip("/*")] = triggers
        return terms

    def _memory_hits(self, query: str, *, as_of: Any, include_rules: bool) -> list[MemoryRecord]:
        wanted = set(tokenize(query))
        hits = []
        for record in self.memory.values():
            if not record.indexed:
                continue
            if not include_rules and record.kind != "fact":
                continue
            if as_of is None and not record.current:
                continue
            if wanted & set(tokenize(record.text)):
                hits.append(record)
        return sorted(hits, key=lambda r: r.record_id)

    # -- state -------------------------------------------------------------
    def _sections(self) -> dict[str, str]:
        # A memory record is an indexed artifact of the memory source in
        # pheasant, so writing one moves `corpus` as well as `memory`. A mock
        # that kept them apart would let a drift rule pass here that fails on
        # every real region.
        return {
            "corpus": digest(
                sorted(
                    [(d.artifact_id, d.content_digest) for d in self.documents.values()]
                    + [(r.record_id, digest_text(r.text)) for r in self.memory.values()]
                )
            ),
            "retrieval": digest({"bm25": {"k1": K1, "b": B}}),
            "memory": digest(sorted((r.record_id, r.superseded_by) for r in self.memory.values())),
        }

    def _state_digest(self) -> str:
        return digest(self._sections())

    def _drifted_sections(self, snapshot: Mapping[str, Any]) -> list[str]:
        current = self._sections()
        recorded = snapshot.get("sections") or {}
        return sorted(name for name, value in current.items() if recorded.get(name) != value)

    def _generation_id(self) -> str:
        return digest({"g": self._generation, "s": self._state_digest()}).removeprefix("sha256:")[
            :16
        ]

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _ok(request_id: Any, result: Any) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    @staticmethod
    def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}

    @staticmethod
    def _tool_error(message: str) -> dict[str, Any]:
        return {"content": [{"type": "text", "text": message}], "isError": True}


class _Refusal(RuntimeError):
    """A deliberate, informative tool refusal."""


class _InjectedFailure(RuntimeError):
    def __init__(self, kind: str, tool: str) -> None:
        super().__init__(f"{kind} on {tool}")
        self.kind = kind
        self.tool = tool

    def to_exception(self) -> BaseException:
        from .client import RateLimited, TransportError

        if self.kind == "timeout":
            return TimeoutError(f"injected timeout on {self.tool}")
        if self.kind == "rate_limit":
            return RateLimited(f"injected 429 on {self.tool}", retry_after=0.0, status_code=429)
        if self.kind == "transport":
            return TransportError(f"injected transport failure on {self.tool}")
        return protocol.JsonRpcError(
            protocol.INTERNAL_ERROR, f"injected internal error on {self.tool}"
        )


def _artifact_id(source_name: str, relative_path: str) -> str:
    """Pheasant's artifact id grammar for a submitted (non-git) file."""

    return f"file:{source_name}:{relative_path}:branch=none"


def _include_rules(memory_mode: Any) -> bool:
    return bool(isinstance(memory_mode, Mapping) and memory_mode.get("include_rules"))


_EXPANSION_KEYS = ("depth", "max_neighbors", "edge_types", "exclude_edge_types")
_DEFAULT_EXPANSION_EXCLUDES = ["has_chunk", "indexes"]


def _expansion(value: Any) -> dict[str, Any] | None:
    """``expand`` read as pheasant reads it (``services.retrieval.parse_expansion``).

    Returns the settings block pheasant reports, or ``None`` for no expansion,
    and refuses a malformed value with pheasant's own text - a mock that
    accepted what the region refuses would hide the bug it is here to show.
    """

    if value is None or value is False:
        return None
    if value is True:
        value = {}
    elif isinstance(value, int):
        if value == 0:
            return None
        value = {"depth": value}
    elif not isinstance(value, Mapping):
        raise _Refusal(
            f"expand must be true, a depth (1-3), or an object with {', '.join(_EXPANSION_KEYS)}"
        )
    unknown = sorted(set(value) - set(_EXPANSION_KEYS))
    if unknown:
        raise _Refusal(
            f"expand does not take {', '.join(unknown)}; it takes {', '.join(_EXPANSION_KEYS)}"
        )
    depth = value.get("depth", 1)
    if isinstance(depth, bool) or not isinstance(depth, int) or not 1 <= depth <= 3:
        raise _Refusal("expand depth must be an integer from 1 to 3")
    max_neighbors = value.get("max_neighbors", 8)
    if (
        isinstance(max_neighbors, bool)
        or not isinstance(max_neighbors, int)
        or not 1 <= max_neighbors <= 50
    ):
        raise _Refusal("expand.max_neighbors must be an integer from 1 to 50")
    edge_types = value.get("edge_types")
    if "exclude_edge_types" in value:
        excludes = list(value.get("exclude_edge_types") or [])
    else:
        excludes = [] if edge_types else list(_DEFAULT_EXPANSION_EXCLUDES)
    return {
        "depth": depth,
        "max_neighbors": max_neighbors,
        "edge_types": list(edge_types) if edge_types else None,
        "exclude_edge_types": excludes,
    }
