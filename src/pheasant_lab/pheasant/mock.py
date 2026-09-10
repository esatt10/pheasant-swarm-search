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

Failure injection is first-class (``FaultPlan``) because the error contract is
a thing this repository must test, and an error path nothing exercises is an
error path nobody has.
"""

from __future__ import annotations

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
from pathlib import Path
from typing import Any

from ..hashing import digest, digest_text
from ..lifecycle import isonow
from . import protocol

TOKEN = re.compile(r"[a-z0-9]+")
K1 = 1.5
B = 0.75


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

    @property
    def tokens(self) -> list[str]:
        return tokenize(self.text)


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
    ) -> None:
        self.knowledge_base = knowledge_base
        self.protocol_version = protocol_version
        self.faults = faults or FaultPlan()
        self.auto_index = auto_index
        self.documents: dict[str, StoredDocument] = {}
        self.receipts: dict[str, dict[str, Any]] = {}
        self.submissions: dict[str, list[str]] = {}
        self.memory: dict[str, MemoryRecord] = {}
        self.snapshots: dict[str, dict[str, Any]] = {}
        self.evidence: list[dict[str, Any]] = []
        self.call_log: list[tuple[str, dict[str, Any]]] = []
        self.initialized = False
        self._tool_names = dict(tool_names or {})
        self._generation = 0
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
            )
            for key, row in (payload.get("documents") or {}).items()
        }
        self.receipts = dict(payload.get("receipts") or {})
        self.submissions = {k: list(v) for k, v in (payload.get("submissions") or {}).items()}
        self.snapshots = dict(payload.get("snapshots") or {})
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
                }
                for key, doc in self.documents.items()
            },
            "receipts": self.receipts,
            "submissions": self.submissions,
            "snapshots": self.snapshots,
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
                },
                ["knowledge_base", "query"],
            ),
            tool(
                "get_file_summary",
                "A compact summary and provenance for one artifact.",
                {**kb, "artifact_id": {"type": "string"}, "file_path": {"type": "string"}},
                ["knowledge_base"],
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
        receipts: list[dict[str, Any]] = []
        keys: list[str] = []
        for entry in arguments.get("documents") or []:
            if not isinstance(entry, Mapping):
                continue
            path = str(entry.get("relative_path") or "")
            text = str(entry.get("text") or "")
            key = str(entry.get("idempotency_key") or digest_text(path + text))
            metadata = dict(entry.get("metadata") or {})
            content_digest = digest_text(text)
            keys.append(key)

            existing = self.receipts.get(key)
            if existing is not None:
                existing["submissions"] = int(existing.get("submissions", 1)) + 1
                existing["deduplicated"] = True
                existing["dedup_outcome"] = "folded_onto_existing_receipt"
                receipts.append(dict(existing))
                continue
            if not path or not text:
                receipt = {
                    "idempotency_key": key,
                    "status": "rejected",
                    "error_code": "EMPTY_DOCUMENT",
                    "error": "a document needs a relative_path and text",
                    "retryable": False,
                    "submissions": 1,
                }
                self.receipts[key] = receipt
                receipts.append(dict(receipt))
                continue

            artifact_id = f"artifact:{self.knowledge_base}:{path}"
            self.documents[artifact_id] = StoredDocument(
                artifact_id=artifact_id,
                relative_path=path,
                text=text,
                metadata=metadata,
                content_digest=content_digest,
                indexed=self.auto_index,
            )
            receipt = {
                "idempotency_key": key,
                "status": "indexed" if self.auto_index else "accepted",
                "artifact_id": artifact_id,
                "document_id": artifact_id,
                "content_digest": content_digest,
                "indexing_state": "indexed" if self.auto_index else "pending",
                "deduplicated": False,
                "submissions": 1,
                "trace_id": digest(key)[7:39],
            }
            self.receipts[key] = receipt
            receipts.append(dict(receipt))
        self.submissions[submission_id] = keys
        self._generation += 1
        return {"submission_id": submission_id, "receipts": receipts, "accepted": len(receipts)}

    def _tool_get_ingest_status(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        key = arguments.get("idempotency_key")
        submission = arguments.get("submission_id")
        if key:
            receipt = self.receipts.get(str(key))
            return {"receipts": [dict(receipt)] if receipt else []}
        keys = (
            self.submissions.get(str(submission), list(self.receipts))
            if submission
            else list(self.receipts)
        )
        return {"receipts": [dict(self.receipts[k]) for k in keys if k in self.receipts]}

    def _tool_acknowledge_ingest(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        submission = arguments.get("submission_id")
        keys = (
            self.submissions.get(str(submission), list(self.receipts))
            if submission
            else list(self.receipts)
        )
        acknowledged = []
        for key in keys:
            receipt = self.receipts.get(key)
            if not receipt or receipt.get("status") == "rejected":
                continue
            artifact = self.documents.get(str(receipt.get("artifact_id")))
            if artifact is None or not artifact.indexed:
                continue
            receipt["status"] = "indexed"
            receipt["indexing_state"] = "indexed"
            acknowledged.append(
                {"idempotency_key": key, "status": "indexed", "artifact_id": receipt["artifact_id"]}
            )
        return {"acknowledged": acknowledged, "count": len(acknowledged)}

    def _tool_reconcile_ingest(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        submission = arguments.get("submission_id")
        keys = (
            self.submissions.get(str(submission), list(self.receipts))
            if submission
            else list(self.receipts)
        )
        claimed = [k for k in keys if self.receipts.get(k, {}).get("artifact_id")]
        held = [k for k in claimed if str(self.receipts[k]["artifact_id"]) in self.documents]
        return {
            "submitted": len(keys),
            "receipts": len(claimed),
            "held": len(held),
            # Not a difference between two totals: two totals can agree while
            # one item was lost and another double-written.
            "silent_loss": len([k for k in claimed if k not in held]),
        }

    def _tool_sync_source(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        mode = str(arguments.get("mode") or "incremental")
        newly = 0
        for document in self.documents.values():
            if not document.indexed:
                document.indexed = True
                newly += 1
        for receipt in self.receipts.values():
            if receipt.get("status") == "accepted":
                receipt["status"] = "indexed"
                receipt["indexing_state"] = "indexed"
        self._generation += 1
        return {
            "source": arguments.get("source_name"),
            "mode": mode,
            "indexed": newly,
            "total": len(self.documents),
            "generation_id": self._generation_id(),
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
                    f"SNAPSHOT_DRIFTED: sections moved since {snapshot_id}: "
                    + ",".join(self._drifted_sections(snapshot))
                )

        memory_mode = arguments.get("memory", "auto")
        memory_enabled = memory_mode not in ("off", None, False)
        steering = self._steering_terms() if memory_enabled else {}
        as_of = arguments.get("as_of")

        indexed = [d for d in self.documents.values() if d.indexed]
        scored = self._bm25(query, indexed, steering)
        results: list[dict[str, Any]] = []
        for rank, (document, score) in enumerate(scored[:limit], start=1):
            metadata = document.metadata
            results.append(
                {
                    "rank": rank,
                    "artifact_id": document.artifact_id,
                    "score": round(score, 6),
                    "content_digest": document.content_digest,
                    "text": _excerpt(document.text, query),
                    "title": metadata.get("title") or document.relative_path,
                    "locator": document.relative_path,
                    "arm": "text",
                    "contributing_arms": ["text"] + (["memory"] if steering else []),
                    "provenance": {
                        "source_id": metadata.get("lab_source_id"),
                        "path": document.relative_path,
                        "source_type": "document_folder",
                    },
                    "metadata": metadata,
                }
            )
        if memory_enabled:
            for record in self._memory_hits(
                query, as_of=as_of, include_rules=_include_rules(memory_mode)
            ):
                results.append(
                    {
                        "rank": len(results) + 1,
                        "artifact_id": f"memory:{record.record_id}",
                        "score": 0.5,
                        "content_digest": digest_text(record.text),
                        "text": record.text,
                        "title": f"memory:{record.subject or record.kind}",
                        "arm": "memory",
                        "memory": {
                            "record_id": record.record_id,
                            "scope": record.scope,
                            "asserted_at": record.asserted_at,
                            "kind": record.kind,
                        },
                        "provenance": {"source_id": None, "source_type": "memory"},
                    }
                )
        return {
            "results": results[:limit] if not memory_enabled else results,
            "query": query,
            "mode": arguments.get("mode", "hybrid"),
            "snapshot_id": snapshot_id,
            "generation_id": self._generation_id(),
            "memory": {"enabled": memory_enabled, "steering_rules": sorted(steering)},
        }

    def _tool_get_file_summary(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        key = str(arguments.get("artifact_id") or arguments.get("file_path") or "")
        document = self.documents.get(key)
        if document is None:
            raise _Refusal(f"Unknown artifact: {key}")
        return {
            "artifact_id": document.artifact_id,
            "path": document.relative_path,
            "content_digest": document.content_digest,
            "text": document.text,
            "metadata": document.metadata,
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
            return {"record_id": record_id, "created": False, "outcome": "duplicate"}
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
        return {
            "record_id": record_id,
            "created": True,
            "outcome": "created",
            "scope": scope,
            "kind": kind,
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
            "sources": [
                {
                    "name": "swarm-lab-literature",
                    "type": "document_folder",
                    "artifacts": len(self.documents),
                }
            ],
            "memory": {"records": len(self.memory), "steering": len(self._steering_terms())},
            "limitation": "mock region: BM25 only, no vector or graph arm",
        }

    def _tool_record_evidence(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        self.evidence.append(dict(arguments))
        return {"recorded": True, "event_type": arguments.get("event_type")}

    # -- index -------------------------------------------------------------
    def _bm25(
        self, query: str, documents: list[StoredDocument], steering: Mapping[str, float]
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
            for rule, weight in steering.items():
                if isinstance(weight, int | float) and rule in document.relative_path.lower():
                    score += float(weight)
            if score > 0:
                scored.append((document, score))
        scored.sort(key=lambda pair: (-pair[1], pair[0].artifact_id))
        return scored

    def _steering_terms(self) -> dict[str, Any]:
        terms: dict[str, Any] = {}
        for record in self.memory.values():
            if not record.current or record.kind not in {"alias", "preference", "exclusion"}:
                continue
            if record.kind == "alias" and "->" in record.text:
                left, _, right = record.text.partition("->")
                terms[left.strip().lower()] = right.strip()
            elif record.kind == "preference":
                for token in tokenize(record.text):
                    terms[token] = 0.4
        return terms

    def _memory_hits(self, query: str, *, as_of: Any, include_rules: bool) -> list[MemoryRecord]:
        wanted = set(tokenize(query))
        hits = []
        for record in self.memory.values():
            if not include_rules and record.kind != "fact":
                continue
            if as_of is None and not record.current:
                continue
            if wanted & set(tokenize(record.text)):
                hits.append(record)
        return sorted(hits, key=lambda r: r.record_id)

    # -- state -------------------------------------------------------------
    def _sections(self) -> dict[str, str]:
        return {
            "corpus": digest(
                sorted((d.artifact_id, d.content_digest) for d in self.documents.values())
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


def _excerpt(text: str, query: str, width: int = 480) -> str:
    """The passage a caller would actually read.

    Centred on the first query term that appears, so a support check compares
    against what was shown rather than against the head of the document.
    """

    terms = tokenize(query)
    lowered = text.lower()
    position = next((lowered.find(term) for term in terms if lowered.find(term) >= 0), 0)
    start = max(0, position - width // 3)
    return text[start : start + width]


def _include_rules(memory_mode: Any) -> bool:
    return bool(isinstance(memory_mode, Mapping) and memory_mode.get("include_rules"))
