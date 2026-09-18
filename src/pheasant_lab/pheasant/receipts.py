"""Ingest receipts.

A transport success is not an ingest. The receipt is the fact; the 200 is a
statement about a socket.

Two dispositions are kept apart because they are separate facts:
``accepted`` (the region stored what you submitted) and ``indexed`` (the
region can retrieve it). A lab that treats the first as the second reports a
retrieval failure as a corpus gap.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ..lifecycle import isonow

TERMINAL_OK = frozenset({"indexed", "verified"})
ACCEPTED = frozenset({"accepted", "indexed", "verified"})


@dataclass
class IngestReceipt:
    """One item's receipt, as this lab records it."""

    receipt_id: str
    run_id: str
    source_id: str
    idempotency_key: str
    submission_id: str | None
    status: str
    artifact_id: str | None = None
    document_id: str | None = None
    content_digest: str | None = None
    accepted_content_digest: str | None = None
    deduplicated: bool = False
    dedup_outcome: str | None = None
    server_trace_id: str | None = None
    indexing_state: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    retryable: bool | None = None
    submissions: int = 1
    recorded_at: str = field(default_factory=isonow)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.status in ACCEPTED

    @property
    def indexed(self) -> bool:
        return self.status in TERMINAL_OK

    @property
    def digest_matches(self) -> bool | None:
        """Did the region store the bytes that were submitted?

        ``None`` when the region did not report a digest: unknown is not the
        same answer as no, and reporting it as no would manufacture silent
        losses at the rate the server omits a field.
        """

        if not self.content_digest or not self.accepted_content_digest:
            return None
        return self.content_digest == self.accepted_content_digest

    def as_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "run_id": self.run_id,
            "source_id": self.source_id,
            "idempotency_key": self.idempotency_key,
            "submission_id": self.submission_id,
            "status": self.status,
            "artifact_id": self.artifact_id,
            "document_id": self.document_id,
            "content_digest": self.content_digest,
            "accepted_content_digest": self.accepted_content_digest,
            "digest_matches": self.digest_matches,
            "deduplicated": self.deduplicated,
            "dedup_outcome": self.dedup_outcome,
            "server_trace_id": self.server_trace_id,
            "indexing_state": self.indexing_state,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "retryable": self.retryable,
            "submissions": self.submissions,
            "recorded_at": self.recorded_at,
        }


class ReceiptLedger:
    """Every receipt this run holds, and what they add up to."""

    def __init__(self) -> None:
        self._by_key: dict[str, IngestReceipt] = {}
        self.submitted_keys: set[str] = set()

    def note_submission(self, idempotency_key: str) -> None:
        """Record that a submission was *attempted* under this key.

        Counted here rather than derived from receipts, because a submission
        that never produced a receipt is exactly the case the receipt rate
        exists to expose.
        """

        self.submitted_keys.add(idempotency_key)

    def record(self, receipt: IngestReceipt) -> IngestReceipt:
        existing = self._by_key.get(receipt.idempotency_key)
        if existing is not None:
            # A retry under a key the region has seen folds onto the receipt
            # it wrote. The submission count is the caller's retries, not the
            # region's own writes: counting an acknowledgement here would make
            # a number that moves when the region acts sit in the field a
            # harness reads to prove that *it* retried.
            receipt.submissions = existing.submissions + 1
        self._by_key[receipt.idempotency_key] = receipt
        return receipt

    def acknowledge(
        self, idempotency_key: str, *, status: str, artifact_id: str | None = None
    ) -> None:
        """Cross the index barrier without counting as a submission."""

        receipt = self._by_key.get(idempotency_key)
        if receipt is None:
            return
        receipt.status = status
        if artifact_id:
            receipt.artifact_id = artifact_id

    def get(self, idempotency_key: str) -> IngestReceipt | None:
        return self._by_key.get(idempotency_key)

    def by_source(self, source_id: str) -> list[IngestReceipt]:
        return [r for r in self._by_key.values() if r.source_id == source_id]

    @property
    def receipts(self) -> list[IngestReceipt]:
        return sorted(self._by_key.values(), key=lambda r: r.idempotency_key)

    # -- the numbers -------------------------------------------------------
    def receipt_rate(self) -> tuple[int, int]:
        """``(verified receipts, submitted eligible sources)``."""

        return (sum(1 for r in self._by_key.values() if r.accepted), len(self.submitted_keys))

    def index_rate(self) -> tuple[int, int]:
        return (sum(1 for r in self._by_key.values() if r.indexed), len(self._by_key))

    def digest_verified(self) -> tuple[int, int]:
        checked = [r for r in self._by_key.values() if r.digest_matches is not None]
        return (sum(1 for r in checked if r.digest_matches), len(checked))

    def failures(self) -> list[IngestReceipt]:
        return [r for r in self._by_key.values() if not r.accepted]

    def submitted_without_receipt(self) -> list[str]:
        return sorted(self.submitted_keys - set(self._by_key))


def parse_receipts(
    payload: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    *,
    run_id: str,
    key_to_source: Mapping[str, str],
    submission_id: str | None = None,
    requested_digests: Mapping[str, str] | None = None,
) -> list[IngestReceipt]:
    """Normalise whatever the region returned into receipts.

    Deliberately tolerant about *shape* and strict about *fields*: a server
    that returns receipts under ``receipts`` or ``items`` or bare is still
    read, but a receipt with no status is a receipt this lab will not claim.
    """

    if isinstance(payload, Mapping):
        rows = payload.get("receipts") or payload.get("items") or payload.get("results") or []
        if not rows:
            rows = [
                row
                for group in ("accepted", "rejected", "failed")
                for row in (payload[group] if isinstance(payload.get(group), list) else [])
                if isinstance(row, Mapping)
            ]
        submission_id = payload.get("submission_id", submission_id)
    else:
        rows = list(payload)

    receipts: list[IngestReceipt] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        key = str(row.get("idempotency_key") or row.get("key") or "")
        source_id = key_to_source.get(key, str(row.get("source_id") or ""))
        status = str(row.get("status") or row.get("disposition") or "unknown")
        receipts.append(
            IngestReceipt(
                receipt_id=str(row.get("receipt_id") or f"receipt-{key}"),
                run_id=run_id,
                source_id=source_id,
                idempotency_key=key,
                submission_id=str(submission_id) if submission_id else None,
                status=status,
                artifact_id=_opt(row, "artifact_id", "artifact"),
                document_id=_opt(row, "document_id", "document"),
                content_digest=(requested_digests or {}).get(key),
                accepted_content_digest=_receipt_digest(row),
                deduplicated=bool(row.get("deduplicated") or row.get("folded") or False),
                dedup_outcome=_opt(row, "dedup_outcome", "outcome"),
                server_trace_id=_opt(row, "trace_id", "server_trace_id"),
                indexing_state=_opt(row, "indexing_state", "index_state"),
                error_code=_opt(row, "error_code", "code"),
                error_message=_opt(row, "error", "message"),
                retryable=row.get("retryable"),
                raw=dict(row),
            )
        )
    return receipts


def _receipt_digest(row: Mapping[str, Any]) -> str | None:
    value = _opt(row, "content_digest", "accepted_content_digest", "digest")
    if value:
        return value
    sha256 = _opt(row, "content_sha256")
    return f"sha256:{sha256.removeprefix('sha256:')}" if sha256 else None


def fold_receipts(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Last row wins, keyed by idempotency key.

    ``ingest-receipts.jsonl`` is append-only, so crossing the index barrier
    appends a *new* row for a key rather than rewriting the old one -
    corrections supersede, they never edit. Every reader therefore has to fold
    the file the same way, which is what this function is for.
    """

    folded: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("idempotency_key") or row.get("receipt_id") or "")
        if key:
            folded[key] = dict(row)
    return folded


def _opt(row: Mapping[str, Any], *names: str) -> str | None:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return str(value)
    return None
