"""The universal event envelope and its append-only writer.

Every number a report states resolves back through this stream. Two
properties make that possible and both are checked by ``verify``:

* **sequence continuity** - one monotonic counter per run, so a missing event
  is detectable rather than merely absent;
* **payload digests** - so a payload edited after the fact stops matching the
  envelope that named it.

Corrections supersede. Nothing here is ever rewritten in place.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import ids
from ..hashing import digest
from ..lifecycle import RunPaths, isonow, utcnow
from ..redaction import Redactor

SCHEMA_VERSION = 1

STATUSES = ("started", "succeeded", "failed", "partial", "skipped")

STAGES = (
    "discovery",
    "acquisition",
    "extraction",
    "mcp",
    "ingest",
    "index",
    "retrieval",
    "answer",
    "evaluation",
    "report",
)


@dataclass
class Event:
    """One envelope. Field order matches ``schemas/event.schema.json``."""

    event_id: str
    occurred_at: str
    recorded_at: str
    run_id: str
    trace_id: str
    span_id: str
    sequence: int
    event_type: str
    status: str
    payload_digest: str
    config_digest: str
    schema_version: int = SCHEMA_VERSION
    parent_span_id: str | None = None
    agent_id: str | None = None
    agent_role: str | None = None
    topic_id: str | None = None
    question_id: str | None = None
    arm_id: str | None = None
    input_refs: list[str] = field(default_factory=list)
    output_refs: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "schema_version": self.schema_version,
            "occurred_at": self.occurred_at,
            "recorded_at": self.recorded_at,
            "run_id": self.run_id,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "sequence": self.sequence,
            "agent_id": self.agent_id,
            "agent_role": self.agent_role,
            "topic_id": self.topic_id,
            "question_id": self.question_id,
            "arm_id": self.arm_id,
            "event_type": self.event_type,
            "status": self.status,
            "input_refs": list(self.input_refs),
            "output_refs": list(self.output_refs),
            "payload": self.payload,
            "payload_digest": self.payload_digest,
            "config_digest": self.config_digest,
        }


class JsonlWriter:
    """Append-only JSONL, safe for the bounded concurrency collection uses."""

    def __init__(self, path: Path, *, flush: bool = True, fsync: bool = False) -> None:
        self.path = path
        self._flush = flush
        self._fsync = fsync
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")

    def append(self, record: Mapping[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, sort_keys=True, default=str)
        with self._lock:
            self._handle.write(line + "\n")
            if self._flush:
                self._handle.flush()
            if self._fsync:
                os.fsync(self._handle.fileno())

    def close(self) -> None:
        with self._lock:
            if not self._handle.closed:
                self._handle.flush()
                self._handle.close()

    def __enter__(self) -> JsonlWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Read a raw file. A truncated final line is reported, not skipped."""

    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: malformed JSONL: {exc}") from exc


class EventLog:
    """The run's one event stream."""

    def __init__(
        self,
        path: Path,
        *,
        run_id: str,
        config_digest: str,
        redactor: Redactor,
        flush: bool = True,
        fsync: bool = False,
        start_sequence: int = 0,
    ) -> None:
        self.run_id = run_id
        self.config_digest = config_digest
        self.redactor = redactor
        self._writer = JsonlWriter(path, flush=flush, fsync=fsync)
        self._lock = threading.Lock()
        self._sequence = start_sequence
        self.path = path

    @classmethod
    def resume(cls, path: Path, **kwargs: Any) -> EventLog:
        """Continue an existing stream without renumbering it."""

        highest = 0
        for record in read_jsonl(path):
            highest = max(highest, int(record.get("sequence", 0)))
        return cls(path, start_sequence=highest, **kwargs)

    @property
    def sequence(self) -> int:
        with self._lock:
            return self._sequence

    def emit(
        self,
        event_type: str,
        *,
        status: str = "succeeded",
        payload: Mapping[str, Any] | None = None,
        trace_id: str,
        span_id: str,
        parent_span_id: str | None = None,
        occurred_at: str | None = None,
        agent_id: str | None = None,
        agent_role: str | None = None,
        topic_id: str | None = None,
        question_id: str | None = None,
        arm_id: str | None = None,
        input_refs: list[str] | None = None,
        output_refs: list[str] | None = None,
    ) -> Event:
        if status not in STATUSES:
            raise ValueError(f"unknown event status '{status}'; known: {STATUSES}")
        body = self.redactor.payload(dict(payload or {}))
        payload_digest = digest(body)
        with self._lock:
            self._sequence += 1
            sequence = self._sequence
            event = Event(
                event_id=ids.event_id(self.run_id, sequence, payload_digest),
                occurred_at=occurred_at or isonow(),
                recorded_at=isonow(),
                run_id=self.run_id,
                trace_id=trace_id,
                span_id=span_id,
                parent_span_id=parent_span_id,
                sequence=sequence,
                event_type=event_type,
                status=status,
                payload=body,
                payload_digest=payload_digest,
                config_digest=self.config_digest,
                agent_id=agent_id,
                agent_role=agent_role,
                topic_id=topic_id,
                question_id=question_id,
                arm_id=arm_id,
                input_refs=list(input_refs or []),
                output_refs=list(output_refs or []),
            )
            self._writer.append(event.as_dict())
        return event

    def close(self) -> None:
        self._writer.close()


class Tracer:
    """The one object the rest of the lab traces through.

    Holds the ambient trace/span context so a call three layers down lands in
    the right span without every function signature carrying one, and so the
    trace id can be injected into every hop the lab makes of its own.
    """

    def __init__(
        self,
        paths: RunPaths,
        *,
        run_id: str,
        config_digest: str,
        redactor: Redactor,
        flush: bool = True,
        fsync: bool = False,
        resume: bool = False,
        spans_enabled: bool = True,
        read_only: bool = False,
    ) -> None:
        from .errors import ErrorSink
        from .spans import SpanRecorder

        self.read_only = read_only
        self.paths = paths
        self.run_id = run_id
        self.config_digest = config_digest
        self.redactor = redactor
        events_path = paths.raw_file("events.jsonl")
        factory = EventLog.resume if resume else EventLog
        self.events: EventLog = factory(
            events_path,
            run_id=run_id,
            config_digest=config_digest,
            redactor=redactor,
            flush=flush,
            fsync=fsync,
        )
        self.spans = SpanRecorder(
            paths.raw_file("spans.jsonl"), enabled=spans_enabled and not read_only, flush=flush
        )
        self.errors = ErrorSink(
            paths.raw_file("errors.jsonl"), run_id=run_id, redactor=redactor, flush=flush
        )
        self._writers: dict[str, JsonlWriter] = {}
        self._writer_lock = threading.Lock()
        self._local = threading.local()
        self.root_trace_id = ids.trace_id()

    # -- ambient context ---------------------------------------------------
    @property
    def trace_id(self) -> str:
        return getattr(self._local, "trace_id", self.root_trace_id)

    @property
    def span_id(self) -> str:
        return getattr(self._local, "span_id", "0" * 16)

    @property
    def parent_span_id(self) -> str | None:
        return getattr(self._local, "parent_span_id", None)

    @property
    def attributes(self) -> dict[str, Any]:
        return dict(getattr(self._local, "attributes", {}))

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Any]:
        """Open a span. Nested spans inherit the trace and become children."""

        from .spans import Span

        parent = getattr(self._local, "span_id", None)
        # Captured through the properties, not the raw thread-locals: the raw
        # values are unset on the first span, and restoring `None` afterwards
        # would leave every later event with a null trace id.
        previous = (self.trace_id, self.span_id, self.parent_span_id, self.attributes)
        merged = {**previous[3], **{k: v for k, v in attributes.items() if v is not None}}
        span = Span(
            span_id=ids.span_id(),
            trace_id=self.trace_id,
            parent_span_id=parent,
            name=name,
            run_id=self.run_id,
            started_at=isonow(),
            attributes=merged,
        )
        self._local.trace_id = span.trace_id
        self._local.span_id = span.span_id
        self._local.parent_span_id = parent
        self._local.attributes = merged
        monotonic_start = utcnow()
        try:
            yield span
            span.status = span.status or "ok"
        except BaseException as exc:
            span.status = "error"
            span.error = self.redactor.scrub_exception(exc)
            raise
        finally:
            span.ended_at = isonow()
            span.duration_ms = (utcnow() - monotonic_start).total_seconds() * 1000.0
            self.spans.record(span)
            (
                self._local.trace_id,
                self._local.span_id,
                self._local.parent_span_id,
                self._local.attributes,
            ) = previous

    def new_trace(self) -> str:
        """Start an independent trace: one evaluation arm session, say."""

        self._local.trace_id = ids.trace_id()
        self._local.span_id = "0" * 16
        self._local.parent_span_id = None
        self._local.attributes = {}
        return self._local.trace_id

    # -- emission ----------------------------------------------------------
    def emit(
        self,
        event_type: str,
        *,
        status: str = "succeeded",
        payload: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Event:
        if self.read_only:
            # `verify` must not change the trace it is checking. A command
            # that appends a "resumed" event to the file whose checksum it is
            # about to compare will always report that file as changed.
            return Event(
                event_id="event-readonly",
                occurred_at=isonow(),
                recorded_at=isonow(),
                run_id=self.run_id,
                trace_id=self.trace_id,
                span_id=self.span_id,
                sequence=0,
                event_type=event_type,
                status=status,
                payload=dict(payload or {}),
                payload_digest="",
                config_digest=self.config_digest,
            )
        ambient = self.attributes
        for key in ("agent_id", "agent_role", "topic_id", "question_id", "arm_id"):
            kwargs.setdefault(key, ambient.get(key))
        return self.events.emit(
            event_type,
            status=status,
            payload=payload,
            trace_id=self.trace_id,
            span_id=self.span_id,
            parent_span_id=self.parent_span_id,
            **kwargs,
        )

    @contextmanager
    def operation(
        self, event_type: str, *, payload: Mapping[str, Any] | None = None, **kwargs: Any
    ) -> Iterator[dict[str, Any]]:
        """Emit ``started`` then ``succeeded``/``failed`` around a block.

        The yielded dict is the outcome payload: mutate it and it is what the
        terminal event carries.
        """

        self.emit(event_type, status="started", payload=payload, **kwargs)
        outcome: dict[str, Any] = {}
        try:
            yield outcome
        except BaseException as exc:
            outcome.setdefault("error", self.redactor.scrub_exception(exc))
            self.emit(event_type, status="failed", payload=outcome, **kwargs)
            raise
        status = str(outcome.pop("_status", "succeeded"))
        self.emit(event_type, status=status, payload=outcome, **kwargs)

    # -- side files --------------------------------------------------------
    def writer(self, filename: str) -> JsonlWriter:
        with self._writer_lock:
            writer = self._writers.get(filename)
            if writer is None:
                writer = JsonlWriter(self.paths.raw_file(filename))
                self._writers[filename] = writer
            return writer

    def append(self, filename: str, record: Mapping[str, Any]) -> None:
        if self.read_only:
            return
        self.writer(filename).append(self.redactor.payload(dict(record)))

    def close(self) -> None:
        self.events.close()
        self.spans.close()
        self.errors.close()
        with self._writer_lock:
            for writer in self._writers.values():
                writer.close()
            self._writers.clear()

    def __enter__(self) -> Tracer:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
