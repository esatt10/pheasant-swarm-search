"""Spans.

Ids are minted here because a span row's primary key must exist with no OTLP
SDK installed. When an SDK *is* installed the span is started there first and
this row adopts **its** ids - otherwise the row and the exported span name two
different calls, and an operator correlating a slow span in their collector to
a local row finds nothing, which is most of the reason to export spans.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .events import JsonlWriter


@dataclass
class Span:
    span_id: str
    trace_id: str
    parent_span_id: str | None
    name: str
    run_id: str
    started_at: str
    attributes: dict[str, Any] = field(default_factory=dict)
    ended_at: str | None = None
    duration_ms: float | None = None
    status: str | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "run_id": self.run_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "error": self.error,
            "attributes": self.attributes,
        }


class SpanRecorder:
    def __init__(self, path: Path, *, enabled: bool = True, flush: bool = True) -> None:
        self.enabled = enabled
        self._writer = JsonlWriter(path, flush=flush) if enabled else None
        self._lock = threading.Lock()
        self.count = 0

    def record(self, span: Span) -> None:
        if not self.enabled or self._writer is None:
            return
        with self._lock:
            self.count += 1
        self._writer.append(span.as_dict())

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()


def otel_ids() -> tuple[str, str] | None:
    """Return the ambient OTLP ``(trace_id, span_id)`` when an SDK is active.

    Returns ``None`` when the SDK is absent or no span is recording, which is
    the ordinary case here and is not an error.
    """

    try:  # pragma: no cover - exercised only with the optional SDK installed
        from opentelemetry import trace as otel_trace
    except ImportError:
        return None
    span = otel_trace.get_current_span()
    context = span.get_span_context()
    if not context.is_valid:
        return None
    return (format(context.trace_id, "032x"), format(context.span_id, "016x"))
