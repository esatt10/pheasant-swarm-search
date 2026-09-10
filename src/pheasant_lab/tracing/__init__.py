"""The trace plane: append-only raw JSONL, spans, errors, and a projection.

Raw JSONL is authoritative. Everything else here - the DuckDB projection, the
lineage resolver, the reports one level up - is derived, disposable and
rebuildable from it. A projection error never mutates a raw trace.
"""

from .errors import ErrorRecord, ErrorSink, classify_exception
from .events import Event, EventLog, JsonlWriter, Tracer
from .spans import Span, SpanRecorder

__all__ = [
    "ErrorRecord",
    "ErrorSink",
    "Event",
    "EventLog",
    "JsonlWriter",
    "Span",
    "SpanRecorder",
    "Tracer",
    "classify_exception",
]
