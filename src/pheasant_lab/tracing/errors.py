"""The error contract.

Three rules this module exists to hold:

* a **partial** Pheasant response is partial, never succeeded;
* an optional diagnostic that fails produces ``not_available`` - it does not
  turn a core metric into ``0.0``;
* nothing here records a secret header, an API key or a complete sensitive
  payload. Request and response bodies are recorded as digests; the message is
  redacted before it is written.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .. import ids
from ..hashing import digest
from ..lifecycle import isonow
from ..redaction import Redactor
from .events import JsonlWriter

Stage = Literal[
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
]

ErrorClass = Literal[
    "timeout",
    "transport",
    "authentication",
    "rate_limit",
    "schema",
    "tool",
    "model",
    "budget",
    "data",
    "invariant",
    "unknown",
]

# Errors that invalidate a comparison rather than merely reducing coverage.
INVALIDATING = frozenset({"invariant"})

RETRYABLE_CLASSES = frozenset({"timeout", "transport", "rate_limit"})


@dataclass
class ErrorImpact:
    affected_queries: list[str] = field(default_factory=list)
    excluded_from_metrics: bool = False
    comparability: Literal["none", "partial", "invalidated"] = "none"

    def as_dict(self) -> dict[str, Any]:
        return {
            "affected_queries": list(self.affected_queries),
            "excluded_from_metrics": self.excluded_from_metrics,
            "comparability": self.comparability,
        }


@dataclass
class ErrorRecord:
    error_id: str
    run_id: str
    occurred_at: str
    stage: str
    error_class: str
    component: str
    operation: str
    retryable: bool | None
    attempt: int
    exception_type: str
    message_redacted: str
    event_id: str | None = None
    trace_id: str | None = None
    span_id: str | None = None
    stack_digest: str | None = None
    request_digest: str | None = None
    response_digest: str | None = None
    related_source_id: str | None = None
    related_pheasant_id: str | None = None
    resolution: Literal["retried", "recovered", "skipped", "terminal", "unresolved"] = "unresolved"
    backoff_seconds: float | None = None
    impact: ErrorImpact = field(default_factory=ErrorImpact)

    def as_dict(self) -> dict[str, Any]:
        return {
            "error_id": self.error_id,
            "run_id": self.run_id,
            "event_id": self.event_id,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "occurred_at": self.occurred_at,
            "stage": self.stage,
            "class": self.error_class,
            "component": self.component,
            "operation": self.operation,
            "retryable": self.retryable,
            "attempt": self.attempt,
            "exception_type": self.exception_type,
            "message_redacted": self.message_redacted,
            "stack_digest": self.stack_digest,
            "request_digest": self.request_digest,
            "response_digest": self.response_digest,
            "related_source_id": self.related_source_id,
            "related_pheasant_id": self.related_pheasant_id,
            "resolution": self.resolution,
            "backoff_seconds": self.backoff_seconds,
            "impact": self.impact.as_dict(),
        }


def classify_exception(exc: BaseException) -> tuple[str, bool | None]:
    """Map an exception onto ``(class, retryable)``.

    Retryability is a property of the *operation* as much as of the error, so
    a caller that knows the operation is non-idempotent overrides this. The
    default answers what the exception alone can support.
    """

    name = type(exc).__name__
    module = type(exc).__module__.split(".")[0]
    text = str(exc).lower()

    if name == "BudgetExceeded" and module == "pheasant_lab":
        return ("budget", False)
    if name in {"TimeoutError", "ReadTimeout", "ConnectTimeout", "WriteTimeout", "PoolTimeout"}:
        return ("timeout", True)
    if "timeout" in text or "timed out" in text:
        return ("timeout", True)
    if name in {
        "ConnectError",
        "ConnectionError",
        "RemoteProtocolError",
        "ReadError",
        "NetworkError",
    }:
        return ("transport", True)
    if "429" in text or "rate limit" in text or "too many requests" in text:
        return ("rate_limit", True)
    if "401" in text or "403" in text or "unauthor" in text or "forbidden" in text:
        return ("authentication", False)
    if name in {"ValidationError", "JSONDecodeError"} or "schema" in text:
        return ("schema", False)
    if name in {"ToolError", "McpToolError", "PheasantToolError"}:
        return ("tool", False)
    if name in {"KeyError", "IndexError", "AttributeError", "TypeError"}:
        return ("invariant", False)
    if name in {"ValueError", "FileNotFoundError"}:
        return ("data", False)
    return ("unknown", None)


class ErrorSink:
    """Every attempt is recorded, including the ones that later succeeded.

    A retry that worked is a fact about reliability. Recording only terminal
    failures produces a run that looks healthier the flakier it was.
    """

    def __init__(self, path: Path, *, run_id: str, redactor: Redactor, flush: bool = True) -> None:
        self.run_id = run_id
        self.redactor = redactor
        self._writer = JsonlWriter(path, flush=flush)
        self._lock = threading.Lock()
        self.records: list[ErrorRecord] = []

    def record(
        self,
        exc: BaseException | None = None,
        *,
        stage: str,
        component: str,
        operation: str,
        attempt: int = 1,
        error_class: str | None = None,
        retryable: bool | None = None,
        message: str | None = None,
        exception_type: str | None = None,
        event_id: str | None = None,
        trace_id: str | None = None,
        span_id: str | None = None,
        request: Any = None,
        response: Any = None,
        related_source_id: str | None = None,
        related_pheasant_id: str | None = None,
        resolution: str = "unresolved",
        backoff_seconds: float | None = None,
        impact: ErrorImpact | None = None,
        stack: str | None = None,
    ) -> ErrorRecord:
        derived_class, derived_retryable = (
            classify_exception(exc) if exc is not None else ("unknown", None)
        )
        record = ErrorRecord(
            error_id="",
            run_id=self.run_id,
            occurred_at=isonow(),
            stage=stage,
            error_class=error_class or derived_class,
            component=component,
            operation=operation,
            retryable=retryable if retryable is not None else derived_retryable,
            attempt=attempt,
            exception_type=exception_type or (type(exc).__name__ if exc else "None"),
            message_redacted=self.redactor.text(message or (str(exc) if exc else "")),
            event_id=event_id,
            trace_id=trace_id,
            span_id=span_id,
            stack_digest=digest(stack) if stack else None,
            request_digest=digest(self.redactor.payload(request)) if request is not None else None,
            response_digest=digest(self.redactor.payload(response))
            if response is not None
            else None,
            related_source_id=related_source_id,
            related_pheasant_id=related_pheasant_id,
            resolution=resolution,  # type: ignore[arg-type]
            backoff_seconds=backoff_seconds,
            impact=impact or ErrorImpact(),
        )
        record.error_id = ids.error_id(
            self.run_id,
            event_id,
            attempt,
            digest({"m": record.message_redacted, "o": operation, "s": stage}),
        )
        with self._lock:
            self.records.append(record)
        self._writer.append(record.as_dict())
        return record

    def invalidating(self) -> list[ErrorRecord]:
        return [r for r in self.records if r.impact.comparability == "invalidated"]

    def counts_by_class(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for record in self.records:
            totals[record.error_class] = totals.get(record.error_class, 0) + 1
        return dict(sorted(totals.items()))

    def close(self) -> None:
        self._writer.close()
