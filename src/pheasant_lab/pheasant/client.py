"""Transports and the client session.

Three transports: Streamable HTTP (the default), stdio, and an in-process
``mock`` used by the tests and by ``pheasant-lab demo``. All three go through
one client, so a behaviour that differs between them is a difference in a
transport - a thing a reader can see - rather than a difference between three
implementations, which is a thing nobody sees until it is reported.

Retry policy, stated once here: **only idempotent operations, or operations
carrying a verified idempotency key, are retried.** Every attempt is recorded,
including the successful one, with its backoff. A partial response is partial;
it is never reported as a success.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from .. import ids
from ..hashing import digest
from ..lifecycle import isonow
from ..redaction import Redactor
from ..settings import PheasantFile
from ..tracing.errors import ErrorImpact
from . import protocol
from .protocol import (
    JsonRpcError,
    McpToolError,
    ProtocolError,
    ToolResult,
)

CLIENT_NAME = "pheasant-swarm-lab"

# Tools whose repetition cannot change server state, and which may therefore
# be retried without an idempotency key.
IDEMPOTENT_TOOL_PREFIXES = (
    "get_",
    "list_",
    "describe_",
    "search_",
    "preview_",
    "explain_",
    "reconcile_",
)


class TransportError(RuntimeError):
    """The transport failed before a JSON-RPC message could be exchanged."""


class Transport(Protocol):  # pragma: no cover - structural
    def open(self) -> None: ...

    def send(self, message: Mapping[str, Any], *, timeout: float) -> dict[str, Any]: ...

    def notify(self, message: Mapping[str, Any]) -> None: ...

    def close(self) -> None: ...

    @property
    def session_id(self) -> str | None: ...


# ---------------------------------------------------------------------------
# Streamable HTTP
# ---------------------------------------------------------------------------


class StreamableHttpTransport:
    """MCP Streamable HTTP.

    Sends ``Accept: application/json, text/event-stream`` and handles both,
    because a server is free to answer either way for the same request; a
    client that only handles one works until the day the server streams.
    """

    def __init__(
        self,
        url: str,
        *,
        token: str | None = None,
        protocol_version: str = "2026-07-28",
        timeout: float = 60.0,
        connect_timeout: float = 10.0,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        self.url = url
        self.token = token
        self.protocol_version = protocol_version
        self.timeout = timeout
        self.connect_timeout = connect_timeout
        self.extra_headers = dict(extra_headers or {})
        self._session_id: str | None = None
        self._client: Any = None
        self._negotiated_version: str | None = None

    @property
    def session_id(self) -> str | None:
        return self._session_id

    def open(self) -> None:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - a declared dependency
            raise TransportError("httpx is required for the streamable_http transport") from exc
        self._client = httpx.Client(
            timeout=httpx.Timeout(self.timeout, connect=self.connect_timeout),
            follow_redirects=True,
        )

    def _headers(self) -> dict[str, str]:
        headers = {
            "content-type": "application/json",
            "accept": "application/json, text/event-stream",
            **self.extra_headers,
        }
        if self._negotiated_version:
            headers["mcp-protocol-version"] = self._negotiated_version
        if self._session_id:
            headers["mcp-session-id"] = self._session_id
        if self.token:
            headers["authorization"] = f"Bearer {self.token}"
        return headers

    def adopt_version(self, version: str) -> None:
        self._negotiated_version = version

    def send(self, message: Mapping[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        import httpx

        if self._client is None:
            self.open()
        response = self._client.post(
            self.url,
            content=json.dumps(message).encode("utf-8"),
            headers=self._headers(),
            timeout=httpx.Timeout(timeout or self.timeout, connect=self.connect_timeout),
        )
        session = response.headers.get("mcp-session-id")
        if session:
            self._session_id = session
        if response.status_code == 429 or response.status_code >= 500:
            retry_after = response.headers.get("retry-after")
            raise RateLimited(
                f"HTTP {response.status_code} from {self.url}",
                retry_after=_parse_retry_after(retry_after),
                status_code=response.status_code,
            )
        if response.status_code >= 400:
            raise TransportError(
                f"HTTP {response.status_code} from {self.url}: {response.text[:400]}"
            )
        content_type = response.headers.get("content-type", "")
        if content_type.startswith("text/event-stream"):
            messages = list(protocol.iter_sse(iter(response.text.splitlines(keepends=True))))
            for candidate in messages:
                if candidate.get("id") == message.get("id"):
                    return candidate
            if messages:
                return messages[-1]
            raise ProtocolError("event stream carried no JSON-RPC message")
        if not response.content:
            raise ProtocolError("empty body where a JSON-RPC response was expected")
        return response.json()

    def notify(self, message: Mapping[str, Any]) -> None:
        import httpx

        if self._client is None:
            self.open()
        response = self._client.post(
            self.url,
            content=json.dumps(message).encode("utf-8"),
            headers=self._headers(),
            timeout=httpx.Timeout(self.timeout, connect=self.connect_timeout),
        )
        if response.status_code >= 400 and response.status_code not in (202, 204):
            raise TransportError(
                f"HTTP {response.status_code} on notification {message.get('method')}"
            )

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


class RateLimited(TransportError):
    def __init__(self, message: str, *, retry_after: float | None, status_code: int) -> None:
        super().__init__(message)
        self.retry_after = retry_after
        self.status_code = status_code


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# stdio
# ---------------------------------------------------------------------------


class StdioTransport:
    """Line-delimited JSON over a child process's stdio.

    The command is never guessed. A stdio transport that infers its command
    starts the wrong server, and the run then measures it.
    """

    def __init__(
        self, command: str, *, timeout: float = 60.0, env: Mapping[str, str] | None = None
    ) -> None:
        if not command.strip():
            raise TransportError("stdio transport requires an explicit command")
        self.command = command
        self.timeout = timeout
        self.env = dict(env) if env else None
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()

    @property
    def session_id(self) -> str | None:
        return None

    def open(self) -> None:
        import shlex

        self._process = subprocess.Popen(
            shlex.split(self.command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            env=self.env,
        )

    def _write(self, message: Mapping[str, Any]) -> None:
        if self._process is None or self._process.stdin is None:
            raise TransportError("stdio transport is not open")
        self._process.stdin.write(json.dumps(message) + "\n")
        self._process.stdin.flush()

    def send(self, message: Mapping[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        with self._lock:
            self._write(message)
            assert self._process is not None and self._process.stdout is not None
            deadline = time.monotonic() + (timeout or self.timeout)
            while time.monotonic() < deadline:
                line = self._process.stdout.readline()
                if not line:
                    raise TransportError("stdio server closed its stdout")
                candidate = json.loads(line)
                if candidate.get("id") == message.get("id"):
                    return candidate
                # A notification or an unrelated response; keep reading.
            raise TimeoutError(
                f"no response to {message.get('method')} within {timeout or self.timeout}s"
            )

    def notify(self, message: Mapping[str, Any]) -> None:
        with self._lock:
            self._write(message)

    def close(self) -> None:
        if self._process is not None:
            if self._process.stdin:
                self._process.stdin.close()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:  # pragma: no cover - defensive
                self._process.kill()
            self._process = None


# ---------------------------------------------------------------------------
# in-process
# ---------------------------------------------------------------------------


class InProcessTransport:
    """Calls a handler object directly. Used by the tests and by ``demo``.

    It is a transport rather than a stubbed client on purpose: the handshake,
    the capability resolution, the retry policy and the error translation are
    the things under test, and a stub that skipped them would test nothing.
    """

    def __init__(self, handler: Any) -> None:
        self.handler = handler
        self._session_id = "mock-" + ids.new_nonce(6)

    @property
    def session_id(self) -> str | None:
        return self._session_id

    def open(self) -> None:
        opener = getattr(self.handler, "open", None)
        if callable(opener):
            opener()

    def send(self, message: Mapping[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        response = self.handler.handle(dict(message))
        if response is None:
            raise ProtocolError(f"handler returned nothing for {message.get('method')}")
        return response

    def notify(self, message: Mapping[str, Any]) -> None:
        self.handler.handle(dict(message))

    def close(self) -> None:
        closer = getattr(self.handler, "close", None)
        if callable(closer):
            closer()


# ---------------------------------------------------------------------------
# the client
# ---------------------------------------------------------------------------


@dataclass
class CallOutcome:
    """One tool call, with everything the trace needs about it."""

    tool: str
    arguments: dict[str, Any]
    result: ToolResult | None
    attempts: int
    duration_ms: float
    status: str
    error: str | None = None
    refusal_code: str | None = None
    partial: bool = False
    server_trace_id: str | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class PheasantSession:
    """What the handshake established."""

    protocol_version: str
    server_name: str
    server_version: str
    session_id: str | None
    capabilities: dict[str, Any]
    tools: dict[str, dict[str, Any]]
    initialize_request_digest: str
    initialize_response_digest: str
    instructions: str | None = None


class PheasantClient:
    """One MCP connection, traced, retried and redacted."""

    def __init__(
        self,
        config: PheasantFile,
        *,
        transport: Transport | None = None,
        redactor: Redactor | None = None,
        tracer: Any = None,
        token: str | None = None,
        package_version: str = "0.1.0",
        clock: Any = time,
    ) -> None:
        self.config = config
        self.redactor = redactor or Redactor(enabled=True)
        self.tracer = tracer
        self.package_version = package_version
        self._clock = clock
        self._transport = transport or self._build_transport(token)
        self._id = 0
        self._id_lock = threading.Lock()
        self.session: PheasantSession | None = None
        self.calls: list[CallOutcome] = []
        self.refusal_table: dict[str, str] = {}

    # -- construction -----------------------------------------------------
    def _build_transport(self, token: str | None) -> Transport:
        if self.config.transport == "streamable_http":
            return StreamableHttpTransport(
                self.config.url,
                token=token,
                protocol_version=self.config.protocol_version,
                timeout=self.config.timeout_seconds,
                connect_timeout=self.config.connect_timeout_seconds,
            )
        if self.config.transport == "stdio":
            return StdioTransport(self.config.command, timeout=self.config.timeout_seconds)
        raise TransportError(
            f"transport '{self.config.transport}' needs an explicit transport object; "
            "the mock transport is constructed by the caller that owns the handler"
        )

    @classmethod
    def in_process(cls, config: PheasantFile, handler: Any, **kwargs: Any) -> PheasantClient:
        return cls(config, transport=InProcessTransport(handler), **kwargs)

    # -- plumbing ---------------------------------------------------------
    def _next_id(self) -> int:
        with self._id_lock:
            self._id += 1
            return self._id

    def _trace(self, event_type: str, *, status: str, payload: Mapping[str, Any]) -> None:
        if self.tracer is not None:
            self.tracer.emit(event_type, status=status, payload=dict(payload))

    def _transcript(self, record: Mapping[str, Any]) -> None:
        if self.tracer is not None and self.config.transport != "none":
            self.tracer.append("mcp-calls.redacted.jsonl", dict(record))

    # -- handshake --------------------------------------------------------
    def connect(self) -> PheasantSession:
        """``initialize`` -> ``notifications/initialized`` -> ``tools/list``."""

        self._transport.open()
        params = protocol.initialize_params(
            protocol_version=self.config.protocol_version,
            client_name=CLIENT_NAME,
            client_version=self.package_version,
        )
        message = protocol.request(protocol.METHOD_INITIALIZE, params, self._next_id())
        started = self._clock.monotonic()
        response = self._transport.send(message, timeout=self.config.timeout_seconds)
        result = protocol.unwrap(response, expect_id=message["id"])
        duration_ms = (self._clock.monotonic() - started) * 1000.0

        negotiated = str(result.get("protocolVersion") or self.config.protocol_version)
        adopt = getattr(self._transport, "adopt_version", None)
        if callable(adopt):
            adopt(negotiated)

        self._transport.notify(protocol.notification(protocol.NOTIFICATION_INITIALIZED))

        tools = self._list_tools()
        server_info = result.get("serverInfo") or {}
        session = PheasantSession(
            protocol_version=negotiated,
            server_name=str(server_info.get("name", "unknown")),
            server_version=str(server_info.get("version", "unknown")),
            session_id=self._transport.session_id,
            capabilities=dict(result.get("capabilities") or {}),
            tools=tools,
            initialize_request_digest=digest(message),
            initialize_response_digest=digest(response),
            instructions=result.get("instructions"),
        )
        self.session = session
        self._trace(
            "mcp.session.initialized",
            status="succeeded",
            payload={
                "protocol_version": negotiated,
                "requested_protocol_version": self.config.protocol_version,
                "server_name": session.server_name,
                "server_version": session.server_version,
                "session_id": session.session_id,
                "tool_count": len(tools),
                "duration_ms": duration_ms,
                "initialize_request_digest": session.initialize_request_digest,
                "initialize_response_digest": session.initialize_response_digest,
            },
        )
        return session

    def _list_tools(self) -> dict[str, dict[str, Any]]:
        tools: dict[str, dict[str, Any]] = {}
        cursor: str | None = None
        while True:
            params: dict[str, Any] = {} if cursor is None else {"cursor": cursor}
            message = protocol.request(protocol.METHOD_TOOLS_LIST, params, self._next_id())
            response = self._transport.send(message, timeout=self.config.timeout_seconds)
            result = protocol.unwrap(response, expect_id=message["id"])
            for tool in result.get("tools", []):
                tools[str(tool["name"])] = dict(tool)
            cursor = result.get("nextCursor")
            if not cursor:
                break
        return tools

    # -- calls ------------------------------------------------------------
    def call(
        self,
        tool: str,
        arguments: Mapping[str, Any],
        *,
        idempotent: bool | None = None,
        idempotency_key: str | None = None,
        timeout: float | None = None,
        stage: str = "mcp",
        question_id: str | None = None,
        arm_id: str | None = None,
        allow_error: bool = False,
    ) -> CallOutcome:
        """Call one tool, with the retry policy this module states.

        ``allow_error`` returns the refusal instead of raising, for the calls
        whose refusal *is* the answer - a snapshot drift check, a denylist
        probe.
        """

        retryable = self._is_retryable(tool, idempotent, idempotency_key)
        attempts = 0
        backoff = self.config.retry_backoff_seconds
        started = self._clock.monotonic()
        last_error: BaseException | None = None

        while attempts < max(1, self.config.max_retries):
            attempts += 1
            message = protocol.request(
                protocol.METHOD_TOOLS_CALL,
                {"name": tool, "arguments": dict(arguments)},
                self._next_id(),
            )
            attempt_started = self._clock.monotonic()
            try:
                response = self._transport.send(
                    message, timeout=timeout or self.config.timeout_seconds
                )
                result = protocol.unwrap(response, expect_id=message["id"])
                parsed = protocol.parse_tool_result(tool, result)
                duration_ms = (self._clock.monotonic() - started) * 1000.0
                outcome = self._finish(
                    tool,
                    arguments,
                    parsed,
                    attempts,
                    duration_ms,
                    message,
                    response,
                    question_id=question_id,
                    arm_id=arm_id,
                    attempt_ms=(self._clock.monotonic() - attempt_started) * 1000.0,
                )
                if parsed.is_error and not allow_error:
                    raise McpToolError(
                        tool,
                        parsed.text or "tool reported an error with no message",
                        code=outcome.refusal_code,
                        content=parsed.content,
                    )
                return outcome
            except (RateLimited, TimeoutError, TransportError, JsonRpcError) as exc:
                last_error = exc
                recoverable = retryable and self._retryable_error(exc)
                # `or` would swallow a Retry-After of 0, which is a server
                # saying "immediately" and not a server saying nothing.
                retry_after = getattr(exc, "retry_after", None)
                wait = retry_after if retry_after is not None else backoff
                self._record_error(
                    exc,
                    stage=stage,
                    tool=tool,
                    attempt=attempts,
                    request=message,
                    resolution="retried"
                    if recoverable and attempts < self.config.max_retries
                    else "terminal",
                    backoff=wait if recoverable else None,
                    question_id=question_id,
                )
                if not recoverable or attempts >= self.config.max_retries:
                    break
                self._clock.sleep(min(wait, self.config.retry_backoff_max_seconds))
                backoff = min(backoff * 2, self.config.retry_backoff_max_seconds)

        duration_ms = (self._clock.monotonic() - started) * 1000.0
        outcome = CallOutcome(
            tool=tool,
            arguments=dict(arguments),
            result=None,
            attempts=attempts,
            duration_ms=duration_ms,
            status="failed",
            error=self.redactor.text(str(last_error)) if last_error else "unknown failure",
        )
        self.calls.append(outcome)
        self._trace(
            "mcp.tool.call",
            status="failed",
            payload={
                "tool": tool,
                "attempt": attempts,
                "duration_ms": duration_ms,
                "error": outcome.error,
                "question_id": question_id,
                "arm_id": arm_id,
            },
        )
        assert last_error is not None
        raise last_error

    def _finish(
        self,
        tool: str,
        arguments: Mapping[str, Any],
        parsed: ToolResult,
        attempts: int,
        duration_ms: float,
        request_message: Mapping[str, Any],
        response_message: Mapping[str, Any],
        *,
        question_id: str | None,
        arm_id: str | None,
        attempt_ms: float,
    ) -> CallOutcome:
        payload = parsed.payload()
        warnings: list[str] = []
        partial = False
        server_trace = None
        if isinstance(payload, Mapping):
            warnings = [str(w) for w in (payload.get("warnings") or [])]
            partial = bool(payload.get("partial") or payload.get("partial_results"))
            server_trace = payload.get("trace_id") or payload.get("server_trace_id")
        status = "failed" if parsed.is_error else ("partial" if partial else "succeeded")
        code = protocol.refusal_code(parsed.text, self.refusal_table) if parsed.is_error else None
        outcome = CallOutcome(
            tool=tool,
            arguments=dict(arguments),
            result=parsed,
            attempts=attempts,
            duration_ms=duration_ms,
            status=status,
            error=parsed.text if parsed.is_error else None,
            refusal_code=code,
            partial=partial,
            server_trace_id=str(server_trace) if server_trace else None,
            warnings=warnings,
        )
        self.calls.append(outcome)
        self._trace(
            "mcp.tool.call",
            status=status,
            payload={
                "tool": tool,
                "attempt": attempts,
                "duration_ms": duration_ms,
                "attempt_ms": attempt_ms,
                "partial": partial,
                "warnings": warnings,
                "refusal_code": code,
                "server_trace_id": outcome.server_trace_id,
                "arguments_digest": digest(self.redactor.payload(dict(arguments))),
                "question_id": question_id,
                "arm_id": arm_id,
            },
        )
        transcript = self.config
        record: dict[str, Any] = {
            "recorded_at": isonow(),
            "tool": tool,
            "attempt": attempts,
            "status": status,
            "duration_ms": duration_ms,
            "session_id": self._transport.session_id,
            "question_id": question_id,
            "arm_id": arm_id,
            "request_digest": digest(request_message),
            "response_digest": digest(response_message),
        }
        record["request"] = self.redactor.payload(dict(request_message))
        record["response"] = _truncate(self.redactor.payload(dict(response_message)), 65536)
        del transcript
        self._transcript(record)
        return outcome

    def _record_error(
        self,
        exc: BaseException,
        *,
        stage: str,
        tool: str,
        attempt: int,
        request: Mapping[str, Any],
        resolution: str,
        backoff: float | None,
        question_id: str | None,
    ) -> None:
        if self.tracer is None:
            return
        self.tracer.errors.record(
            exc,
            stage=stage,
            component="pheasant.client",
            operation=tool,
            attempt=attempt,
            request=dict(request),
            resolution=resolution,
            backoff_seconds=backoff,
            trace_id=getattr(self.tracer, "trace_id", None),
            span_id=getattr(self.tracer, "span_id", None),
            impact=ErrorImpact(
                affected_queries=[question_id] if question_id else [],
                excluded_from_metrics=resolution == "terminal",
                comparability="partial" if resolution == "terminal" else "none",
            ),
        )

    @staticmethod
    def _retryable_error(exc: BaseException) -> bool:
        if isinstance(exc, RateLimited | TimeoutError):
            return True
        if isinstance(exc, JsonRpcError):
            return exc.retryable
        return isinstance(exc, TransportError)

    def _is_retryable(
        self, tool: str, idempotent: bool | None, idempotency_key: str | None
    ) -> bool:
        if idempotent is not None:
            return idempotent
        if idempotency_key:
            return True
        return tool.startswith(IDEMPOTENT_TOOL_PREFIXES)

    def cancel(self, request_id: int | str, reason: str = "client cancelled") -> None:
        self._transport.notify(
            protocol.notification(
                protocol.NOTIFICATION_CANCELLED, {"requestId": request_id, "reason": reason}
            )
        )

    def ping(self) -> bool:
        message = protocol.request(protocol.METHOD_PING, {}, self._next_id())
        try:
            protocol.unwrap(
                self._transport.send(message, timeout=self.config.connect_timeout_seconds),
                expect_id=message["id"],
            )
            return True
        except (JsonRpcError, ProtocolError, TransportError):
            return False

    def latency_summary(self) -> dict[str, float]:
        durations = sorted(call.duration_ms for call in self.calls)
        if not durations:
            return {}
        return {
            "count": float(len(durations)),
            "p50_ms": _percentile(durations, 0.50),
            "p95_ms": _percentile(durations, 0.95),
            "max_ms": durations[-1],
        }

    def close(self) -> None:
        self._transport.close()

    def __enter__(self) -> PheasantClient:
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, max(0, round(fraction * (len(sorted_values) - 1))))
    return sorted_values[index]


def _truncate(payload: Any, limit: int) -> Any:
    text = json.dumps(payload, default=str)
    if len(text) <= limit:
        return payload
    return {"_truncated": True, "_bytes": len(text), "_head": text[:limit]}


def iter_messages(lines: Iterator[str]) -> Iterator[dict[str, Any]]:
    """Parse a stdio stream. Exposed for the contract tests."""

    for line in lines:
        if line.strip():
            yield json.loads(line)
