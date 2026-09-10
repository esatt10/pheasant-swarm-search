"""JSON-RPC 2.0 and the MCP message shapes this lab speaks.

Written out rather than taken from an SDK for one reason: this repository's
job is to *measure* an MCP server, and a measurement that inherits an SDK's
retry policy, its error translation and its transport defaults is measuring
the SDK as much as the server. Two of those defaults have already changed
under Pheasant between SDK majors.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

JSONRPC_VERSION = "2.0"

# JSON-RPC reserved codes.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

METHOD_INITIALIZE = "initialize"
METHOD_TOOLS_LIST = "tools/list"
METHOD_TOOLS_CALL = "tools/call"
METHOD_RESOURCES_LIST = "resources/list"
METHOD_RESOURCES_READ = "resources/read"
METHOD_PING = "ping"
NOTIFICATION_INITIALIZED = "notifications/initialized"
NOTIFICATION_CANCELLED = "notifications/cancelled"


class ProtocolError(RuntimeError):
    """The peer said something that is not a valid message for this exchange."""


class JsonRpcError(RuntimeError):
    """A JSON-RPC error object came back where a result was expected."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message
        self.data = data

    @property
    def retryable(self) -> bool:
        # Internal errors may be transient; a method the server does not have
        # and parameters it rejected will not become valid on a second try.
        return self.code == INTERNAL_ERROR


class McpToolError(RuntimeError):
    """A tool call that came back ``isError``.

    MCP carries a string and nothing else here, which is why Pheasant
    publishes a refusal-code table and an agent maps the text onto a code. The
    text is preserved verbatim: an SDK that summarises it to "Error executing
    tool" is deciding what this lab is allowed to measure.
    """

    def __init__(
        self, tool: str, message: str, *, code: str | None = None, content: Any = None
    ) -> None:
        super().__init__(message)
        self.tool = tool
        self.message = message
        self.code = code
        self.content = content


def request(method: str, params: Mapping[str, Any] | None, request_id: int | str) -> dict[str, Any]:
    return {
        "jsonrpc": JSONRPC_VERSION,
        "id": request_id,
        "method": method,
        "params": dict(params or {}),
    }


def notification(method: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
    message: dict[str, Any] = {"jsonrpc": JSONRPC_VERSION, "method": method}
    if params is not None:
        message["params"] = dict(params)
    return message


def initialize_params(
    *,
    protocol_version: str,
    client_name: str,
    client_version: str,
    capabilities: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "protocolVersion": protocol_version,
        "capabilities": dict(capabilities or {}),
        "clientInfo": {"name": client_name, "version": client_version},
    }


def unwrap(message: Mapping[str, Any], *, expect_id: int | str | None = None) -> Any:
    """Return a response's ``result``, or raise its ``error``."""

    if message.get("jsonrpc") != JSONRPC_VERSION:
        raise ProtocolError(f"not a JSON-RPC 2.0 message: {message.get('jsonrpc')!r}")
    if expect_id is not None and message.get("id") != expect_id:
        raise ProtocolError(
            f"response id {message.get('id')!r} does not answer request {expect_id!r}"
        )
    if "error" in message:
        error = message["error"] or {}
        raise JsonRpcError(
            int(error.get("code", INTERNAL_ERROR)), str(error.get("message", "")), error.get("data")
        )
    if "result" not in message:
        raise ProtocolError("response carries neither result nor error")
    return message["result"]


@dataclass(frozen=True)
class ToolResult:
    """A ``tools/call`` result, with the two shapes MCP allows for content."""

    tool: str
    content: list[dict[str, Any]]
    structured: dict[str, Any] | None
    is_error: bool

    @property
    def text(self) -> str:
        return "\n".join(
            block.get("text", "") for block in self.content if block.get("type") == "text"
        )

    def payload(self) -> Any:
        """The tool's answer as data.

        Prefers ``structuredContent`` when the server offers it, and otherwise
        parses the text block - Pheasant's tools return JSON there. A text
        block that is not JSON is returned as a string rather than raising:
        the refusal text of a failed call is exactly that case, and losing it
        is how an agent ends up told only that something failed.
        """

        if self.structured is not None:
            return self.structured
        text = self.text.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text


def parse_tool_result(tool: str, result: Mapping[str, Any]) -> ToolResult:
    content = list(result.get("content") or [])
    structured = result.get("structuredContent")
    return ToolResult(
        tool=tool,
        content=[dict(block) for block in content],
        structured=dict(structured) if isinstance(structured, Mapping) else None,
        is_error=bool(result.get("isError", False)),
    )


def iter_sse(stream: Iterator[str]) -> Iterator[dict[str, Any]]:
    """Parse ``text/event-stream`` lines into JSON-RPC messages.

    Only the ``data:`` field carries a message. ``event:``, ``id:`` and
    comment lines are ignored, and a blank line dispatches the accumulated
    data, per the SSE grammar.
    """

    buffer: list[str] = []
    for raw in stream:
        line = raw.rstrip("\n").rstrip("\r")
        if line.startswith(":"):
            continue
        if not line:
            if buffer:
                payload = "\n".join(buffer)
                buffer.clear()
                if payload.strip():
                    yield json.loads(payload)
            continue
        field, _, value = line.partition(":")
        if field == "data":
            buffer.append(value[1:] if value.startswith(" ") else value)
    if buffer:
        payload = "\n".join(buffer)
        if payload.strip():
            yield json.loads(payload)


def refusal_code(message: str, table: Mapping[str, str] | None) -> str | None:
    """Map a refusal's text onto a machine-readable code.

    ``table`` comes from the region's own readiness contract. Matching is on
    the code string appearing in the message, which is how Pheasant spells its
    refusals; nothing is inferred from wording.
    """

    if not table:
        return None
    upper = message.upper()
    for code in sorted(table, key=len, reverse=True):
        if code.upper() in upper:
            return code
    return None
