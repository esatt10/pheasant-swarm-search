"""The MCP wire contract.

These drive the real client against the in-process region: the handshake, the
negotiated version, tool discovery, the two content shapes a tool result may
take, and the error translation. Nothing here is stubbed above the transport,
because the handshake and the error translation are the things under test.
"""

from __future__ import annotations

import json

import pytest

from pheasant_lab.pheasant import protocol
from pheasant_lab.pheasant.client import PheasantClient
from pheasant_lab.pheasant.mock import MockPheasantServer
from pheasant_lab.pheasant.protocol import JsonRpcError, McpToolError, ProtocolError


def test_initialize_then_initialized_then_tools_list(config, mock_server):
    client = PheasantClient.in_process(config.pheasant, mock_server)
    session = client.connect()
    assert mock_server.initialized is True, "the initialized notification was never sent"
    assert session.server_name == "pheasant-mock"
    assert "search_context" in session.tools
    assert session.initialize_request_digest.startswith("sha256:")


def test_the_protocol_version_is_negotiated_not_asserted(config, mock_server):
    config.pheasant.protocol_version = "2025-01-01"
    client = PheasantClient.in_process(config.pheasant, mock_server)
    session = client.connect()
    assert session.protocol_version == "2025-01-01", "a server must be able to negotiate down"


def test_a_newer_client_revision_is_capped_by_the_server(config):
    server = MockPheasantServer(protocol_version="2026-07-28")
    config.pheasant.protocol_version = "2099-01-01"
    session = PheasantClient.in_process(config.pheasant, server).connect()
    assert session.protocol_version == "2026-07-28"


def test_tool_discovery_pages(config, mock_server):
    session = PheasantClient.in_process(config.pheasant, mock_server).connect()
    assert len(session.tools) == len(mock_server._tools())


def test_a_tool_refusal_carries_the_servers_own_text(config, mock_server):
    client = PheasantClient.in_process(config.pheasant, mock_server)
    client.connect()
    with pytest.raises(McpToolError) as caught:
        client.call("search_context", {"knowledge_base": "pheasant-lab", "query": ""})
    assert "Empty query" in str(caught.value), (
        "an SDK that summarises a refusal to 'error executing tool' decides what this lab "
        "is allowed to measure"
    )


def test_an_unknown_knowledge_base_is_refused_informatively(config, mock_server):
    client = PheasantClient.in_process(config.pheasant, mock_server)
    client.connect()
    with pytest.raises(McpToolError, match="Unknown knowledge base"):
        client.call("list_knowledge_bases", {"knowledge_base": "not-this-one"})


def test_allow_error_returns_the_refusal_instead_of_raising(config, mock_server):
    client = PheasantClient.in_process(config.pheasant, mock_server)
    client.connect()
    outcome = client.call(
        "get_snapshot",
        {"knowledge_base": "pheasant-lab", "snapshot_id": "snap-nope"},
        allow_error=True,
    )
    assert outcome.status == "failed"
    assert "Unknown snapshot" in (outcome.error or "")


def test_a_jsonrpc_error_object_is_raised_not_returned():
    message = {"jsonrpc": "2.0", "id": 1, "error": {"code": -32601, "message": "no such method"}}
    with pytest.raises(JsonRpcError) as caught:
        protocol.unwrap(message, expect_id=1)
    assert caught.value.code == -32601
    assert caught.value.retryable is False


def test_an_internal_error_is_the_only_retryable_jsonrpc_code():
    assert JsonRpcError(-32603, "internal").retryable is True
    assert JsonRpcError(-32602, "bad params").retryable is False


def test_a_response_answering_another_request_is_refused():
    with pytest.raises(ProtocolError, match="does not answer"):
        protocol.unwrap({"jsonrpc": "2.0", "id": 9, "result": {}}, expect_id=1)


def test_a_result_with_neither_result_nor_error_is_refused():
    with pytest.raises(ProtocolError, match="neither result nor error"):
        protocol.unwrap({"jsonrpc": "2.0", "id": 1}, expect_id=1)


def test_structured_content_is_preferred_over_the_text_block():
    result = protocol.parse_tool_result(
        "t",
        {"content": [{"type": "text", "text": '{"a": 1}'}], "structuredContent": {"a": 2}},
    )
    assert result.payload() == {"a": 2}


def test_non_json_text_comes_back_as_a_string_rather_than_raising():
    result = protocol.parse_tool_result("t", {"content": [{"type": "text", "text": "not json"}]})
    assert result.payload() == "not json"


def test_a_malformed_body_does_not_lose_the_refusal_text(config):
    from pheasant_lab.pheasant.mock import FaultPlan

    server = MockPheasantServer(faults=FaultPlan(malformed={"search_context"}))
    client = PheasantClient.in_process(config.pheasant, server)
    client.connect()
    outcome = client.call("search_context", {"knowledge_base": "pheasant-lab", "query": "x"})
    assert outcome.result is not None
    assert outcome.result.payload() == "{not json"


def test_sse_framing_is_parsed():
    stream = iter(
        [
            ": a comment\n",
            "event: message\n",
            'data: {"jsonrpc":"2.0","id":1,"result":{"ok":true}}\n',
            "\n",
        ]
    )
    messages = list(protocol.iter_sse(stream))
    assert messages == [{"jsonrpc": "2.0", "id": 1, "result": {"ok": True}}]


def test_multiline_sse_data_is_joined():
    stream = iter(["data: {\n", 'data: "a": 1}\n', "\n"])
    assert list(protocol.iter_sse(stream)) == [{"a": 1}]


def test_a_cancellation_is_a_notification_with_no_id(config, mock_server):
    client = PheasantClient.in_process(config.pheasant, mock_server)
    client.connect()
    client.cancel(7, "test")  # must not raise, and must not expect a response


def test_refusal_codes_are_mapped_from_the_published_table():
    table = {"CORPUS_DENYLISTED": "x", "SNAPSHOT_DRIFTED": "y"}
    assert protocol.refusal_code("SNAPSHOT_DRIFTED: corpus moved", table) == "SNAPSHOT_DRIFTED"
    assert protocol.refusal_code("something else", table) is None
    assert protocol.refusal_code("SNAPSHOT_DRIFTED", None) is None


def test_the_transcript_never_carries_an_authorization_header(config, mock_server, tracer):
    client = PheasantClient.in_process(config.pheasant, mock_server, tracer=tracer)
    client.connect()
    client.call("list_knowledge_bases", {"knowledge_base": "pheasant-lab"})
    tracer.close()
    body = (tracer.paths.raw_file("mcp-calls.redacted.jsonl")).read_text()
    assert "authorization" not in body.lower()
    for line in body.splitlines():
        record = json.loads(line)
        assert record["request_digest"].startswith("sha256:")
