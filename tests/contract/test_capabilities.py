"""Capability resolution refuses before it costs anything."""

from __future__ import annotations

import pytest

from pheasant_lab.pheasant.capabilities import MissingCapability, enforce, resolve
from pheasant_lab.pheasant.client import PheasantClient
from pheasant_lab.pheasant.mock import MockPheasantServer


def tools_of(server: MockPheasantServer) -> dict:
    return {tool["name"]: tool for tool in server._tools()}


def test_every_configured_capability_resolves_against_the_mock(config, mock_server):
    resolved = resolve(config.pheasant, tools_of(mock_server))
    assert resolved.missing_required == []
    assert resolved.tool("search") == "search_context"


def test_a_configured_tool_absent_from_tools_list_is_a_preflight_failure(config, mock_server):
    config.pheasant.capabilities["search"].tool = "search_everything"
    resolved = resolve(config.pheasant, tools_of(mock_server))
    with pytest.raises(MissingCapability, match="not in tools/list"):
        enforce(resolved)


def test_an_unset_tool_name_is_reported_as_unset_not_as_missing(config, mock_server):
    config.pheasant.capabilities["search"].tool = ""
    resolved = resolve(config.pheasant, tools_of(mock_server))
    assert resolved.resolutions["search"].reason.startswith("no tool name configured")


def test_heuristic_matching_is_off_by_default(config, mock_server):
    config.pheasant.capabilities["search"].tool = "search"
    resolved = resolve(config.pheasant, tools_of(mock_server))
    assert resolved.resolutions["search"].present is False


def test_heuristic_matching_refuses_an_ambiguous_name(config, mock_server):
    config.pheasant.discovery.allow_heuristic_name_matching = True
    config.pheasant.capabilities["search"].tool = "get_"
    with pytest.raises(MissingCapability, match="ambiguously"):
        resolve(config.pheasant, tools_of(mock_server))


def test_a_schema_that_does_not_accept_our_arguments_is_reported(config, mock_server):
    tools = tools_of(mock_server)
    tools["search_context"]["inputSchema"] = {
        "type": "object",
        "properties": {"knowledge_base": {"type": "string"}},
        "additionalProperties": False,
    }
    resolved = resolve(config.pheasant, tools)
    assert resolved.resolutions["search"].schema_ok is False
    assert "does not accept" in resolved.resolutions["search"].reason


def test_a_required_parameter_we_never_send_is_reported(config, mock_server):
    tools = tools_of(mock_server)
    tools["search_context"]["required"] = ["knowledge_base"]
    tools["search_context"]["inputSchema"]["required"] = ["knowledge_base", "tenant_id"]
    resolved = resolve(config.pheasant, tools)
    assert resolved.resolutions["search"].schema_ok is False
    assert "tenant_id" in resolved.resolutions["search"].reason


def test_a_server_that_renames_a_tool_is_handled_by_configuration(config):
    server = MockPheasantServer(tool_names={"search_context": "find_context"})
    config.pheasant.capabilities["search"].tool = "find_context"
    client = PheasantClient.in_process(config.pheasant, server)
    session = client.connect()
    resolved = resolve(config.pheasant, session.tools)
    enforce(resolved)
    outcome = client.call(resolved.tool("search"), {"knowledge_base": "pheasant-lab", "query": "x"})
    assert outcome.status == "succeeded"


def test_an_optional_capability_that_is_absent_does_not_block_the_run(config, mock_server):
    config.pheasant.capabilities["write_memory"].tool = "not_a_tool"
    resolved = resolve(config.pheasant, tools_of(mock_server))
    enforce(resolved)
    assert resolved.has("write_memory") is False


# -- what the adapter will actually send -----------------------------------


def test_preflight_checks_the_arguments_the_configured_map_will_send(config, mock_server):
    """A mapped name the tool does not accept is caught before any spend.

    This is the check that was missing: `EXPECTED_ARGUMENTS` listed only
    `knowledge_base` and `query` for search, so a map claiming a `snapshot_id`
    the tool has never heard of resolved as usable and failed on the first
    call of a paid run.
    """

    config.pheasant.argument_map["search"]["snapshot_id"] = "snapshot_id"
    tools = tools_of(mock_server)
    tools["search_context"]["inputSchema"]["properties"].pop("snapshot_id")
    resolved = resolve(config.pheasant, tools)
    assert resolved.resolutions["search"].schema_ok is False
    assert "snapshot_id" in resolved.resolutions["search"].reason


def test_document_level_map_keys_are_not_checked_as_tool_arguments(config, mock_server):
    """`relative_path` is a field inside a document, not an ingest argument.

    Checking it against `submit_documents`' schema would report a mismatch
    that is not one, and a preflight that cries wolf is one an operator turns
    off.
    """

    resolved = resolve(config.pheasant, tools_of(mock_server))
    assert resolved.resolutions["ingest"].schema_ok is not False
