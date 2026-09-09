"""Capability resolution.

The rule: **a configured tool that does not appear in ``tools/list`` fails
before any spend.** Heuristic name matching is off by default, because a lab
that guesses ``search`` from ``search_context`` will one day guess
``search_memory`` and measure the wrong thing while reporting the right name.

The second rule: a tool's advertised input schema is checked against the
arguments this lab actually sends. A tool that exists under the right name and
takes different parameters fails in the same place - preflight - rather than
on the first call of a paid run.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ..settings import PheasantFile


class MissingCapability(RuntimeError):
    """A required capability this build cannot satisfy."""


@dataclass
class CapabilityResolution:
    name: str
    configured_tool: str
    required: bool
    present: bool
    schema_ok: bool | None
    reason: str | None = None
    input_schema: dict[str, Any] | None = None

    @property
    def usable(self) -> bool:
        return self.present and self.schema_ok is not False

    def as_dict(self) -> dict[str, Any]:
        return {
            "capability": self.name,
            "tool": self.configured_tool,
            "required": self.required,
            "present": self.present,
            "schema_ok": self.schema_ok,
            "usable": self.usable,
            "reason": self.reason,
        }


@dataclass
class CapabilityMap:
    """What this region can do, resolved against what it advertises."""

    resolutions: dict[str, CapabilityResolution] = field(default_factory=dict)
    server_tools: list[str] = field(default_factory=list)

    def tool(self, capability: str) -> str:
        resolution = self.resolutions.get(capability)
        if resolution is None or not resolution.usable:
            raise MissingCapability(
                f"capability '{capability}' is not usable here"
                + (f": {resolution.reason}" if resolution and resolution.reason else "")
            )
        return resolution.configured_tool

    def has(self, capability: str) -> bool:
        resolution = self.resolutions.get(capability)
        return bool(resolution and resolution.usable)

    @property
    def missing_required(self) -> list[CapabilityResolution]:
        return [r for r in self.resolutions.values() if r.required and not r.usable]

    def as_dict(self) -> dict[str, Any]:
        return {
            "server_tools": sorted(self.server_tools),
            "capabilities": [
                r.as_dict() for r in sorted(self.resolutions.values(), key=lambda r: r.name)
            ],
            "missing_required": [r.name for r in self.missing_required],
        }


# The arguments this lab sends per capability. Preflight checks these against
# the server's advertised schema, so a rename is caught before it costs money.
EXPECTED_ARGUMENTS: dict[str, tuple[str, ...]] = {
    "health": (),
    "ingest": ("knowledge_base", "documents"),
    "ingest_status": ("knowledge_base",),
    "ingest_acknowledge": ("knowledge_base",),
    "ingest_reconcile": ("knowledge_base",),
    "sync": ("knowledge_base", "source_name"),
    "search": ("knowledge_base", "query"),
    "fetch": ("knowledge_base",),
    "write_memory": ("knowledge_base", "text"),
    "snapshot": ("knowledge_base",),
    "snapshot_get": ("knowledge_base", "snapshot_id"),
    "describe_retrieval": ("knowledge_base",),
    "record_evidence": ("knowledge_base", "query", "target_id", "event_type"),
}


def resolve(
    config: PheasantFile,
    server_tools: Mapping[str, Mapping[str, Any]],
    *,
    argument_overrides: Mapping[str, tuple[str, ...]] | None = None,
) -> CapabilityMap:
    """Resolve the configured capability map against ``tools/list``."""

    expected = {**EXPECTED_ARGUMENTS, **(argument_overrides or {})}
    resolutions: dict[str, CapabilityResolution] = {}

    for name, spec in config.capabilities.items():
        configured = (spec.tool or "").strip()
        if not configured:
            resolutions[name] = CapabilityResolution(
                name=name,
                configured_tool="",
                required=spec.required,
                present=False,
                schema_ok=None,
                reason="no tool name configured (the environment variable resolved empty)",
            )
            continue

        tool = server_tools.get(configured)
        if tool is None:
            reason = f"'{configured}' is not in tools/list"
            if config.discovery.allow_heuristic_name_matching:
                match = _heuristic(
                    configured, server_tools, config.discovery.fail_on_ambiguous_capability
                )
                if match:
                    configured, tool = match, server_tools[match]
                    reason = None
            if tool is None:
                resolutions[name] = CapabilityResolution(
                    name=name,
                    configured_tool=spec.tool,
                    required=spec.required,
                    present=False,
                    schema_ok=None,
                    reason=reason,
                )
                continue

        schema = dict(tool.get("inputSchema") or {})
        schema_ok: bool | None = None
        reason = None
        if config.discovery.verify_input_schema and schema:
            missing = _unsatisfied(schema, expected.get(name, ()))
            schema_ok = not missing
            if missing:
                reason = (
                    f"'{configured}' does not accept {sorted(missing)}; "
                    "the lab's arguments and the server's schema disagree"
                )
            unmet_required = _unmet_required(schema, expected.get(name, ()))
            if unmet_required:
                schema_ok = False
                reason = (
                    f"'{configured}' requires {sorted(unmet_required)}, which the lab does not send"
                )
        resolutions[name] = CapabilityResolution(
            name=name,
            configured_tool=configured,
            required=spec.required,
            present=True,
            schema_ok=schema_ok,
            reason=reason,
            input_schema=schema or None,
        )

    return CapabilityMap(resolutions=resolutions, server_tools=sorted(server_tools))


def _unsatisfied(schema: Mapping[str, Any], arguments: tuple[str, ...]) -> set[str]:
    """Arguments the lab sends that the tool's schema does not declare."""

    properties = schema.get("properties")
    if not isinstance(properties, Mapping):
        return set()
    if schema.get("additionalProperties") is True:
        return set()
    return {argument for argument in arguments if argument not in properties}


def _unmet_required(schema: Mapping[str, Any], arguments: tuple[str, ...]) -> set[str]:
    """Required parameters the lab never sends.

    A tool with a required argument the lab has no value for cannot be called
    at all, which is a preflight failure rather than a runtime surprise.
    """

    required = schema.get("required")
    if not isinstance(required, list):
        return set()
    # `knowledge_base` and the argument map's fields are always available.
    always_sent = set(arguments) | {"knowledge_base", "kb_id"}
    return {str(name) for name in required if str(name) not in always_sent}


def _heuristic(configured: str, tools: Mapping[str, Any], fail_on_ambiguous: bool) -> str | None:
    candidates = [name for name in tools if configured in name or name in configured]
    if len(candidates) == 1:
        return candidates[0]
    if candidates and fail_on_ambiguous:
        raise MissingCapability(
            f"'{configured}' matches {sorted(candidates)} ambiguously; "
            "name the tool exactly rather than letting the lab pick"
        )
    return None


def enforce(capabilities: CapabilityMap) -> None:
    """Raise unless every required capability is usable."""

    missing = capabilities.missing_required
    if not missing:
        return
    lines = [f"  - {r.name}: {r.reason or 'unusable'}" for r in missing]
    raise MissingCapability(
        "required Pheasant capabilities are unusable:\n"
        + "\n".join(lines)
        + f"\nserver offers: {', '.join(sorted(capabilities.server_tools)) or '(none)'}"
    )
