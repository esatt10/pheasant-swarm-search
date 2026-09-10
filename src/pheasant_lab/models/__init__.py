"""Model access, behind one interface.

Three providers ship: ``replay`` (deterministic, offline, free), ``openai``
and ``anthropic``. Everything above this package speaks :class:`ModelRequest`
and :class:`ModelResponse`, so switching provider changes a manifest field and
nothing else - which is what makes "the model is fixed in the run manifest" a
property rather than a promise.
"""

from __future__ import annotations

from ..settings import RoleModel
from .base import ModelProvider, ModelRequest, ModelResponse, StructuredOutputError

_REGISTRY: dict[str, type[ModelProvider]] = {}


def register(name: str, provider: type[ModelProvider]) -> None:
    _REGISTRY[name] = provider


def build(spec: RoleModel, *, role: str, api_key: str | None = None) -> ModelProvider:
    """Construct the provider a role's configuration names."""

    _ensure_defaults()
    try:
        factory = _REGISTRY[spec.provider]
    except KeyError as exc:
        raise ValueError(
            f"unknown model provider '{spec.provider}' for role '{role}'; known: {sorted(_REGISTRY)}"
        ) from exc
    return factory(spec=spec, role=role, api_key=api_key)


def known_providers() -> list[str]:
    _ensure_defaults()
    return sorted(_REGISTRY)


def _ensure_defaults() -> None:
    if _REGISTRY:
        return
    from .anthropic import AnthropicProvider
    from .openai import OpenAIProvider
    from .replay import ReplayProvider

    register("replay", ReplayProvider)
    register("openai", OpenAIProvider)
    register("anthropic", AnthropicProvider)


__all__ = [
    "ModelProvider",
    "ModelRequest",
    "ModelResponse",
    "StructuredOutputError",
    "build",
    "known_providers",
    "register",
]
