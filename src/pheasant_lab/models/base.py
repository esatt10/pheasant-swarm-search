"""The model interface.

Structured output is required rather than parsed hopefully: every role in this
lab returns a shape another stage consumes, and a role that returns prose
where a schema was expected is a failure to record, not a string to salvage.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..settings import RoleModel

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class StructuredOutputError(ValueError):
    """The provider returned something that is not the requested shape."""


@dataclass
class ModelRequest:
    """One call.

    ``context`` carries the structured inputs a deterministic provider needs
    and a hosted provider mostly ignores (it gets them rendered into ``user``).
    Keeping both means the offline path is a real implementation of the role
    rather than a recorded transcript of one.
    """

    role: str
    schema: str
    system: str
    user: str
    context: dict[str, Any] = field(default_factory=dict)
    max_output_tokens: int = 4000
    temperature: float = 0.0
    seed: int | None = None
    tools: list[dict[str, Any]] = field(default_factory=list)

    @property
    def prompt_text(self) -> str:
        return f"{self.system}\n\n{self.user}"


@dataclass
class ModelResponse:
    data: dict[str, Any]
    text: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    finish_reason: str = "stop"
    deterministic: bool = False
    raw: dict[str, Any] = field(default_factory=dict)


class ModelProvider(ABC):
    """One provider, one role."""

    name = "abstract"

    def __init__(self, *, spec: RoleModel, role: str, api_key: str | None = None) -> None:
        self.spec = spec
        self.role = role
        self.api_key = api_key

    @property
    def model(self) -> str:
        return self.spec.model

    @abstractmethod
    def complete(self, request: ModelRequest) -> ModelResponse:
        """Answer one request, or raise."""

    # -- shared helpers ---------------------------------------------------
    @staticmethod
    def parse_structured(text: str) -> dict[str, Any]:
        """Parse the JSON object a role was asked for.

        Tolerates a fenced block, because every hosted provider emits one
        sometimes; refuses anything else, because a role that answered in
        prose did not answer.
        """

        candidate = text.strip()
        fenced = _FENCE.search(candidate)
        if fenced:
            candidate = fenced.group(1).strip()
        if not candidate:
            raise StructuredOutputError("empty response where a JSON object was required")
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            start, end = candidate.find("{"), candidate.rfind("}")
            if start >= 0 and end > start:
                try:
                    parsed = json.loads(candidate[start : end + 1])
                except json.JSONDecodeError:
                    raise StructuredOutputError(f"response is not JSON: {exc}") from exc
            else:
                raise StructuredOutputError(f"response is not JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise StructuredOutputError(f"response is a {type(parsed).__name__}, not an object")
        return parsed
