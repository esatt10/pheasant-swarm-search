"""Anthropic provider, over the Messages API.

Same posture as the OpenAI adapter: the HTTP surface directly, so the lab's
retry policy, budget reservation and error ledger are the only ones in play.
"""

from __future__ import annotations

import os
from typing import Any

from .base import ModelProvider, ModelRequest, ModelResponse

DEFAULT_BASE_URL = "https://api.anthropic.com/v1"
API_VERSION = "2023-06-01"


class AnthropicProvider(ModelProvider):
    name = "anthropic"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.base_url = os.environ.get("ANTHROPIC_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
        self.api_key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")

    def complete(self, request: ModelRequest) -> ModelResponse:
        import httpx

        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set; doctor should have refused this run")

        body: dict[str, Any] = {
            "model": self.model,
            "system": request.system,
            "max_tokens": request.max_output_tokens,
            "temperature": request.temperature,
            "messages": [
                {"role": "user", "content": request.user},
                # Prefilling the opening brace is how this API is asked for a
                # bare JSON object; the response is completed from there, so
                # the brace is put back before parsing.
                {"role": "assistant", "content": "{"},
            ],
        }
        if self.spec.reasoning_effort in {"high", "medium"}:
            budget = 8000 if self.spec.reasoning_effort == "high" else 4000
            body["thinking"] = {"type": "enabled", "budget_tokens": budget}
            body["max_tokens"] = max(body["max_tokens"], budget + 1024)
            body.pop("temperature", None)
            body["messages"] = body["messages"][:1]

        with httpx.Client(timeout=httpx.Timeout(180.0, connect=15.0)) as client:
            response = client.post(
                f"{self.base_url}/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": API_VERSION,
                    "content-type": "application/json",
                },
                json=body,
            )
            response.raise_for_status()
            payload = response.json()

        text = "".join(
            str(block.get("text", ""))
            for block in payload.get("content", [])
            if block.get("type") == "text"
        )
        if "thinking" not in body and not text.lstrip().startswith("{"):
            text = "{" + text
        usage = payload.get("usage") or {}
        return ModelResponse(
            data=self.parse_structured(text),
            text=text,
            provider=self.name,
            model=self.model,
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            finish_reason=str(payload.get("stop_reason") or "end_turn"),
            raw={"id": payload.get("id"), "model": payload.get("model")},
        )
