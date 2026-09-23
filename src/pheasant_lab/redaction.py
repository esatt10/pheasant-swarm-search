"""Secret redaction.

Secrets are resolved at runtime and replaced with a **stable** token in every
trace, transcript and report. Stable matters: a reader has to be able to tell
"this call used the same credential as that one" from a trace that contains
neither.

The registry is populated from the environment at settings-resolution time,
so a value that never appears in a config file - an authorization header the
transport adds, an API key an SDK reads directly - is still redacted when it
reaches a payload.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping
from typing import Any

from .hashing import digest, short

# Environment variables whose *values* are secret wherever they appear.
SECRET_ENV_PATTERNS = (
    re.compile(r"(?i).*(api[_-]?key|token|secret|password|passwd|credential|authorization).*"),
)

FORBIDDEN_HEADERS = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "cookie",
        "set-cookie",
        "x-api-key",
        "api-key",
        "x-subscription-token",
    }
)

# Shapes worth catching even when the value never passed through the registry.
_INLINE_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}\b"),
    re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{16,}\b"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{12,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
)

_MIN_SECRET_LENGTH = 8


class Redactor:
    """Replaces known secret values with ``[redacted:<name>:<digest8>]``."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._values: dict[str, str] = {}

    # -- registry ---------------------------------------------------------
    def register(self, name: str, value: str | None) -> None:
        if not value or len(value) < _MIN_SECRET_LENGTH:
            return
        self._values[value] = f"[redacted:{name}:{short(digest(value), 8)}]"

    def register_environment(self, environ: Mapping[str, str] | None = None) -> None:
        env = os.environ if environ is None else environ
        for name, value in env.items():
            if any(pattern.match(name) for pattern in SECRET_ENV_PATTERNS):
                self.register(name, value)

    @property
    def known(self) -> int:
        return len(self._values)

    # -- application ------------------------------------------------------
    def text(self, value: str) -> str:
        if not self.enabled or not value:
            return value
        out = value
        for secret, token in self._values.items():
            if secret in out:
                out = out.replace(secret, token)
        for pattern in _INLINE_PATTERNS:
            out = pattern.sub("[redacted:pattern]", out)
        return out

    def headers(self, headers: Mapping[str, str]) -> dict[str, str]:
        """Forbidden headers are dropped, not masked.

        A masked ``authorization`` still tells a reader the scheme and the
        length. There is no diagnostic worth that.
        """

        return {
            key: self.text(value)
            for key, value in headers.items()
            if key.lower() not in FORBIDDEN_HEADERS
        }

    def payload(self, value: Any) -> Any:
        if not self.enabled:
            return value
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, Mapping):
            return {
                key: (
                    "[redacted:header]"
                    if str(key).lower() in FORBIDDEN_HEADERS
                    else self.payload(item)
                )
                for key, item in value.items()
            }
        if isinstance(value, list | tuple):
            return [self.payload(item) for item in value]
        return value

    def scrub_exception(self, exc: BaseException) -> str:
        return self.text(f"{type(exc).__name__}: {exc}")


def hash_principal(principal: str | None, salt: str) -> str | None:
    """Pseudonymise a principal identifier, stably within one run."""

    if principal is None:
        return None
    return "principal-" + short(digest({"p": principal, "s": salt}), 16)


def redact_all(redactor: Redactor, payloads: Iterable[Any]) -> list[Any]:
    return [redactor.payload(item) for item in payloads]
