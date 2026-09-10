"""Canonical serialisation and content digests.

Every id in this lab is content-addressed, and two processes that hold the
same object must produce the same digest without coordinating. That needs one
canonical spelling of a payload, which is what this module is: sorted keys,
no insignificant whitespace, a fixed float repr, and instants normalised to
UTC with a trailing ``Z``.

The rule this module exists to enforce: **a digest must be a function of
exactly the thing it names.** A measured field (a latency, a wall-clock stamp,
a probe outcome) inside a digested payload makes an id move when nothing that
the id is about has changed, and every consumer downstream then believes the
region changed shape.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

DIGEST_PREFIX = "sha256:"
_CHUNK = 1 << 20


def _normalise(value: Any) -> Any:
    """Return ``value`` in the one spelling this module digests."""

    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ValueError(f"non-finite float is not canonically representable: {value!r}")
        # An integral float folds to an int, so a payload that made a JSON
        # round trip digests the same as the one that did not. Everything else
        # is left alone: repr round-trips exactly for IEEE-754 doubles in
        # CPython and is stable across platforms, where a fixed-precision
        # format is not.
        if value == int(value) and abs(value) < 1e15:
            return int(value)
        return value
    if isinstance(value, datetime):
        stamp = value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
        return stamp.isoformat(timespec="microseconds").replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Enum):
        return _normalise(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return DIGEST_PREFIX + hashlib.sha256(value).hexdigest()
    if isinstance(value, Mapping):
        return {str(k): _normalise(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, set | frozenset):
        return sorted((_normalise(v) for v in value), key=lambda v: json.dumps(v, sort_keys=True))
    if isinstance(value, Sequence):
        return [_normalise(v) for v in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _normalise(model_dump(mode="python"))
    raise TypeError(f"{type(value).__name__} has no canonical spelling; convert it first")


def canonical_json(payload: Any) -> str:
    """Serialise ``payload`` to the one string this module will digest."""

    return json.dumps(
        _normalise(payload),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def digest(payload: Any) -> str:
    """Content digest of ``payload``, prefixed with its algorithm."""

    return DIGEST_PREFIX + hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def digest_bytes(data: bytes) -> str:
    return DIGEST_PREFIX + hashlib.sha256(data).hexdigest()


def digest_text(text: str) -> str:
    return digest_bytes(text.encode("utf-8"))


def digest_file(path: str | Path) -> str:
    """Streaming digest, so a downloaded paper never has to fit in memory."""

    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(_CHUNK):
            hasher.update(chunk)
    return DIGEST_PREFIX + hasher.hexdigest()


def short(value: str, length: int = 12) -> str:
    """The digest tail used inside human-facing ids."""

    return value.removeprefix(DIGEST_PREFIX)[:length]


def digest_files(paths: Mapping[str, str | Path]) -> dict[str, str]:
    """Digest a named set of files, for a manifest."""

    return {name: digest_file(path) for name, path in sorted(paths.items())}
