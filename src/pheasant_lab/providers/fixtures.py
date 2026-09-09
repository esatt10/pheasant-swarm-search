"""The offline provider.

Reads a local pack of normalised records and answers queries out of it with
the same interface the live providers use. Two things it must keep true, or
the offline path stops being a test of anything:

* it **ranks**, rather than returning everything - a corpus where every query
  succeeds proves nothing about retrieval;
* it can return **decoys** - records that look relevant and are not - because
  precision has no meaning in a fixture that contains only correct answers.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..textkit import content_terms, tokens
from .base import (
    LiteratureProvider,
    ProviderError,
    ProviderRegistry,
    SourceCandidate,
    normalise_record,
)

ENV_VAR = "PHEASANT_LAB_FIXTURES"
DEFAULT_DIR = Path("tests/fixtures/literature")


@ProviderRegistry.register
class FixtureProvider(LiteratureProvider):
    name = "fixtures"

    def __init__(self, *, directory: str | Path | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.directory = Path(directory or os.environ.get(ENV_VAR) or DEFAULT_DIR)
        self._records: list[dict[str, Any]] | None = None

    def load(self) -> list[dict[str, Any]]:
        if self._records is not None:
            return self._records
        if not self.directory.is_dir():
            raise ProviderError(
                self.name,
                f"fixture directory {self.directory} does not exist; set {ENV_VAR} or pass directory=",
                retryable=False,
            )
        records: list[dict[str, Any]] = []
        for path in sorted(self.directory.rglob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload.get("records", payload) if isinstance(payload, dict) else payload
            for row in rows:
                row.setdefault("_fixture_file", path.name)
                records.append(row)
        self._records = records
        return records

    def search(
        self,
        query: str,
        *,
        limit: int = 20,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[SourceCandidate]:
        terms = set(content_terms(query))
        scored: list[tuple[float, dict[str, Any]]] = []
        for row in self.load():
            if not _in_window(row.get("published_at"), date_from, date_to):
                continue
            haystack = set(
                tokens(
                    f"{row.get('title', '')} {row.get('abstract', '')} {' '.join(row.get('keywords', []) or [])}"
                )
            )
            hits = terms & haystack
            if not hits:
                continue
            # Title hits are worth more, the way any lexical ranker weights
            # them; without that a decoy whose body mentions the term once
            # outranks the paper the term is about.
            title_hits = terms & set(tokens(str(row.get("title", ""))))
            scored.append((len(hits) + 2.0 * len(title_hits), row))
        scored.sort(key=lambda pair: (-pair[0], str(pair[1].get("title", ""))))
        return [normalise_record(row, self.name) for _score, row in scored[:limit]]

    def all_candidates(self) -> list[SourceCandidate]:
        return [normalise_record(row, self.name) for row in self.load()]


def _in_window(published: str | None, date_from: str | None, date_to: str | None) -> bool:
    if not published:
        return True
    if date_from and published < date_from:
        return False
    return not (date_to and published > date_to)


def fixture_records(directory: str | Path | None = None) -> Sequence[dict[str, Any]]:
    return FixtureProvider(directory=directory).load()
