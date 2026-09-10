"""Crossref.

The registration agency, so it is the authority on what a DOI *is* even when
it has no abstract to give - which is why an abstract-free hit here is still
worth keeping as metadata rather than discarded as empty.
"""

from __future__ import annotations

import re
from typing import Any

from .base import LiteratureProvider, ProviderError, ProviderRegistry, SourceCandidate

BASE_URL = "https://api.crossref.org/works"
_JATS = re.compile(r"<[^>]+>")

TYPE_MAP = {
    "journal-article": "journal_article",
    "posted-content": "preprint",
    "proceedings-article": "proceedings",
    "book-chapter": "book_chapter",
    "dataset": "dataset",
}


@ProviderRegistry.register
class CrossrefProvider(LiteratureProvider):
    name = "crossref"

    def search(
        self,
        query: str,
        *,
        limit: int = 20,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[SourceCandidate]:
        params: dict[str, Any] = {"query.bibliographic": query, "rows": min(100, max(1, limit))}
        filters = []
        if date_from:
            filters.append(f"from-pub-date:{date_from}")
        if date_to:
            filters.append(f"until-pub-date:{date_to}")
        if filters:
            params["filter"] = ",".join(filters)
        if self.contact_email:
            params["mailto"] = self.contact_email

        try:
            with self._client() as client:
                response = client.get(BASE_URL, params=params)
                if response.status_code == 429:
                    raise ProviderError(self.name, "rate limited", retryable=True)
                response.raise_for_status()
                payload = response.json()
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(self.name, str(exc)) from exc

        items = (payload.get("message") or {}).get("items") or []
        return [self._candidate(item) for item in items][:limit]

    def _candidate(self, item: dict[str, Any]) -> SourceCandidate:
        titles = item.get("title") or []
        authors = [
            " ".join(part for part in (a.get("given"), a.get("family")) if part)
            for a in item.get("author") or []
        ]
        affiliations = [
            aff.get("name")
            for a in item.get("author") or []
            for aff in (a.get("affiliation") or [])
            if aff.get("name")
        ]
        licenses = item.get("license") or []
        return SourceCandidate(
            provider=self.name,
            title=str(titles[0]) if titles else "",
            stable_identifier=self.clean_doi(item.get("DOI")),
            canonical_url=item.get("URL"),
            abstract=_strip_jats(item.get("abstract")),
            authors=[a for a in authors if a],
            published_at=_issued(item),
            source_type=TYPE_MAP.get(str(item.get("type")), "journal_article"),
            venue=(item.get("container-title") or [None])[0],
            license=str(licenses[0].get("URL")) if licenses else "unknown",
            cited_by_count=item.get("is-referenced-by-count"),
            family_key=("affiliation:" + str(affiliations[0])) if affiliations else None,
            raw={"type": item.get("type"), "publisher": item.get("publisher")},
        )


def _strip_jats(abstract: str | None) -> str | None:
    if not abstract:
        return None
    return _JATS.sub(" ", abstract).replace("  ", " ").strip() or None


def _issued(item: dict[str, Any]) -> str | None:
    parts = ((item.get("issued") or {}).get("date-parts") or [[]])[0]
    if not parts:
        return None
    padded = list(parts) + [1] * (3 - len(parts))
    return f"{padded[0]:04d}-{padded[1]:02d}-{padded[2]:02d}"
