"""OpenAlex.

Chosen first by default because it is the only free index here that reports
both an open-access status and an institutional affiliation, and affiliation
is what makes ``family_key`` - and therefore "independent sources" - mean
something rather than counting one lab's output as diversity.
"""

from __future__ import annotations

from typing import Any

from .base import LiteratureProvider, ProviderError, ProviderRegistry, SourceCandidate

BASE_URL = "https://api.openalex.org/works"

TYPE_MAP = {
    "article": "journal_article",
    "review": "review",
    "preprint": "preprint",
    "proceedings-article": "proceedings",
    "book-chapter": "book_chapter",
    "dataset": "dataset",
}


@ProviderRegistry.register
class OpenAlexProvider(LiteratureProvider):
    name = "openalex"

    def search(
        self,
        query: str,
        *,
        limit: int = 20,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[SourceCandidate]:
        filters = ["has_abstract:true"]
        if date_from:
            filters.append(f"from_publication_date:{date_from}")
        if date_to:
            filters.append(f"to_publication_date:{date_to}")
        params: dict[str, Any] = {
            "search": query,
            "per-page": min(200, max(1, limit)),
            "filter": ",".join(filters),
        }
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

        return [self._candidate(work) for work in payload.get("results", [])][:limit]

    def _candidate(self, work: dict[str, Any]) -> SourceCandidate:
        location = work.get("primary_location") or {}
        venue = (location.get("source") or {}).get("display_name")
        authorships = work.get("authorships") or []
        authors = [str((a.get("author") or {}).get("display_name") or "") for a in authorships]
        family = None
        for authorship in authorships:
            institutions = authorship.get("institutions") or []
            if institutions:
                family = "ror:" + str(
                    institutions[0].get("id") or institutions[0].get("display_name")
                )
                break
        best_oa = work.get("best_oa_location") or {}
        return SourceCandidate(
            provider=self.name,
            title=str(work.get("display_name") or work.get("title") or ""),
            stable_identifier=self.clean_doi(work.get("doi")),
            canonical_url=str(work.get("doi") or work.get("id") or "") or None,
            abstract=_inverted_index_to_text(work.get("abstract_inverted_index")),
            authors=[a for a in authors if a],
            published_at=work.get("publication_date"),
            source_type=TYPE_MAP.get(str(work.get("type")), "journal_article"),
            venue=venue,
            license=str(best_oa.get("license") or location.get("license") or "unknown"),
            open_access=bool((work.get("open_access") or {}).get("is_oa")),
            full_text_url=best_oa.get("pdf_url"),
            cited_by_count=work.get("cited_by_count"),
            family_key=family,
            referenced_works=[str(r) for r in (work.get("referenced_works") or [])[:50]],
            raw={"id": work.get("id"), "type": work.get("type")},
        )


def _inverted_index_to_text(index: dict[str, list[int]] | None) -> str | None:
    """Reconstruct an abstract from OpenAlex's inverted index.

    A gap in the positions is left as a gap rather than closed: closing it
    silently would fabricate adjacency between two words that were not
    adjacent, and an extraction pass would then read a claim out of it.
    """

    if not index:
        return None
    positions: dict[int, str] = {}
    for word, offsets in index.items():
        for offset in offsets:
            positions[offset] = word
    if not positions:
        return None
    return " ".join(positions.get(i, "…") for i in range(max(positions) + 1))
