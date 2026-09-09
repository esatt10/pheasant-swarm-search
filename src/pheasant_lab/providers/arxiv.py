"""arXiv.

Preprints, and the one provider here whose full text is reliably licensed for
retrieval. Everything it returns is marked ``preprint``: an arXiv posting that
was later published is still, as retrieved from here, the preprint version -
and a stopping rule that counts it as peer-reviewed would be counting the
wrong object.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

from .base import LiteratureProvider, ProviderError, ProviderRegistry, SourceCandidate

BASE_URL = "https://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


@ProviderRegistry.register
class ArxivProvider(LiteratureProvider):
    name = "arxiv"

    def search(
        self,
        query: str,
        *,
        limit: int = 20,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[SourceCandidate]:
        search_query = f"all:{query}"
        if date_from or date_to:
            start = (date_from or "1900-01-01").replace("-", "")
            end = (date_to or "2999-12-31").replace("-", "")
            search_query += f" AND submittedDate:[{start}0000 TO {end}2359]"
        params: dict[str, Any] = {
            "search_query": search_query,
            "max_results": min(100, max(1, limit)),
            "sortBy": "relevance",
        }
        try:
            with self._client() as client:
                response = client.get(BASE_URL, params=params)
                if response.status_code == 429:
                    raise ProviderError(self.name, "rate limited", retryable=True)
                response.raise_for_status()
                root = ET.fromstring(response.text)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(self.name, str(exc)) from exc

        return [self._candidate(entry) for entry in root.findall("atom:entry", NS)][:limit]

    def _candidate(self, entry: ET.Element) -> SourceCandidate:
        def text(path: str) -> str | None:
            node = entry.find(path, NS)
            return node.text.strip() if node is not None and node.text else None

        pdf = None
        for link in entry.findall("atom:link", NS):
            if link.get("title") == "pdf":
                pdf = link.get("href")
        authors = [
            node.text.strip() for node in entry.findall("atom:author/atom:name", NS) if node.text
        ]
        doi = text("arxiv:doi")
        identifier = text("atom:id")
        published = (text("atom:published") or "")[:10] or None
        return SourceCandidate(
            provider=self.name,
            title=" ".join((text("atom:title") or "").split()),
            stable_identifier=self.clean_doi(doi)
            or (identifier.rsplit("/", 1)[-1] if identifier else None),
            canonical_url=identifier,
            abstract=" ".join((text("atom:summary") or "").split()) or None,
            authors=authors,
            published_at=published,
            source_type="preprint",
            venue="arXiv",
            license="arxiv-nonexclusive",
            open_access=True,
            full_text_url=pdf,
            raw={
                "primary_category": (
                    entry.find("arxiv:primary_category", NS) or ET.Element("x")
                ).get("term")
            },
        )
