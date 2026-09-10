"""PubMed, through E-utilities.

Two requests per query - ``esearch`` for identifiers, ``efetch`` for records -
because the search endpoint returns no abstracts. Both are counted, so the
per-round provider budget means what it says.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

from .base import LiteratureProvider, ProviderError, ProviderRegistry, SourceCandidate

ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


@ProviderRegistry.register
class PubMedProvider(LiteratureProvider):
    name = "pubmed"

    def search(
        self,
        query: str,
        *,
        limit: int = 20,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[SourceCandidate]:
        params: dict[str, Any] = {
            "db": "pubmed",
            "term": query,
            "retmax": min(100, max(1, limit)),
            "retmode": "json",
            "sort": "relevance",
        }
        if date_from or date_to:
            params["datetype"] = "pdat"
            params["mindate"] = (date_from or "1900/01/01").replace("-", "/")
            params["maxdate"] = (date_to or "2999/12/31").replace("-", "/")
        if self.api_key:
            params["api_key"] = self.api_key
        if self.contact_email:
            params["email"] = self.contact_email

        try:
            with self._client() as client:
                found = client.get(ESEARCH, params=params)
                found.raise_for_status()
                identifiers = ((found.json().get("esearchresult") or {}).get("idlist")) or []
                if not identifiers:
                    return []
                fetch_params: dict[str, Any] = {
                    "db": "pubmed",
                    "id": ",".join(identifiers),
                    "retmode": "xml",
                }
                if self.api_key:
                    fetch_params["api_key"] = self.api_key
                records = client.get(EFETCH, params=fetch_params)
                records.raise_for_status()
                root = ET.fromstring(records.text)
        except Exception as exc:
            raise ProviderError(self.name, str(exc)) from exc

        return [self._candidate(article) for article in root.findall(".//PubmedArticle")][:limit]

    def _candidate(self, article: ET.Element) -> SourceCandidate:
        citation = article.find("MedlineCitation")
        assert citation is not None
        art = citation.find("Article")
        title = (
            "".join((art.find("ArticleTitle") or ET.Element("x")).itertext()).strip()
            if art is not None
            else ""
        )
        abstract_parts = [
            "".join(node.itertext()).strip()
            for node in (art.findall("Abstract/AbstractText") if art is not None else [])
        ]
        authors: list[str] = []
        affiliation = None
        for author in art.findall("AuthorList/Author") if art is not None else []:
            last = author.findtext("LastName")
            first = author.findtext("ForeName")
            if last:
                authors.append(" ".join(part for part in (first, last) if part))
            if affiliation is None:
                affiliation = author.findtext("AffiliationInfo/Affiliation")
        doi = None
        for identifier in article.findall(".//ArticleIdList/ArticleId"):
            if identifier.get("IdType") == "doi" and identifier.text:
                doi = identifier.text
        pmid = citation.findtext("PMID")
        types = (
            {node.text for node in art.findall("PublicationTypeList/PublicationType")}
            if art is not None
            else set()
        )
        year = citation.findtext("Article/Journal/JournalIssue/PubDate/Year")

        return SourceCandidate(
            provider=self.name,
            title=title,
            stable_identifier=self.clean_doi(doi) or (f"pmid:{pmid}" if pmid else None),
            canonical_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
            abstract=" ".join(part for part in abstract_parts if part) or None,
            authors=authors,
            published_at=f"{year}-01-01" if year else None,
            source_type="review" if "Review" in types else "journal_article",
            venue=citation.findtext("Article/Journal/Title"),
            family_key=("affiliation:" + affiliation) if affiliation else None,
            raw={"pmid": pmid, "publication_types": sorted(t for t in types if t)},
        )
