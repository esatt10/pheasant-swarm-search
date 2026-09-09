"""The provider interface and the normalised candidate record.

``family_key`` is the field that makes "independent sources" mean something.
Six papers from one lab are one family however many journals they appear in,
and a diversity number computed without it measures productivity.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .. import ids
from ..hashing import digest, digest_text
from ..textkit import normalise

PEER_REVIEWED_TYPES = frozenset({"journal_article", "review", "proceedings", "book_chapter"})

_DOI = re.compile(r"10\.\d{4,9}/[-._;()/:a-z0-9]+", re.IGNORECASE)


class ProviderError(RuntimeError):
    """A provider failed. The caller records it and tries the next one."""

    def __init__(self, provider: str, message: str, *, retryable: bool = True) -> None:
        super().__init__(f"{provider}: {message}")
        self.provider = provider
        self.retryable = retryable


@dataclass
class SourceCandidate:
    """One literature item, normalised.

    Everything a downstream stage needs is here, including the fields the
    provider did **not** report: ``license: "unknown"`` and
    ``abstract: None`` are answers, and defaulting them to something
    convenient is how a corpus ends up with provenance it cannot support.
    """

    provider: str
    title: str
    stable_identifier: str | None = None
    canonical_url: str | None = None
    abstract: str | None = None
    authors: list[str] = field(default_factory=list)
    published_at: str | None = None
    source_type: str = "journal_article"
    venue: str | None = None
    license: str = "unknown"
    open_access: bool = False
    full_text_url: str | None = None
    cited_by_count: int | None = None
    family_key: str | None = None
    referenced_works: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def content_digest(self) -> str:
        return digest_text(self.abstract or self.title)

    @property
    def source_id(self) -> str:
        return ids.source_id(self.stable_identifier, self.canonical_url, self.content_digest)

    @property
    def candidate_id(self) -> str:
        return (
            "cand-" + digest({"p": self.provider, "s": self.source_id}).removeprefix("sha256:")[:16]
        )

    @property
    def peer_reviewed(self) -> bool:
        return self.source_type in PEER_REVIEWED_TYPES

    @property
    def normalised_title(self) -> str:
        return normalise(self.title)

    @property
    def has_text(self) -> bool:
        return bool((self.abstract or "").strip())

    def derived_family(self) -> str:
        """The family this item belongs to, when the provider named none.

        Falls back to the first author's surname, which is a *weaker* key than
        an affiliation and is labelled as derived so a diversity number can
        say how it was computed.
        """

        if self.family_key:
            return self.family_key
        if self.authors:
            return "author:" + normalise(
                self.authors[0].split()[-1] if self.authors[0].split() else self.authors[0]
            )
        return "unknown:" + self.source_id

    def as_record(self, **extra: Any) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "candidate_id": self.candidate_id,
            "provider": self.provider,
            "title": self.title,
            "stable_identifier": self.stable_identifier,
            "canonical_url": self.canonical_url,
            # The abstract is stored so a later command can rebuild the
            # research package without re-fetching: the specialist arm reads
            # what this run collected, not what the provider holds today.
            "abstract": self.abstract,
            "authors": list(self.authors),
            "published_at": self.published_at,
            "source_type": self.source_type,
            "venue": self.venue,
            "license": self.license,
            "open_access": self.open_access,
            "full_text_url": self.full_text_url,
            "cited_by_count": self.cited_by_count,
            "family_key": self.derived_family(),
            "family_key_derived": self.family_key is None,
            "content_digest": self.content_digest,
            "peer_reviewed": self.peer_reviewed,
            "has_text": self.has_text,
            **extra,
        }


class LiteratureProvider(ABC):
    """One index."""

    name = "abstract"
    # A provider that costs money per request declares it; the budget guard
    # reserves before the call rather than discovering the bill afterwards.
    usd_per_request = 0.0

    def __init__(
        self, *, timeout: float = 20.0, contact_email: str | None = None, api_key: str | None = None
    ) -> None:
        self.timeout = timeout
        self.contact_email = contact_email
        self.api_key = api_key

    @abstractmethod
    def search(
        self,
        query: str,
        *,
        limit: int = 20,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[SourceCandidate]:
        """Return candidates for one query, best first."""

    # -- shared -----------------------------------------------------------
    def _client(self) -> Any:
        import httpx

        headers = {"user-agent": self._user_agent()}
        return httpx.Client(timeout=self.timeout, headers=headers, follow_redirects=True)

    def _user_agent(self) -> str:
        contact = f"; mailto:{self.contact_email}" if self.contact_email else ""
        return (
            f"pheasant-swarm-lab/0.1 (https://github.com/esatt10/pheasant-deep-research{contact})"
        )

    @staticmethod
    def clean_doi(value: str | None) -> str | None:
        if not value:
            return None
        match = _DOI.search(value)
        return match.group(0).lower().rstrip(".") if match else None


class ProviderRegistry:
    _providers: dict[str, type[LiteratureProvider]] = {}

    @classmethod
    def register(cls, provider: type[LiteratureProvider]) -> type[LiteratureProvider]:
        cls._providers[provider.name] = provider
        return provider

    @classmethod
    def get(cls, name: str) -> type[LiteratureProvider]:
        _ensure_loaded()
        try:
            return cls._providers[name]
        except KeyError as exc:
            raise ProviderError(
                name, f"unknown provider; known: {sorted(cls._providers)}", retryable=False
            ) from exc

    @classmethod
    def names(cls) -> list[str]:
        _ensure_loaded()
        return sorted(cls._providers)


def _ensure_loaded() -> None:
    if ProviderRegistry._providers:
        return
    from . import arxiv, crossref, fixtures, openalex, pubmed  # noqa: F401


def build_provider(name: str, **kwargs: Any) -> LiteratureProvider:
    return ProviderRegistry.get(name)(**kwargs)


def known_providers() -> list[str]:
    return ProviderRegistry.names()


def deduplicate(
    candidates: Sequence[SourceCandidate],
) -> tuple[list[SourceCandidate], list[tuple[str, str]]]:
    """Collapse equivalents, and say which collapsed into which.

    Equivalence is DOI first, then normalised title - the preprint/version
    pair that shares neither is *not* collapsed here, because merging on a
    similarity threshold is how two genuinely independent replications become
    one source.
    """

    kept: list[SourceCandidate] = []
    duplicates: list[tuple[str, str]] = []
    by_doi: dict[str, SourceCandidate] = {}
    by_title: dict[str, SourceCandidate] = {}

    for candidate in candidates:
        doi = candidate.stable_identifier
        title = candidate.normalised_title
        existing = (by_doi.get(doi) if doi else None) or (by_title.get(title) if title else None)
        if existing is not None:
            duplicates.append((candidate.source_id, existing.source_id))
            continue
        kept.append(candidate)
        if doi:
            by_doi[doi] = candidate
        if title:
            by_title[title] = candidate
    return kept, duplicates


def normalise_record(row: Mapping[str, Any], provider: str) -> SourceCandidate:
    """Build a candidate from an already-normalised mapping (fixtures, replay)."""

    return SourceCandidate(
        provider=provider,
        title=str(row.get("title", "")),
        stable_identifier=row.get("doi") or row.get("stable_identifier"),
        canonical_url=row.get("url") or row.get("canonical_url"),
        abstract=row.get("abstract"),
        authors=[str(a) for a in row.get("authors", []) or []],
        published_at=row.get("published_at") or row.get("date"),
        source_type=str(row.get("source_type", "journal_article")),
        venue=row.get("venue"),
        license=str(row.get("license", "unknown")),
        open_access=bool(row.get("open_access", False)),
        full_text_url=row.get("full_text_url"),
        cited_by_count=row.get("cited_by_count"),
        family_key=row.get("family_key"),
        referenced_works=[str(r) for r in row.get("referenced_works", []) or []],
        raw=dict(row),
    )
