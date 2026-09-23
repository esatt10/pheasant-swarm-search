"""Collection profiles: which evidence a run collects, and what counts as authoritative.

A profile is a named set of *defaults* for the ``collection`` and ``stopping``
sections. It is chosen with ``collection.profile`` and never overrides a value
the experiment file (or ``--set``) states explicitly - the file is the more
specific statement, and a profile that silently won would make the resolved
configuration disagree with the file a reader is looking at.

Three ship:

``scholarly``
    Peer-reviewed and preprint literature from the scholarly indexes. The
    original behaviour, and still the default: a run that names no profile
    resolves to exactly the values it resolved to before profiles existed.

``web``
    Broad web search - company publications, filings, job postings,
    interviews, press, essays and forums. For fields whose evidence is
    practitioner and industry writing rather than papers. "Authoritative"
    here means a *primary* source: the organisation's own statement, a filing
    or a posting, not somebody's account of it.

``balanced``
    Both, with each provider capped per query so the first index in the list
    cannot fill a subtopic before the others are asked. Authoritative means
    peer-reviewed *or* primary.

The profile only chooses defaults. What a run actually did is the resolved
configuration, which is what the manifest digests - so two runs under
different profiles are not comparable, and the digest says so without anyone
having to remember which profile was used.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

ProfileName = Literal["scholarly", "web", "balanced"]
PROFILES: tuple[str, ...] = ("scholarly", "web", "balanced")
DEFAULT_PROFILE: ProfileName = "scholarly"

#: Scholarly source types. The peer-reviewed subset is what the ``scholarly``
#: profile counts as authoritative; preprints and datasets are admitted but
#: do not count toward that minimum.
SCHOLARLY_TYPES: tuple[str, ...] = (
    "journal_article",
    "preprint",
    "review",
    "proceedings",
    "dataset",
)
PEER_REVIEWED_TYPES: tuple[str, ...] = ("journal_article", "review", "proceedings", "book_chapter")

#: Web source types, as :mod:`pheasant_lab.providers.web` classifies them.
#: ``company_publication``, ``filing`` and ``job_posting`` are an
#: organisation speaking for itself; ``interview`` is a practitioner doing so.
WEB_PRIMARY_TYPES: tuple[str, ...] = ("company_publication", "filing", "job_posting", "interview")
WEB_SECONDARY_TYPES: tuple[str, ...] = ("press", "essay")
#: Admitted by ``web`` only: a forum thread or an unclassifiable page can
#: point at evidence, but is weak evidence on its own.
WEB_COMMUNITY_TYPES: tuple[str, ...] = ("forum", "web_page")
WEB_TYPES: tuple[str, ...] = WEB_PRIMARY_TYPES + WEB_SECONDARY_TYPES + WEB_COMMUNITY_TYPES

_PROFILE_DEFAULTS: dict[str, dict[str, dict[str, Any]]] = {
    "scholarly": {
        "collection": {
            "providers": ["openalex", "crossref", "arxiv", "pubmed"],
            "allowed_source_types": list(SCHOLARLY_TYPES),
            "authoritative_source_types": list(PEER_REVIEWED_TYPES),
            "require_stable_identifier": True,
            "max_sources_per_subtopic": 30,
            "max_results_per_provider": None,
        },
        "stopping": {
            "minimum_sources_per_subtopic": 6,
            "minimum_independent_source_families": 3,
            "minimum_review_or_primary_sources": 2,
            "maximum_duplicate_rate": 0.25,
        },
    },
    "web": {
        "collection": {
            "providers": ["brave", "tavily"],
            "allowed_source_types": list(WEB_TYPES),
            "authoritative_source_types": list(WEB_PRIMARY_TYPES),
            # A web provider's identifier is the normalised URL, so this holds
            # for every candidate it returns; kept on so a provider that
            # returns none is refused rather than admitted unaddressable.
            "require_stable_identifier": True,
            "max_research_agents": 8,
            "max_search_rounds_per_agent": 12,
            "max_sources_per_subtopic": 60,
            "max_results_per_provider": 20,
        },
        "stopping": {
            "minimum_sources_per_subtopic": 10,
            # Web results are abundant and one organisation publishes a lot;
            # independence has to be asked for harder, not less.
            "minimum_independent_source_families": 5,
            "minimum_review_or_primary_sources": 3,
            # Syndication and reposts make near-duplicates common on the web.
            "maximum_duplicate_rate": 0.40,
        },
    },
    "balanced": {
        "collection": {
            "providers": ["openalex", "crossref", "brave", "tavily"],
            "allowed_source_types": list(SCHOLARLY_TYPES)
            + list(WEB_PRIMARY_TYPES)
            + list(WEB_SECONDARY_TYPES),
            "authoritative_source_types": list(PEER_REVIEWED_TYPES) + list(WEB_PRIMARY_TYPES),
            "require_stable_identifier": True,
            "max_research_agents": 6,
            "max_search_rounds_per_agent": 10,
            "max_sources_per_subtopic": 40,
            "max_results_per_provider": 8,
        },
        "stopping": {
            "minimum_sources_per_subtopic": 8,
            "minimum_independent_source_families": 4,
            "minimum_review_or_primary_sources": 3,
            "maximum_duplicate_rate": 0.30,
        },
    },
}


def profile_defaults(name: str) -> dict[str, dict[str, Any]]:
    """The defaults one profile supplies, per section. A copy; mutate freely."""

    try:
        table = _PROFILE_DEFAULTS[name]
    except KeyError as exc:
        raise ValueError(f"unknown collection profile '{name}'; known: {list(PROFILES)}") from exc
    return {
        section: {
            key: (list(value) if isinstance(value, list) else value) for key, value in keys.items()
        }
        for section, keys in table.items()
    }


def apply_profile(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Fill a raw experiment tree's unset ``collection``/``stopping`` keys from its profile.

    Only keys the tree does not state are filled: an explicit value always
    wins. Returns a new tree; the input is not mutated.
    """

    tree = dict(raw)
    collection = dict(tree.get("collection") or {})
    name = collection.get("profile") or DEFAULT_PROFILE
    collection["profile"] = name
    defaults = profile_defaults(str(name))
    for key, value in defaults["collection"].items():
        collection.setdefault(key, value)
    tree["collection"] = collection
    stopping = dict(tree.get("stopping") or {})
    for key, value in defaults["stopping"].items():
        stopping.setdefault(key, value)
    tree["stopping"] = stopping
    return tree
