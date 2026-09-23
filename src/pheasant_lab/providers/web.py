"""Web search providers, for the ``web`` and ``balanced`` collection profiles.

A scholarly index tells you what a record *is*. A web search result does not:
a URL and a snippet arrive with no type, no author and no affiliation. So this
module has two jobs beyond calling an API.

**Classification.** Every result is given a ``source_type`` from its URL
alone - a filing, a job posting, a company's own publication, an interview,
press, an essay, a forum thread, or ``web_page`` when nothing matched. It is a
heuristic and says so: the rule that fired is kept in ``raw["classified_by"]``
so a reader can see why a page counted as primary, and ``web_page`` is an
honest "could not tell" rather than a guess.

**Families.** Independence on the web is organisational: ten posts on one
company's blog are one family. ``family_key`` is the registrable domain, and
a hosted platform (Substack, Medium, a job board, YouTube) is keyed one level
deeper, because the platform is not the author.

The identifier is the normalised URL, so two results for one page collapse,
and ``require_stable_identifier`` means something here as well.

Snippets only: the text retained is what the search API returned, never a
fetched page. Licence is ``unknown`` - which is the truth - so acquisition
stays abstract-only, as it does for every other provider.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .base import LiteratureProvider, ProviderError, ProviderRegistry, SourceCandidate

# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------

#: Hosts whose path's first segment names the author, not the host.
_HOSTED_PLATFORMS = {
    "medium.com",
    "github.com",
    "linkedin.com",
    "youtube.com",
    "x.com",
    "twitter.com",
    "boards.greenhouse.io",
    "job-boards.greenhouse.io",
    "jobs.lever.co",
    "jobs.ashbyhq.com",
    "apply.workable.com",
}
#: Second-level labels that sit under a country code (``co.uk``, ``com.au``).
_COMPOUND_SLDS = {"co", "com", "ac", "gov", "org", "net", "edu"}

_FILING_HOSTS = {"sec.gov", "annualreports.com", "companieshouse.gov.uk"}
_JOB_HOSTS = {
    "greenhouse.io",
    "lever.co",
    "ashbyhq.com",
    "workable.com",
    "wellfound.com",
    "workatastartup.com",
    "indeed.com",
    "glassdoor.com",
}
_INTERVIEW_HOSTS = {
    "youtube.com",
    "youtu.be",
    "podcasts.apple.com",
    "open.spotify.com",
    "vimeo.com",
}
_FORUM_HOSTS = {
    "news.ycombinator.com",
    "reddit.com",
    "stackoverflow.com",
    "quora.com",
    "teamblind.com",
}
_ESSAY_HOSTS = {"substack.com", "medium.com", "linkedin.com"}
_PRESS_HOSTS = {
    "techcrunch.com",
    "theinformation.com",
    "bloomberg.com",
    "reuters.com",
    "wsj.com",
    "ft.com",
    "nytimes.com",
    "forbes.com",
    "businessinsider.com",
    "theverge.com",
    "wired.com",
    "fortune.com",
    "cnbc.com",
    "axios.com",
    "venturebeat.com",
    "hbr.org",
}

_JOB_PATH = re.compile(r"/(careers?|jobs?|positions?|openings|join-us|job)(/|$)", re.IGNORECASE)
_INTERVIEW_PATH = re.compile(r"/(podcasts?|episodes?|interviews?|watch)(/|$)", re.IGNORECASE)
_COMPANY_PATH = re.compile(
    r"/(blog|news|newsroom|press|engineering|customers?|case-stud(y|ies)|resources|insights|docs)(/|$)",
    re.IGNORECASE,
)
_FILING_PATH = re.compile(
    r"(10-?k|10-?q|annual-report|investor-relations|/investors?/)", re.IGNORECASE
)


def registrable_domain(host: str) -> str:
    """``eng.example.co.uk`` -> ``example.co.uk``; ``www.example.com`` -> ``example.com``.

    A small approximation of the public-suffix list, sufficient for keying
    families. Where it errs, it errs toward *merging* two sites, which
    understates independence rather than inflating it.
    """

    labels = [label for label in host.lower().strip(".").split(".") if label]
    if len(labels) <= 2:
        return ".".join(labels)
    if len(labels[-1]) == 2 and labels[-2] in _COMPOUND_SLDS:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def _host(url: str) -> str:
    return (urlsplit(url).hostname or "").lower().removeprefix("www.")


def normalise_url(url: str) -> str:
    """Drop the fragment, tracking parameters, ``www.`` and a trailing slash."""

    parts = urlsplit(url.strip())
    query = "&".join(
        pair
        for pair in parts.query.split("&")
        if pair and not pair.lower().startswith(("utm_", "ref=", "fbclid=", "gclid="))
    )
    host = (parts.hostname or "").lower().removeprefix("www.")
    port = f":{parts.port}" if parts.port else ""
    path = parts.path.rstrip("/") or ""
    return urlunsplit(((parts.scheme or "https").lower(), host + port, path, query, ""))


def family_for(url: str) -> str:
    host = _host(url)
    domain = registrable_domain(host)
    if host.endswith(".substack.com"):
        return "site:" + host
    if host in _HOSTED_PLATFORMS or domain in _HOSTED_PLATFORMS:
        segments = [s for s in urlsplit(url).path.split("/") if s]
        if segments:
            return f"site:{host}/{segments[0].lower()}"
    return "site:" + domain


def classify(url: str) -> tuple[str, str]:
    """``(source_type, rule)`` for one URL. The rule is kept for the reader."""

    host = _host(url)
    domain = registrable_domain(host)
    path = urlsplit(url).path

    def on(hosts: set[str]) -> bool:
        return host in hosts or domain in hosts

    if on(_FILING_HOSTS) or (_FILING_PATH.search(path) and not on(_PRESS_HOSTS)):
        return "filing", "filing host or investor-relations path"
    if on(_JOB_HOSTS) or host.startswith(("jobs.", "careers.")) or _JOB_PATH.search(path):
        return "job_posting", "job board host or careers path"
    if on(_INTERVIEW_HOSTS) or host.startswith("podcast") or _INTERVIEW_PATH.search(path):
        return "interview", "audio/video host or podcast path"
    if on(_FORUM_HOSTS):
        return "forum", "forum host"
    if on(_PRESS_HOSTS):
        return "press", "press host"
    if on(_ESSAY_HOSTS) or host.endswith(".substack.com"):
        return "essay", "essay platform"
    if _COMPANY_PATH.search(path):
        return "company_publication", "organisation's own blog/news/docs path"
    return "web_page", "no rule matched"


def _date(value: Any) -> str | None:
    """An ISO date from whatever the API sent, or ``None`` - never a guess."""

    if not value:
        return None
    match = re.match(r"(\d{4}-\d{2}-\d{2})", str(value))
    return match.group(1) if match else None


def web_candidate(
    provider: str,
    *,
    url: str,
    title: str,
    text: str | None,
    published_at: str | None,
    raw: dict[str, Any],
) -> SourceCandidate:
    source_type, rule = classify(url)
    canonical = normalise_url(url)
    return SourceCandidate(
        provider=provider,
        title=title.strip(),
        stable_identifier="url:" + canonical,
        canonical_url=canonical,
        abstract=(text or "").strip() or None,
        published_at=published_at,
        source_type=source_type,
        venue=_host(url) or None,
        license="unknown",
        family_key=family_for(url),
        raw={**raw, "classified_by": rule},
    )


# ---------------------------------------------------------------------------
# providers
# ---------------------------------------------------------------------------


class WebSearchProvider(LiteratureProvider):
    """Shared request handling. A subclass supplies the call and the parse."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not self.api_key:
            raise ProviderError(
                self.name,
                f"no API key; set {self.api_key_env}",
                retryable=False,
            )

    def _check(self, response: Any) -> dict[str, Any]:
        if response.status_code == 429:
            raise ProviderError(self.name, "rate limited", retryable=True)
        if response.status_code in (401, 403):
            raise ProviderError(self.name, "key rejected", retryable=False)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _in_window(published: str | None, date_from: str | None, date_to: str | None) -> bool:
        # An undated page is kept: most of the web is undated, and dropping it
        # would make "in window" mean "happened to carry a date".
        if not published:
            return True
        if date_from and published < date_from:
            return False
        return not (date_to and published > date_to)


@ProviderRegistry.register
class BraveProvider(WebSearchProvider):
    """Brave Search API (``https://api.search.brave.com``)."""

    name = "brave"
    api_key_env = "BRAVE_SEARCH_API_KEY"
    # Brave's published paid tier, per request. An estimate the ledger
    # reserves against; check it against your plan.
    usd_per_request = 0.005
    BASE_URL = "https://api.search.brave.com/res/v1/web/search"

    def search(
        self,
        query: str,
        *,
        limit: int = 20,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[SourceCandidate]:
        params: dict[str, Any] = {"q": query, "count": min(20, max(1, limit)), "extra_snippets": 1}
        if date_from:
            params["freshness"] = f"{date_from}to{date_to or date.today().isoformat()}"
        try:
            with self._client() as client:
                response = client.get(
                    self.BASE_URL,
                    params=params,
                    headers={
                        "X-Subscription-Token": str(self.api_key),
                        "accept": "application/json",
                    },
                )
                payload = self._check(response)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(self.name, str(exc)) from exc

        found: list[SourceCandidate] = []
        for item in (payload.get("web") or {}).get("results") or []:
            url = item.get("url")
            if not url:
                continue
            published = _date(item.get("page_age"))
            if not self._in_window(published, date_from, date_to):
                continue
            snippets = [item.get("description") or "", *(item.get("extra_snippets") or [])]
            found.append(
                web_candidate(
                    self.name,
                    url=url,
                    title=_strip_tags(item.get("title") or ""),
                    text=_strip_tags(" ".join(s for s in snippets if s)),
                    published_at=published,
                    raw={"age": item.get("age")},
                )
            )
        return found[:limit]


@ProviderRegistry.register
class TavilyProvider(WebSearchProvider):
    """Tavily search API (``https://api.tavily.com``)."""

    name = "tavily"
    api_key_env = "TAVILY_API_KEY"
    # An advanced search is two credits; this is that at the pay-as-you-go
    # rate. An estimate the ledger reserves against; check it against your plan.
    usd_per_request = 0.016
    BASE_URL = "https://api.tavily.com/search"

    def search(
        self,
        query: str,
        *,
        limit: int = 20,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[SourceCandidate]:
        body: dict[str, Any] = {
            "query": query,
            "max_results": min(20, max(1, limit)),
            "search_depth": "advanced",
            "include_answer": False,
        }
        if date_from:
            body["start_date"] = date_from
        if date_to:
            body["end_date"] = date_to
        try:
            with self._client() as client:
                response = client.post(
                    self.BASE_URL,
                    json=body,
                    headers={"authorization": f"Bearer {self.api_key}"},
                )
                payload = self._check(response)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(self.name, str(exc)) from exc

        found: list[SourceCandidate] = []
        for item in payload.get("results") or []:
            url = item.get("url")
            if not url:
                continue
            published = _date(item.get("published_date"))
            if not self._in_window(published, date_from, date_to):
                continue
            found.append(
                web_candidate(
                    self.name,
                    url=url,
                    title=item.get("title") or "",
                    text=item.get("content"),
                    published_at=published,
                    raw={"score": item.get("score")},
                )
            )
        return found[:limit]


_TAGS = re.compile(r"<[^>]+>")


def _strip_tags(text: str) -> str:
    return _TAGS.sub("", text).strip()
