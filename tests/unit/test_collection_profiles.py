"""Collection profiles, web classification, and the web providers - offline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from pheasant_lab.orchestration.state import CollectionState, RoundRecord, SourceRecord, SourceState
from pheasant_lab.orchestration.stopping import StoppingCalculus
from pheasant_lab.profiles import PEER_REVIEWED_TYPES, WEB_PRIMARY_TYPES, profile_defaults
from pheasant_lab.providers.base import ProviderError, SourceCandidate, deduplicate
from pheasant_lab.providers.web import (
    BraveProvider,
    TavilyProvider,
    classify,
    family_for,
    normalise_url,
    registrable_domain,
)
from pheasant_lab.settings import (
    CollectionSection,
    ConfigError,
    Facet,
    StoppingSection,
    Topic,
    load_config,
)

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / "configs" / "experiment.example.yaml"


def _load(profile: str | None = None, **overrides: str):
    environ = {"COLLECTION_PROFILE": profile} if profile else {}
    return load_config(
        EXAMPLE, env_file=None, environ=environ, overrides=overrides, project_root=REPO
    )


# ---------------------------------------------------------------------------
# profile resolution
# ---------------------------------------------------------------------------


def test_default_profile_resolves_to_what_the_example_resolved_to_before_profiles():
    # The values experiment.example.yaml stated explicitly before profiles
    # existed. A run naming no profile must resolve to exactly these.
    config = _load()
    assert config.collection.profile == "scholarly"
    assert config.collection.providers == ["openalex", "crossref", "arxiv", "pubmed"]
    assert config.collection.allowed_source_types == [
        "journal_article",
        "preprint",
        "review",
        "proceedings",
        "dataset",
    ]
    assert config.collection.require_stable_identifier is True
    assert config.collection.max_research_agents == 6
    assert config.collection.max_search_rounds_per_agent == 8
    assert config.collection.max_sources_per_subtopic == 30
    assert config.collection.max_results_per_provider is None
    assert config.stopping.minimum_sources_per_subtopic == 6
    assert config.stopping.minimum_independent_source_families == 3
    assert config.stopping.minimum_review_or_primary_sources == 2
    assert config.stopping.maximum_duplicate_rate == 0.25


@pytest.mark.parametrize("profile", ["scholarly", "web", "balanced"])
def test_each_profile_supplies_its_defaults(profile):
    config = _load(profile)
    expected = profile_defaults(profile)
    for key, value in expected["collection"].items():
        assert getattr(config.collection, key) == value, key
    for key, value in expected["stopping"].items():
        assert getattr(config.stopping, key) == value, key


def test_web_profile_admits_no_scholarly_types_and_counts_primary_sources():
    config = _load("web")
    assert "journal_article" not in config.collection.allowed_source_types
    assert set(config.collection.authoritative_source_types) == set(WEB_PRIMARY_TYPES)


def test_balanced_profile_caps_each_provider_and_mixes_indexes():
    config = _load("balanced")
    providers = config.collection.providers
    assert {"openalex", "brave"} <= set(providers)
    assert config.collection.max_results_per_provider is not None
    assert config.collection.max_results_per_provider < config.collection.max_sources_per_subtopic
    assert set(PEER_REVIEWED_TYPES) & set(config.collection.authoritative_source_types)
    assert set(WEB_PRIMARY_TYPES) <= set(config.collection.authoritative_source_types)


def test_an_explicit_value_beats_the_profile():
    config = _load(
        "web",
        **{"collection.providers": "[tavily]", "stopping.minimum_sources_per_subtopic": "4"},
    )
    assert config.collection.providers == ["tavily"]
    assert config.stopping.minimum_sources_per_subtopic == 4
    # ...and the keys not stated still come from the profile.
    assert config.collection.max_results_per_provider == 20


def test_set_override_selects_the_profile():
    assert _load(**{"collection.profile": "balanced"}).collection.profile == "balanced"


def test_profiles_are_not_comparable_runs():
    assert _load("scholarly").digest() != _load("web").digest()


def test_unknown_profile_is_refused():
    with pytest.raises((ConfigError, ValueError), match=r"unknown collection\.profile"):
        _load("everything")


def test_authoritative_types_none_of_which_are_admitted_is_refused():
    with pytest.raises(ValueError, match="could never be met"):
        CollectionSection(
            allowed_source_types=["press"], authoritative_source_types=["journal_article"]
        )


# ---------------------------------------------------------------------------
# the stopping calculus counts what the profile calls authoritative
# ---------------------------------------------------------------------------

TOPIC = Topic(id="t", title="t", facets=[Facet(id="f1", label="one", weight=1.0)])
STOPPING = StoppingSection(
    minimum_sources_per_subtopic=2,
    minimum_independent_source_families=2,
    minimum_review_or_primary_sources=1,
    marginal_unique_claim_window=1,
    marginal_unique_claim_threshold=0.5,
    consecutive_saturated_rounds=1,
)


def _web_state(types: list[str]) -> CollectionState:
    state = CollectionState("run-1", TOPIC.id)
    for index, source_type in enumerate(types):
        candidate = SourceCandidate(
            provider="brave",
            title=f"page {index}",
            stable_identifier=f"url:https://org{index}.example/{index}",
            abstract="body",
            family_key=f"site:org{index}.example",
            source_type=source_type,
        )
        state.add_source(
            SourceRecord(
                source_id=candidate.source_id,
                candidate=candidate,
                subtopic_id="sub-1",
                facet_ids=["f1"],
                state=SourceState.indexed,
                researcher_agent_id="agent-1",
                discovery_event_id="event-1",
            )
        )
    state.rounds.append(
        RoundRecord(
            number=1,
            subtopics=["sub-1"],
            new_eligible_claims=0,
            eligible_claims_before=1,
            acquired=len(types),
            duplicates=0,
            cost_usd=0.0,
        )
    )
    return state


def test_press_alone_is_not_authoritative_under_the_web_profile():
    calculus = StoppingCalculus(STOPPING, TOPIC, authoritative_types=WEB_PRIMARY_TYPES)
    _coverage, rows = calculus.facet_coverage(_web_state(["press", "essay"]))
    assert rows[0]["authoritative"] == 0
    assert rows[0]["meets_minimum"] is False
    assert any(item.startswith("authoritative 0<1") for item in rows[0]["unmet"])


def test_a_filing_satisfies_the_web_profile_minimum():
    calculus = StoppingCalculus(STOPPING, TOPIC, authoritative_types=WEB_PRIMARY_TYPES)
    _coverage, rows = calculus.facet_coverage(_web_state(["press", "filing"]))
    assert rows[0]["authoritative"] == 1
    assert rows[0]["meets_minimum"] is True


def test_the_default_calculus_still_means_peer_reviewed():
    calculus = StoppingCalculus(STOPPING, TOPIC)
    _coverage, rows = calculus.facet_coverage(_web_state(["filing", "company_publication"]))
    assert rows[0]["authoritative"] == 0


# ---------------------------------------------------------------------------
# classification and families
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://www.sec.gov/Archives/edgar/data/1321655/10-k.htm", "filing"),
        ("https://investors.example.com/annual-report-2025", "filing"),
        ("https://boards.greenhouse.io/acme/jobs/12345", "job_posting"),
        ("https://acme.com/careers/forward-deployed-engineer", "job_posting"),
        ("https://www.youtube.com/watch?v=abc", "interview"),
        ("https://acme.com/podcast/episode-12", "interview"),
        ("https://news.ycombinator.com/item?id=1", "forum"),
        ("https://techcrunch.com/2025/01/01/fde", "press"),
        ("https://someone.substack.com/p/the-fde", "essay"),
        ("https://acme.com/blog/how-we-deploy", "company_publication"),
        ("https://acme.com/engineering/field-teams", "company_publication"),
        ("https://acme.com/", "web_page"),
    ],
)
def test_classify(url, expected):
    source_type, rule = classify(url)
    assert source_type == expected
    assert rule


def test_registrable_domain_handles_country_code_suffixes():
    assert registrable_domain("eng.example.co.uk") == "example.co.uk"
    assert registrable_domain("blog.example.com") == "example.com"
    assert registrable_domain("example.com") == "example.com"


def test_one_organisation_is_one_family_across_subdomains():
    assert family_for("https://eng.acme.com/a") == family_for("https://www.acme.com/blog/b")


def test_a_hosting_platform_is_not_the_author():
    assert family_for("https://medium.com/@alice/post") != family_for(
        "https://medium.com/@bob/post"
    )
    assert family_for("https://alice.substack.com/p/x") != family_for(
        "https://bob.substack.com/p/y"
    )


def test_url_normalisation_collapses_one_page():
    a = normalise_url("https://www.Acme.com/blog/post/?utm_source=x#section")
    b = normalise_url("https://acme.com/blog/post")
    assert a == b


# ---------------------------------------------------------------------------
# providers, against a stubbed client - no network
# ---------------------------------------------------------------------------


class _Response:
    def __init__(self, payload: dict[str, Any], status: int = 200) -> None:
        self._payload = payload
        self.status_code = status

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _Client:
    def __init__(self, response: _Response) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def __enter__(self) -> _Client:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def get(self, url: str, **kwargs: Any) -> _Response:
        self.calls.append({"method": "GET", "url": url, **kwargs})
        return self.response

    def post(self, url: str, **kwargs: Any) -> _Response:
        self.calls.append({"method": "POST", "url": url, **kwargs})
        return self.response


def _stub(monkeypatch, provider, payload, status=200) -> _Client:
    client = _Client(_Response(payload, status))
    monkeypatch.setattr(provider, "_client", lambda: client)
    return client


def test_a_web_provider_without_a_key_refuses_to_build():
    with pytest.raises(ProviderError, match="BRAVE_SEARCH_API_KEY"):
        BraveProvider(api_key=None)
    with pytest.raises(ProviderError, match="TAVILY_API_KEY"):
        TavilyProvider(api_key="")


def test_brave_results_are_classified_keyed_and_windowed(monkeypatch):
    provider = BraveProvider(api_key="k")
    client = _stub(
        monkeypatch,
        provider,
        {
            "web": {
                "results": [
                    {
                        "title": "Forward deployed <strong>engineering</strong>",
                        "url": "https://acme.com/blog/fde?utm_source=x",
                        "description": "How our <strong>FDE</strong> team works",
                        "extra_snippets": ["Embedded with customers."],
                        "page_age": "2025-03-04T00:00:00",
                    },
                    {
                        "title": "Old post",
                        "url": "https://acme.com/blog/old",
                        "description": "x",
                        "page_age": "2012-01-01T00:00:00",
                    },
                    {"title": "no url"},
                ]
            }
        },
    )
    found = provider.search("forward deployed engineer", limit=5, date_from="2015-01-01")
    assert len(found) == 1
    hit = found[0]
    assert hit.title == "Forward deployed engineering"
    assert hit.source_type == "company_publication"
    assert hit.family_key == "site:acme.com"
    assert hit.stable_identifier == "url:https://acme.com/blog/fde"
    assert hit.published_at == "2025-03-04"
    assert hit.abstract == "How our FDE team works Embedded with customers."
    assert hit.license == "unknown"
    assert client.calls[0]["headers"]["X-Subscription-Token"] == "k"
    assert client.calls[0]["params"]["freshness"].startswith("2015-01-01to")


def test_tavily_sends_the_window_and_keeps_undated_pages(monkeypatch):
    provider = TavilyProvider(api_key="k")
    client = _stub(
        monkeypatch,
        provider,
        {
            "results": [
                {
                    "title": "FDE roles",
                    "url": "https://jobs.lever.co/acme/123",
                    "content": "Forward Deployed Engineer",
                },
            ]
        },
    )
    found = provider.search("fde", limit=3, date_from="2020-01-01", date_to="2026-01-01")
    assert [c.source_type for c in found] == ["job_posting"]
    assert found[0].published_at is None
    body = client.calls[0]["json"]
    assert body["start_date"] == "2020-01-01" and body["end_date"] == "2026-01-01"
    assert body["max_results"] == 3


def test_a_rejected_key_is_not_retryable(monkeypatch):
    provider = BraveProvider(api_key="bad")
    _stub(monkeypatch, provider, {}, status=401)
    with pytest.raises(ProviderError) as info:
        provider.search("x")
    assert info.value.retryable is False


def test_two_providers_returning_one_page_deduplicate():
    a = SourceCandidate(provider="brave", title="A", stable_identifier="url:https://acme.com/p")
    b = SourceCandidate(
        provider="tavily", title="A (copy)", stable_identifier="url:https://acme.com/p"
    )
    kept, duplicates = deduplicate([a, b])
    assert kept == [a] and len(duplicates) == 1


# ---------------------------------------------------------------------------
# the per-provider cap reaches every provider
# ---------------------------------------------------------------------------


class _CountingProvider:
    usd_per_request = 0.0

    def __init__(self, name: str) -> None:
        self.name = name
        self.limits: list[int] = []

    def search(self, query, *, limit=20, date_from=None, date_to=None):
        self.limits.append(limit)
        return [
            SourceCandidate(
                provider=self.name,
                title=f"{self.name} {query} {index}",
                stable_identifier=f"url:https://{self.name}.example/{query}/{index}",
                abstract="body",
            )
            for index in range(limit)
        ]


class _Tracer:
    from contextlib import contextmanager

    trace_id = span_id = None

    @contextmanager
    def span(self, *args, **kwargs):
        yield

    @contextmanager
    def operation(self, *args, **kwargs):
        yield {}


def _discover(config, providers):
    from pheasant_lab.orchestration.planner import Subtopic
    from pheasant_lab.orchestration.researcher import BranchResult, Researcher

    researcher = Researcher.__new__(Researcher)
    researcher.config = config
    researcher.topic = config.topics[0]
    researcher.providers = providers
    researcher.ledger = None
    researcher.tracer = _Tracer()
    subtopic = Subtopic(subtopic_id="s", facet_ids=["f"], question="q", terminology=[])
    return researcher.search_candidates(
        subtopic, BranchResult(subtopic_id="s", agent_id="a"), max_rounds=1, max_sources=20
    )


def test_without_a_cap_the_first_provider_fills_the_subtopic():
    first, second = _CountingProvider("one"), _CountingProvider("two")
    found = _discover(_load("scholarly"), [first, second])
    assert {c.provider for c in found} == {"one"}
    assert second.limits == []


def test_the_balanced_cap_asks_every_provider():
    config = _load("balanced", **{"collection.max_results_per_provider": "8"})
    first, second = _CountingProvider("one"), _CountingProvider("two")
    found = _discover(config, [first, second])
    assert first.limits == [8] and second.limits == [8]
    assert {c.provider for c in found} == {"one", "two"}
