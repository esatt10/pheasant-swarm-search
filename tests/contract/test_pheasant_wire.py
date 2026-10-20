"""The adapters against what pheasant actually sends.

Every payload read here was captured from a running pheasant
(``tests/fixtures/pheasant/responses-<version>.json``, written by
``scripts/capture_pheasant_fixtures.py``). The readers are checked against each
supported release, not only the newest: 0.12.16, 0.13.0, 0.13.1 (which
adds search ``expand``), 0.13.2 (which adds ``get_index_queue``), 0.13.4
(whose ``describe_source`` arrived in 0.13.3) and 0.13.11 (whose tools only
gain optional arguments the lab does not send). The last three were captured
from a role-split region with no indexer running, so their syncs - the memory
source's included - answer ``queued``. The mock region is
kept faithful to the same shapes, but a mock is only as faithful as whoever
last checked it - the receipts, the acknowledgement counts, the hit shape and
the nested memory record were each read wrongly here for as long as only the
mock was asked.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from pheasant_lab.pheasant.capabilities import resolve
from pheasant_lab.pheasant.client import PheasantClient
from pheasant_lab.pheasant.ingestion import (
    Ingestor,
    IngestPayload,
    IngestRequest,
    IngestSource,
    RegistrationRefused,
)
from pheasant_lab.pheasant.mock import MockPheasantServer, _Refusal
from pheasant_lab.pheasant.protocol import McpToolError
from pheasant_lab.pheasant.receipts import parse_receipts
from pheasant_lab.pheasant.retrieval import Retriever, SearchRequest, normalise_results

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "pheasant"

# pheasant/memory/steering.py's preference grammar, verbatim. A record it does
# not match is ignored by the region without a word.
PHEASANT_PREFERENCE = re.compile(
    r"\bwhen\s*:\s*(?P<when>[^\n]*?)\s*(?:->|→|=>)\s*prefer\s*:\s*(?P<prefer>[^\n]+)", re.I
)


#: Every capture the readers must understand, oldest first.
RELEASES = ("0.12.16", "0.13.0", "0.13.1", "0.13.2", "0.13.4", "0.13.11")


def captured(version: str) -> dict:
    return json.loads((FIXTURES / f"responses-{version}.json").read_text())


@pytest.fixture(scope="module", params=RELEASES)
def wire(request) -> dict:
    return captured(request.param)


def search_request() -> SearchRequest:
    return SearchRequest(
        run_id="run-1", arm_id="P0", question_id="q-1", query="dsup", namespace="pheasant-lab"
    )


# -- receipts --------------------------------------------------------------


def test_real_receipts_are_read_from_the_accepted_and_rejected_lists(wire):
    payload = wire["submit_documents"]
    accepted = payload["accepted"][0]
    receipts = parse_receipts(
        payload,
        run_id="run-1",
        key_to_source={
            "pheasant-lab:source-aaa:1": "source-aaa",
            "pheasant-lab:source-bbb:2": "source-bbb",
        },
        requested_digests={"pheasant-lab:source-aaa:1": "sha256:" + accepted["content_sha256"]},
    )
    by_key = {r.idempotency_key: r for r in receipts}
    assert set(by_key) == {"pheasant-lab:source-aaa:1", "pheasant-lab:source-bbb:2"}

    good = by_key["pheasant-lab:source-aaa:1"]
    assert good.accepted and not good.indexed
    assert good.artifact_id == accepted["artifact_id"]
    # Verified against the digest the region computed, not the `content_digest`
    # this lab put in the metadata and the region echoed back in `detail`.
    assert good.accepted_content_digest == "sha256:" + accepted["content_sha256"]
    assert good.digest_matches is True

    bad = by_key["pheasant-lab:source-bbb:2"]
    assert bad.status == "rejected" and bad.accepted is False
    assert bad.error_code == "INVALID_REQUEST"
    assert bad.error_message and "empty" in bad.error_message


# -- search hits -----------------------------------------------------------


def test_a_real_hit_resolves_to_the_labs_source_not_the_regions(wire):
    hit = wire["search_context"]["results"][0]
    results = normalise_results(
        wire["search_context"],
        search_request(),
        artifact_sources={hit["node_id"]: "source-from-receipts"},
    )
    first = results[0]
    assert first.artifact_id == hit["node_id"]
    assert first.source_id == "source-from-receipts"
    assert first.region_source == hit["provenance"]["source_id"]
    assert first.locator == hit["relative_path"]
    assert first.retrieval_arm == hit["retrieved_by"]
    assert first.matched_text, "the preview is the passage until it is hydrated"


def test_without_receipts_a_hit_falls_back_to_the_front_matter_not_the_region(wire):
    results = normalise_results(wire["search_context"], search_request())
    region_sources = {
        name
        for r in wire["search_context"]["results"]
        for name in (r.get("source_id"), (r.get("provenance") or {}).get("source_id"))
        if name
    }
    for result in results:
        assert result.source_id not in region_sources, (
            "the region's source name read as the lab's collapses every hit onto one source"
        )
        # A graph-arm node hit has no front matter to read: absent, not guessed.
        assert result.source_id is None or result.source_id.startswith("source-")
    assert any(result.source_id for result in results)


def test_a_graph_node_hit_names_the_regions_source_from_its_top_level():
    """pheasant's graph arm puts the source on the hit, not under provenance.

    A node hit's provenance is the node's own and need not name a source, so
    reading only ``provenance.source_id`` fetched it back with no
    ``source_name``. Present in 0.12.16 too; the 0.13.0 capture is the first to
    rank one into the result list.
    """

    payload = captured("0.13.0")["search_context"]
    node = next(hit for hit in payload["results"] if hit.get("kind") == "node")
    assert "source_id" not in node["provenance"]
    results = normalise_results(payload, search_request())
    result = next(r for r in results if r.artifact_id == node["node_id"])
    assert result.region_source == node["source_id"]
    assert result.retrieval_arm == "graph"
    assert result.source_id != node["source_id"]
    assert result.graph_neighbors == []


# -- graph expansion ---------------------------------------------------------


def test_an_expanded_hit_keeps_its_neighbourhood():
    payload = captured("0.13.1")["search_context_expanded"]
    results = normalise_results(payload, search_request())
    assert results and all(result.graph_neighbors for result in results)
    for hit, result in zip(payload["results"], results, strict=True):
        assert [n["node_id"] for n in result.graph_neighbors] == [
            n["node_id"] for n in hit["graph"]["neighbors"]
        ]
        for neighbor in result.graph_neighbors:
            assert {"node_id", "type", "depth", "edge_types", "via"} <= set(neighbor)
            assert neighbor["depth"] in (1, 2)
    # Expansion adds structure and changes nothing it was added to.
    plain = normalise_results(captured("0.13.1")["search_context"], search_request())
    assert [r.artifact_id for r in plain] == [r.artifact_id for r in results]
    assert all(r.graph_neighbors == [] for r in plain)


def test_the_mock_expands_in_pheasants_shape(config):
    """The mock has no graph, so its neighbourhoods are empty - in the real shape."""

    real = captured("0.13.1")["search_context_expanded"]
    server = MockPheasantServer()
    ingestor = ingestor_for(config, server)
    ingestor.submit([request_for("dsup shields chromatin")])
    ingestor.sync()
    expand = {"depth": 2, "max_neighbors": 6}
    mocked = server._tool_search_context(
        {"knowledge_base": "pheasant-lab", "query": "dsup", "memory": "off", "expand": expand}
    )
    assert set(mocked["expansion"]) == set(real["expansion"])
    assert {
        k: mocked["expansion"][k] for k in ("depth", "max_neighbors", "exclude_edge_types")
    } == {k: real["expansion"][k] for k in ("depth", "max_neighbors", "exclude_edge_types")}
    assert set(mocked["results"][0]["graph"]) == set(real["results"][0]["graph"])
    unexpanded = server._tool_search_context({"knowledge_base": "pheasant-lab", "query": "dsup"})
    assert "expansion" not in unexpanded and "graph" not in unexpanded["results"][0]


@pytest.mark.parametrize(
    ("value", "refusal"),
    [
        ({"depth": 4}, "expand depth must be an integer from 1 to 3"),
        ({"hops": 2}, "expand does not take hops"),
        ({"max_neighbors": 0}, "expand.max_neighbors must be an integer from 1 to 50"),
        ("yes", "expand must be true, a depth"),
    ],
)
def test_the_mock_refuses_a_malformed_expansion_as_pheasant_does(config, value, refusal):
    server = MockPheasantServer()
    with pytest.raises(_Refusal, match=re.escape(refusal)):
        server._tool_search_context(
            {"knowledge_base": "pheasant-lab", "query": "x", "expand": value}
        )


# -- memory ----------------------------------------------------------------


def test_the_record_id_is_read_from_pheasants_nested_record(wire, config):
    from pheasant_lab.arms.pheasant_memory import MemorySeeder

    payload = wire["memory_write"]
    client = SimpleNamespace(
        call=lambda *a, **k: SimpleNamespace(result=SimpleNamespace(payload=lambda: payload))
    )
    capabilities = SimpleNamespace(has=lambda name: True, tool=lambda name: "memory_write")
    seeder = MemorySeeder(client, capabilities, config.pheasant)
    written = seeder._write({"text": "x", "kind": "fact", "scope": "org"}, "q-1")
    assert written is not None
    assert written["record_id"] == payload["record"]["record_id"]


def _answering(payloads: dict):
    """A client whose every call answers with the captured payload for its tool."""

    return SimpleNamespace(
        call=lambda tool, *a, **k: SimpleNamespace(
            result=SimpleNamespace(payload=lambda: payloads[tool])
        )
    )


def test_a_queued_memory_sync_is_remembered_and_awaited():
    """0.13.2 on a fleet: the record is stored, and its sync is only published."""

    from pheasant_lab.arms.pheasant_memory import MemorySeeder
    from pheasant_lab.settings import PheasantFile

    wire = captured("0.13.2")
    queued = wire["memory_write"]["sync"]
    assert queued["status"] == "queued" and queued["source_id"] == "agent-memory"
    listing = json.loads(json.dumps(wire["get_index_queue"]))
    payloads = {"memory_write": wire["memory_write"], "get_index_queue": listing}
    capabilities = SimpleNamespace(
        has=lambda name: True,
        tool=lambda name: {"write_memory": "memory_write", "index_queue": "get_index_queue"}[name],
    )
    pheasant = PheasantFile(transport="mock", knowledge_base="pheasant-lab")
    seeder = MemorySeeder(_answering(payloads), capabilities, pheasant, clock=FakeClock())
    seeder._write({"text": "x", "kind": "fact", "scope": "org"}, "q-1")
    assert seeder.queued_tasks == {queued["task_id"]}

    # Still listed: the wait runs out and says so rather than calling it indexed.
    outcome = seeder.wait_until_indexed(wait_seconds=1.0)
    assert outcome["outcome"] == "timed_out"
    assert outcome["outstanding"] == [queued["task_id"]]
    assert seeder.limitation is None, "only seed() records the outcome P1 is reported with"
    seeder.index_outcome = outcome
    assert "not confirmed" in seeder.limitation

    # The indexer ran it: the task leaves the listing, and only that is "indexed".
    listing["tasks"] = [t for t in listing["tasks"] if t["source"] != "agent-memory"]
    assert seeder.wait_until_indexed()["outcome"] == "indexed"

    # A dead-lettered task never leaves the listing; it ends the wait at once.
    listing["tasks"] = [{**wire["get_index_queue"]["tasks"][0], "state": "dead"}]
    assert seeder.wait_until_indexed()["outcome"] == "failed"

    # A region that cannot list is an unknown, never "nothing outstanding".
    listing["listing"] = "unavailable"
    assert seeder.wait_until_indexed()["outcome"] == "unknown"


def test_a_standalone_memory_sync_needs_no_wait(config):
    from pheasant_lab.arms.pheasant_memory import MemorySeeder

    payload = captured("0.13.1")["memory_write"]
    capabilities = SimpleNamespace(has=lambda name: True, tool=lambda name: "memory_write")
    seeder = MemorySeeder(_answering({"memory_write": payload}), capabilities, config.pheasant)
    seeder._write({"text": "x", "kind": "fact", "scope": "org"}, "q-1")
    assert seeder.queued_tasks == set()


def test_the_mock_holds_a_queued_memory_record_back_until_it_is_indexed(config):
    from pheasant_lab.arms.pheasant_memory import MemorySeeder

    server = MockPheasantServer(claim_seconds=0.05)
    client = PheasantClient.in_process(config.pheasant, server)
    capabilities = resolve(config.pheasant, client.connect().tools)
    seeder = MemorySeeder(client, capabilities, config.pheasant)
    written = seeder._write({"text": "dsup binds nucleosomes", "kind": "fact"}, "q-1")
    assert written is not None and len(seeder.queued_tasks) == 1
    assert server._memory_hits("dsup nucleosomes", as_of=None, include_rules=False) == []

    outcome = seeder.wait_until_indexed(wait_seconds=10.0)
    assert outcome["outcome"] == "indexed"
    assert server._memory_hits("dsup nucleosomes", as_of=None, include_rules=False)


@pytest.mark.parametrize("version", ["0.13.2", "0.13.4", "0.13.11"])
def test_the_mock_queues_in_pheasants_fleet_shape(version):
    wire = captured(version)
    server = MockPheasantServer(claim_seconds=60)
    server.sources["fixture-wire"] = "/state/uploads/x"
    sync = server._tool_sync_source({"source_name": "fixture-wire"})
    assert sync.keys() == wire["sync_source_queued"].keys()
    memory = server._tool_memory_write({"text": "fixture fact", "scope": "org", "sync": True})
    assert memory.keys() == wire["memory_write"].keys()
    assert memory["sync"].keys() == wire["memory_write"]["sync"].keys()
    listing = server._tool_get_index_queue({})
    assert listing["listing"] == wire["get_index_queue"]["listing"]
    for task in listing["tasks"]:
        missing = set(wire["get_index_queue"]["tasks"][0]) - set(task)
        assert not missing, f"the mock's task lacks {sorted(missing)}"
    assert {t["source"] for t in listing["tasks"]} == {"fixture-wire", "agent-memory"}


def test_the_ingest_barrier_reads_only_its_own_source_from_the_queue(config):
    wire = captured("0.13.2")
    capabilities = SimpleNamespace(has=lambda name: True, tool=lambda name: "get_index_queue")
    from pheasant_lab.pheasant.ingestion import read_index_queue

    client = _answering({"get_index_queue": wire["get_index_queue"]})
    (task,) = read_index_queue(client, capabilities, {}, source="fixture-wire")
    assert task["task_id"] == wire["sync_source_queued"]["task_id"]
    assert task["state"] == "awaiting_claim" and task["position"] == 2


def test_seeded_preferences_parse_under_pheasants_rule_grammar(config):
    from pheasant_lab.arms.pheasant_memory import MemorySeeder

    seeder = MemorySeeder(None, None, config.pheasant)
    artifact = "file:src:topic/a.md:branch=none"
    passage = "Dsup binds nucleosomes and shields chromatin from hydroxyl radicals."
    entry = {
        "question_id": "q-1",
        "question_text": "How does Dsup protect chromatin from radiation?",
        "answer_text": passage,
        "claims": [{"text": passage, "citations": [artifact]}],
        "read_passages": [{"artifact_id": artifact, "text": passage}],
        "queries_used": ["dsup"],
        "retrieved_artifact_ids": [artifact],
        "search_calls": [
            {"query": "dsup", "round": 1, "results": [{"artifact_id": artifact, "rank": 1}]}
        ],
    }
    preferences = [c for c in seeder._candidates(entry) if c["kind"] == "preference"]
    assert len(preferences) == 1
    text = preferences[0]["text"]
    matches = list(PHEASANT_PREFERENCE.finditer(text))
    assert matches, f"pheasant would ignore {text!r}"
    assert len(matches) == 1, "the explanation must not parse as a second rule"
    assert matches[0].group("prefer").strip() == "topic/a.md"


# -- the index barrier, through the faithful mock ---------------------------


def request_for(text: str, source_id: str = "source-1") -> IngestRequest:
    return IngestRequest(
        run_id="run-1",
        topic_id="topic-1",
        source_id=source_id,
        namespace="pheasant-lab",
        source=IngestSource(title="t", stable_identifier=f"10.1/{source_id}"),
        payload=IngestPayload(text=text),
    )


def ingestor_for(config, server, *, drop: tuple[str, ...] = ()) -> Ingestor:
    client = PheasantClient.in_process(config.pheasant, server)
    session = client.connect()
    tools = {name: tool for name, tool in session.tools.items() if name not in drop}
    return Ingestor(client, resolve(config.pheasant, tools), config.pheasant, run_id="run-1")


def test_a_submission_is_unsearchable_until_its_directory_is_registered(config):
    server = MockPheasantServer()
    ingestor = ingestor_for(config, server, drop=("register_source",))
    ingestor.submit([request_for("body about dsup")])
    with pytest.raises(McpToolError, match="Unknown source"):
        ingestor.sync()


def test_sync_registers_where_the_region_landed_the_submission(config):
    server = MockPheasantServer()
    ingestor = ingestor_for(config, server)
    ingestor.submit([request_for("body about dsup")])
    ingestor.sync()
    assert server.sources == {config.pheasant.source_name: ingestor.landing_directory}
    ingestor.acknowledge()
    assert all(receipt.indexed for receipt in ingestor.ledger.receipts)


def test_a_refused_registration_names_the_setting_that_fixes_it(config):
    class Strict(MockPheasantServer):
        def _tool_register_source(self, arguments):
            raise _Refusal("Path is outside the allowed workspace roots")

    ingestor = ingestor_for(config, Strict())
    ingestor.submit([request_for("body")])
    with pytest.raises(RegistrationRefused, match="allow_workspace_roots"):
        ingestor.sync()


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def test_the_barrier_waits_for_an_indexer_that_runs_elsewhere(config):
    """On a fleet, `sync_source` publishes; the indexer catches up later."""

    class Fleet(MockPheasantServer):
        lag = 2

        def _tool_sync_source(self, arguments):
            return {"source_id": arguments.get("source_name"), "status": "queued"}

        def _tool_acknowledge_ingest(self, arguments):
            self.lag -= 1
            if self.lag <= 0:
                for document in self.documents.values():
                    document.indexed = True
            return super()._tool_acknowledge_ingest(arguments)

    server = Fleet()
    server.sources[config.pheasant.source_name] = "/state/uploads/x"
    ingestor = ingestor_for(config, server)
    ingestor._clock = FakeClock()
    ingestor.submit([request_for("body")])
    ingestor.sync()
    result = ingestor.acknowledge()
    assert result["still_accepted"] == 0
    assert ingestor._clock.slept, (
        "the barrier polled instead of declaring an unindexed item indexed"
    )
    assert all(receipt.indexed for receipt in ingestor.ledger.receipts)


def test_a_barrier_that_never_clears_says_so(config):
    class Stuck(MockPheasantServer):
        def _tool_sync_source(self, arguments):
            return {"status": "queued"}

    server = Stuck()
    server.sources[config.pheasant.source_name] = "/state/uploads/x"
    ingestor = ingestor_for(config, server)
    ingestor._clock = FakeClock()
    ingestor.submit([request_for("body")])
    ingestor.sync()
    result = ingestor.acknowledge(wait_seconds=1.0)
    assert result["barrier"] == "timed_out"
    assert not any(receipt.indexed for receipt in ingestor.ledger.receipts)


# -- hydration -------------------------------------------------------------


def test_passages_are_read_back_whole_rather_than_as_previews(config):
    from pheasant_lab.arms.base import passages_from

    server = MockPheasantServer()
    ingestor = ingestor_for(config, server)
    body = "dsup " + "chromatin shielding " * 60 + "the decisive sentence is at the end."
    ingestor.submit([request_for(body)])
    ingestor.sync()

    client = PheasantClient.in_process(config.pheasant, server)
    retriever = Retriever(client, resolve(config.pheasant, client.connect().tools), config.pheasant)
    response = retriever.search(search_request())
    passages = passages_from([response], limit=5)
    assert "decisive sentence" not in passages[0]["text"], "the fixture must exceed the preview"
    retriever.hydrate(passages)
    assert passages[0]["text_source"] == "document"
    assert passages[0]["text"] == body


def test_a_graph_relationship_hit_is_hydrated_from_the_labs_own_submission(config):
    """Pheasant's graph arm returns `X —mentions→ Y` hits with no path at all."""

    from pheasant_lab.arms.base import passages_from

    server = MockPheasantServer()
    ingestor = ingestor_for(config, server)
    request = request_for("dsup shields chromatin, the whole abstract")
    ingestor.submit([request])
    ingestor.sync()
    artifact_id = ingestor.ledger.receipts[0].artifact_id

    client = PheasantClient.in_process(config.pheasant, server)
    retriever = Retriever(client, resolve(config.pheasant, client.connect().tools), config.pheasant)
    relationship = {
        "kind": "relationship",
        "node_id": artifact_id,
        "type": "relationship",
        "edge_type": "mentions",
        "title": f"{request.path()} —mentions→ Chromatin",
        "summary": f"{request.path()} mentions Chromatin",
        "score": 0.016,
        "rank": 1,
        "retrieved_by": "graph",
    }
    response = SimpleNamespace(
        results=normalise_results({"results": [relationship]}, search_request())
    )
    passages = passages_from([response], limit=5)
    assert passages[0]["locator"] is None
    retriever.hydrate(passages)
    assert passages[0]["text_source"] == "preview", "no path, and nothing told it one"

    retriever.artifact_paths = {artifact_id: request.path()}
    passages = passages_from([response], limit=5)
    retriever.hydrate(passages)
    assert passages[0]["text_source"] == "document"
    assert "the whole abstract" in passages[0]["text"]


# -- the region's own inventory (0.13.3+) -----------------------------------


def _inventory_ingestor(config, payload, *, indexed: int, total: int) -> Ingestor:
    from pheasant_lab.pheasant.receipts import IngestReceipt

    capabilities = SimpleNamespace(has=lambda name: True, tool=lambda name: "describe_source")
    client = SimpleNamespace(
        call=lambda tool, *a, **k: SimpleNamespace(
            result=SimpleNamespace(payload=lambda: payload, is_error=False)
        )
    )
    ingestor = Ingestor(client, capabilities, config.pheasant, run_id="run-1")
    for index in range(total):
        ingestor.ledger.record(
            IngestReceipt(
                receipt_id=f"r-{index}",
                run_id="run-1",
                source_id=f"source-{index}",
                idempotency_key=f"key-{index}",
                submission_id=None,
                status="indexed" if index < indexed else "accepted",
            )
        )
    return ingestor


@pytest.mark.parametrize("version", ["0.13.4", "0.13.11"])
def test_the_regions_inventory_is_read_beside_the_receipts(config, version):
    """``describe_source`` (0.13.3): a count read off the region's own index."""

    payload = captured(version)["describe_source"]
    held = payload["totals"]["documents"]
    ingestor = _inventory_ingestor(config, payload, indexed=held, total=held)
    summary = ingestor.inventory()
    assert summary["disposition"] == "consistent"
    assert summary["region_documents"] == held == summary["receipts_indexed"]
    assert summary["region_bytes"] == payload["totals"]["size_bytes"]


@pytest.mark.parametrize("version", ["0.13.4", "0.13.11"])
def test_a_region_holding_more_than_the_receipts_say_is_a_mismatch(config, version):
    payload = captured(version)["describe_source"]
    held = payload["totals"]["documents"]
    ingestor = _inventory_ingestor(config, payload, indexed=held - 2, total=held)
    assert ingestor.inventory()["disposition"] == "mismatch"


def test_an_inventory_the_region_cannot_give_is_unknown_not_a_finding(config):
    capabilities = SimpleNamespace(has=lambda name: False, tool=lambda name: "")
    ingestor = Ingestor(SimpleNamespace(), capabilities, config.pheasant, run_id="run-1")
    assert ingestor.inventory() is None


@pytest.mark.parametrize("version", ["0.13.4", "0.13.11"])
def test_the_mock_describes_a_source_in_pheasants_shape(config, version):
    real = captured(version)["describe_source"]
    server = MockPheasantServer()
    ingestor = ingestor_for(config, server)
    ingestor.submit([request_for("body about dsup")])
    ingestor.sync()
    ingestor.acknowledge()
    mock = server._tool_describe_source({"source_name": config.pheasant.source_name})
    assert mock.keys() == real.keys()
    assert mock["totals"].keys() == real["totals"].keys()
    assert mock["source"].keys() == real["source"].keys()
    assert ingestor.inventory()["disposition"] == "consistent"
