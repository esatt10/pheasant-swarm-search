"""Retry policy and the ingest contract.

The rule under test: **only idempotent operations, or operations carrying a
verified idempotency key, are retried.** And a transport success with no
receipt is not an ingest.
"""

from __future__ import annotations

import pytest

from pheasant_lab.pheasant.client import PheasantClient, RateLimited
from pheasant_lab.pheasant.ingestion import (
    Ingestor,
    IngestPayload,
    IngestRequest,
    IngestSource,
)
from pheasant_lab.pheasant.mock import FaultPlan, MockPheasantServer
from pheasant_lab.pheasant.receipts import ReceiptLedger, parse_receipts


class FakeClock:
    """A clock the retry policy can be measured against without waiting."""

    def __init__(self) -> None:
        self.slept: list[float] = []
        self._now = 0.0

    def monotonic(self) -> float:
        self._now += 0.001
        return self._now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)


def client_for(config, server, **kwargs):
    return PheasantClient.in_process(config.pheasant, server, **kwargs)


def test_a_read_only_tool_is_retried_and_every_attempt_is_recorded(config, tracer):
    server = MockPheasantServer(faults=FaultPlan(fail_first={"search_context": 1}))
    clock = FakeClock()
    client = client_for(config, server, tracer=tracer, clock=clock)
    client.connect()
    outcome = client.call("search_context", {"knowledge_base": "pheasant-lab", "query": "dsup"})
    assert outcome.status == "succeeded"
    assert outcome.attempts == 2
    assert clock.slept, "a retry with no backoff is a retry storm"
    assert any(record.resolution == "retried" for record in tracer.errors.records)


def test_a_write_without_an_idempotency_key_is_not_retried(config, tracer):
    server = MockPheasantServer(faults=FaultPlan(fail_first={"memory_write": 1}))
    clock = FakeClock()
    client = client_for(config, server, tracer=tracer, clock=clock)
    client.connect()
    with pytest.raises(TimeoutError):
        client.call(
            "memory_write",
            {"knowledge_base": "pheasant-lab", "text": "x"},
            idempotent=False,
        )
    assert clock.slept == [], "a non-idempotent write must not be replayed"


def test_a_write_carrying_a_key_is_retried(config, tracer):
    server = MockPheasantServer(faults=FaultPlan(fail_first={"submit_documents": 1}))
    clock = FakeClock()
    client = client_for(config, server, tracer=tracer, clock=clock)
    client.connect()
    outcome = client.call(
        "submit_documents",
        {"knowledge_base": "pheasant-lab", "documents": []},
        idempotency_key="submission-1",
    )
    assert outcome.attempts == 2 and outcome.status == "succeeded"


def test_retry_after_is_honoured_over_the_configured_backoff(config, tracer):
    server = MockPheasantServer(
        faults=FaultPlan(
            fail_first={"search_context": 1}, fail_with={"search_context": "rate_limit"}
        )
    )
    clock = FakeClock()
    config.pheasant.retry_backoff_seconds = 30.0
    client = client_for(config, server, tracer=tracer, clock=clock)
    client.connect()
    client.call("search_context", {"knowledge_base": "pheasant-lab", "query": "x"})
    assert clock.slept == [0.0], "Retry-After must win over the client's own backoff"


def test_retries_are_bounded_and_the_last_failure_is_raised(config, tracer):
    server = MockPheasantServer(
        faults=FaultPlan(always_fail={"search_context"}, fail_with={"search_context": "rate_limit"})
    )
    clock = FakeClock()
    client = client_for(config, server, tracer=tracer, clock=clock)
    client.connect()
    with pytest.raises(RateLimited):
        client.call("search_context", {"knowledge_base": "pheasant-lab", "query": "x"})
    assert len(clock.slept) == config.pheasant.max_retries - 1


def test_a_partial_response_is_partial_not_succeeded(config, tracer):
    server = MockPheasantServer(faults=FaultPlan(partial={"search_context"}))
    client = client_for(config, server, tracer=tracer)
    client.connect()
    outcome = client.call("search_context", {"knowledge_base": "pheasant-lab", "query": "x"})
    assert outcome.status == "partial"
    assert outcome.partial is True
    assert outcome.warnings


# -- ingest ----------------------------------------------------------------


def request_for(text: str, source_id: str = "source-1", region_source: str = "") -> IngestRequest:
    return IngestRequest(
        run_id="run-1",
        topic_id="topic-1",
        source_id=source_id,
        namespace="pheasant-lab",
        # The identifier follows the source id: two sources sharing a DOI are
        # one source, and a fixture that pretends otherwise would land two
        # records on one artifact path.
        source=IngestSource(title="t", stable_identifier=f"10.1/{source_id}"),
        payload=IngestPayload(text=text),
        region_source=region_source,
    )


def ingestor_for(config, server, tracer=None, pheasant=None) -> Ingestor:
    from pheasant_lab.pheasant.capabilities import resolve

    pheasant = pheasant or config.pheasant
    client = PheasantClient.in_process(pheasant, server, tracer=tracer)
    session = client.connect()
    return Ingestor(
        client,
        resolve(pheasant, session.tools),
        pheasant,
        run_id="run-1",
        tracer=tracer,
    )


def test_a_submission_produces_one_receipt_per_item(config, mock_server, tracer):
    ingestor = ingestor_for(config, mock_server, tracer)
    receipts = ingestor.submit([request_for("body one"), request_for("body two", "source-2")])
    assert len(receipts) == 2
    assert all(receipt.accepted for receipt in receipts)
    assert all(receipt.artifact_id for receipt in receipts)


def test_acceptance_is_not_searchability(config, mock_server, tracer):
    ingestor = ingestor_for(config, mock_server, tracer)
    receipts = ingestor.submit([request_for("body")])
    assert receipts[0].accepted is True
    assert receipts[0].indexed is False, "an accepted document is not yet retrievable"
    ingestor.sync()
    ingestor.acknowledge()
    assert ingestor.ledger.get(receipts[0].idempotency_key).indexed is True


def test_a_retry_under_a_seen_key_folds_rather_than_duplicating(config, mock_server, tracer):
    ingestor = ingestor_for(config, mock_server, tracer)
    request = request_for("body")
    ingestor.submit([request])
    ingestor.submit([request])
    receipt = ingestor.ledger.get(request.idempotency_key)
    assert receipt.submissions == 2, "the caller retried twice"
    assert receipt.deduplicated is True
    assert len(mock_server.documents) == 1, "the region holds one copy"


@pytest.mark.parametrize("older_region", [True, False])
def test_a_second_source_on_one_region_does_not_inherit_the_first_ones_barrier(
    config, tracer, older_region
):
    """Before 0.14.0 pheasant folded a key it had seen onto its receipt and
    kept `indexed`, even when the resubmission landed in another source. With
    a key that did not name the region source, a second run into a new source
    on the same region crossed its barrier at once and its arms searched an
    empty source. Found against a real 0.13.11 fleet. 0.14.0 resets such a
    receipt; the lab's key is scoped either way, for the regions before it."""

    mock_server = MockPheasantServer(keep_indexed_on_move=older_region)
    first = ingestor_for(config, mock_server, tracer)
    earlier = request_for("body", region_source=config.pheasant.source_name)
    first.submit([earlier])
    first.sync()
    first.acknowledge()
    assert first.ledger.get(earlier.idempotency_key).indexed is True

    other = config.pheasant.model_copy(update={"source_name": "swarm-lab-second"})
    second = ingestor_for(config, mock_server, tracer, pheasant=other)
    later = request_for("body", region_source=other.source_name)
    assert later.idempotency_key != earlier.idempotency_key
    receipts = second.submit([later])
    assert receipts[0].accepted is True
    assert receipts[0].indexed is False, "nothing in the new source is indexed yet"
    second.sync()
    second.acknowledge()
    assert second.ledger.get(later.idempotency_key).indexed is True


def test_a_receipt_that_moves_is_accepted_again_on_0_14(config, mock_server, tracer):
    """pheasant 0.14.0's fold, which the mock plays by default: one key sent
    to another source is not indexed there until that source is synced."""

    first = ingestor_for(config, mock_server, tracer)
    request = request_for("body")
    first.submit([request])
    first.sync()
    first.acknowledge()
    other = config.pheasant.model_copy(update={"source_name": "swarm-lab-second"})
    receipt = ingestor_for(config, mock_server, tracer, pheasant=other).submit([request])[0]
    assert receipt.accepted is True and receipt.indexed is False


def test_acknowledging_does_not_count_as_a_submission(config, mock_server, tracer):
    ingestor = ingestor_for(config, mock_server, tracer)
    request = request_for("body")
    ingestor.submit([request])
    ingestor.sync()
    ingestor.acknowledge()
    receipt = ingestor.ledger.get(request.idempotency_key)
    assert receipt.submissions == 1, (
        "counting the region's own acknowledgement would move a number that a harness reads "
        "to prove *it* retried"
    )


def test_a_transport_success_with_no_receipt_is_recorded_as_no_receipt(config, tracer):
    class Silent(MockPheasantServer):
        def _tool_submit_documents(self, arguments):
            return {"submission_id": "s-1", "receipts": []}

    ingestor = ingestor_for(config, Silent(), tracer)
    receipts = ingestor.submit([request_for("body")])
    assert receipts[0].status == "no_receipt"
    assert receipts[0].accepted is False


def test_an_empty_document_is_rejected_with_a_code(config, mock_server, tracer):
    ingestor = ingestor_for(config, mock_server, tracer)
    receipts = ingestor.submit([request_for("")])
    assert receipts[0].status == "rejected"
    # Pheasant's code for it: an empty item is an invalid request, permanently.
    assert receipts[0].error_code == "INVALID_REQUEST"
    assert receipts[0].retryable is False
    assert receipts[0].error_message, "the region's reason travels with the receipt"


def test_reconcile_reports_silent_loss_not_a_difference_of_totals(config, mock_server, tracer):
    ingestor = ingestor_for(config, mock_server, tracer)
    ingestor.submit([request_for("body one"), request_for("body two", "source-2")])
    ingestor.sync()
    ingestor.acknowledge()
    # One artifact disappears from the region while its receipt says indexed.
    mock_server.documents.pop(next(iter(mock_server.documents)))
    report = ingestor.reconcile()
    assert report["silent_loss"] == 1


def test_the_digest_the_region_accepted_is_compared_with_the_one_submitted(
    config, mock_server, tracer
):
    ingestor = ingestor_for(config, mock_server, tracer)
    receipts = ingestor.submit([request_for("body")])
    assert receipts[0].digest_matches is True


def test_an_unreported_digest_is_unknown_rather_than_a_mismatch():
    receipts = parse_receipts(
        {"receipts": [{"idempotency_key": "k", "status": "accepted"}]},
        run_id="run-1",
        key_to_source={"k": "source-1"},
    )
    assert receipts[0].digest_matches is None


def test_the_ledger_reports_submissions_without_a_receipt():
    ledger = ReceiptLedger()
    ledger.note_submission("k1")
    ledger.note_submission("k2")
    ledger.record(
        parse_receipts(
            {"receipts": [{"idempotency_key": "k1", "status": "indexed"}]},
            run_id="r",
            key_to_source={},
        )[0]
    )
    assert ledger.submitted_without_receipt() == ["k2"]
    assert ledger.receipt_rate() == (1, 2)


def test_a_snapshot_is_idempotent_over_an_unchanged_region(config, mock_server, tracer):
    ingestor = ingestor_for(config, mock_server, tracer)
    ingestor.submit([request_for("body")])
    ingestor.sync()
    first = ingestor.seal_snapshot(label="a")
    second = ingestor.seal_snapshot(label="b")
    assert first["snapshot_id"] == second["snapshot_id"]


def test_a_search_pinned_to_a_drifted_snapshot_is_refused(config, mock_server, tracer):
    from pheasant_lab.pheasant.capabilities import resolve
    from pheasant_lab.pheasant.protocol import McpToolError
    from pheasant_lab.pheasant.retrieval import Retriever, SearchRequest

    client = PheasantClient.in_process(config.pheasant, mock_server, tracer=tracer)
    session = client.connect()
    capabilities = resolve(config.pheasant, session.tools)
    ingestor = Ingestor(client, capabilities, config.pheasant, run_id="run-1", tracer=tracer)
    ingestor.submit([request_for("body about dsup")])
    ingestor.sync()
    snapshot = ingestor.seal_snapshot(label="frozen")

    retriever = Retriever(client, capabilities, config.pheasant)
    request = SearchRequest(
        run_id="run-1",
        arm_id="P0",
        question_id="q-1",
        query="dsup",
        namespace="pheasant-lab",
        snapshot_id=snapshot["snapshot_id"],
    )
    assert retriever.search(request).results, "the snapshot must answer while the region stands"

    ingestor.submit([request_for("a second document", "source-2")])
    ingestor.sync()
    # Pheasant's SNAPSHOT_DRIFTED refusal, as its text reads: the code itself
    # is not in the message, only in the readiness contract's table.
    with pytest.raises(McpToolError, match="no longer describes this region"):
        retriever.search(request)


# -- the pin the region may not offer --------------------------------------


def test_the_pin_is_not_sent_when_the_region_declares_no_name_for_it(config):
    """A region whose search tool takes no pin gets no pin.

    Before 0.12.6 pheasant exposed the snapshot pin on HTTP and not on MCP,
    and other regions may still. The lab must not send an argument the tool does not accept, and must not
    record a run as pinned when it was not: a run that looks pinned and is not
    is the one failure a sealed snapshot exists to prevent.
    """

    from pheasant_lab.pheasant.retrieval import SearchRequest

    request = SearchRequest(
        run_id="run-1",
        arm_id="P0",
        question_id="q-1",
        query="dsup",
        namespace="pheasant-lab",
        snapshot_id="snap-1",
        as_of="2020-01-01",
    )
    unmapped = {"query": "query", "max_results": "max_results", "mode": "mode", "memory": "memory"}
    arguments = request.as_arguments(unmapped, "knowledge_base", "pheasant-lab")
    assert "snapshot_id" not in arguments
    assert "as_of" not in arguments
    assert request.pin_sent(unmapped) is False

    mapped = {**unmapped, "snapshot_id": "snapshot_id", "as_of": "as_of"}
    arguments = request.as_arguments(mapped, "knowledge_base", "pheasant-lab")
    assert arguments["snapshot_id"] == "snap-1"
    assert arguments["as_of"] == "2020-01-01"
    assert request.pin_sent(mapped) is True


def test_the_response_records_whether_the_pin_reached_the_region(config, mock_server, tracer):
    from pheasant_lab.pheasant.capabilities import resolve
    from pheasant_lab.pheasant.retrieval import Retriever, SearchRequest

    client = PheasantClient.in_process(config.pheasant, mock_server, tracer=tracer)
    session = client.connect()
    retriever = Retriever(client, resolve(config.pheasant, session.tools), config.pheasant)
    assert retriever.supports_pinning is True

    ingestor = ingestor_for(config, mock_server, tracer)
    ingestor.submit([request_for("body about dsup")])
    ingestor.sync()
    snapshot = ingestor.seal_snapshot(label="frozen")
    response = retriever.search(
        SearchRequest(
            run_id="run-1",
            arm_id="P0",
            question_id="q-1",
            query="dsup",
            namespace="pheasant-lab",
            snapshot_id=snapshot["snapshot_id"],
        )
    )
    assert response.pin_sent is True
    assert response.as_record()["pin_sent"] is True


def _shipped_pheasant_file(environ: dict[str, str] | None = None):
    from pathlib import Path

    import yaml

    from pheasant_lab.settings import PheasantFile, interpolate

    repo = Path(__file__).resolve().parents[2]
    body = yaml.safe_load((repo / "configs/pheasant-mcp.example.yaml").read_text())
    return PheasantFile.model_validate(interpolate(body, environ or {}, []))


def _pheasant_tools(version: str = "0.14.0") -> dict:
    import json
    from pathlib import Path

    fixture = Path(__file__).resolve().parents[1] / f"fixtures/pheasant/tools-{version}.json"
    return json.loads(fixture.read_text())["tools"]


@pytest.mark.parametrize("version", ["0.13.4", "0.13.11", "0.14.0"])
def test_the_shipped_example_config_resolves_against_a_real_pheasant(version):
    """A regression guard on the configuration itself, against the real thing.

    The schemas are pheasant's own ``tools/list``, captured from a
    running region rather than written down, so this fails when the shipped
    map names a tool or an argument the region does not have - including a
    pin it does not accept, which would make a run *look* pinned. The mock
    cannot answer this: it accepts whatever it was written to accept.
    """

    from pheasant_lab.pheasant.capabilities import configured_arguments, resolve

    config = _shipped_pheasant_file()
    tools = _pheasant_tools(version)
    capabilities = resolve(config, tools)
    unusable = {r.name: r.reason for r in capabilities.resolutions.values() if not r.usable}
    assert unusable == {}
    # pheasant >= 0.12.6 takes the pin on `search_context`, so P0 is pinned;
    # 0.13.1 takes `expand`, so graph expansion is one experiment setting away.
    search_arguments = set(tools["search_context"]["inputSchema"]["properties"])
    assert {"snapshot_id", "as_of", "expand"} <= set(configured_arguments(config, "search"))
    assert set(configured_arguments(config, "search")) <= search_arguments


def test_a_0_13_1_region_resolves_without_the_index_queue():
    """``get_index_queue`` arrived in 0.13.2, and it is optional.

    That is the property worth holding: an older region still resolves, with
    the pre-claim interval and P1's memory indexing reported as unknown.
    """

    from pheasant_lab.pheasant.capabilities import resolve

    capabilities = resolve(_shipped_pheasant_file(), _pheasant_tools("0.13.1"))
    unusable = {r.name: r.reason for r in capabilities.resolutions.values() if not r.usable}
    assert unusable == {
        "index_queue": "'get_index_queue' is not in tools/list",
        "source_inventory": "'describe_source' is not in tools/list",
    }
    assert capabilities.resolutions["index_queue"].required is False


def test_a_0_13_2_region_resolves_without_the_source_inventory():
    """``describe_source`` arrived in 0.13.3 and is optional: 0.13.2 still resolves."""

    from pheasant_lab.pheasant.capabilities import resolve

    capabilities = resolve(_shipped_pheasant_file(), _pheasant_tools("0.13.2"))
    unusable = {r.name: r.reason for r in capabilities.resolutions.values() if not r.usable}
    assert unusable == {"source_inventory": "'describe_source' is not in tools/list"}
    assert capabilities.resolutions["source_inventory"].required is False


@pytest.mark.parametrize("version", ["0.12.16", "0.13.0"])
def test_an_older_region_refuses_the_shipped_map_until_expand_is_unmapped(version):
    """The documented way back to pheasant 0.12.6-0.13.0, run against their schemas.

    Mapped, ``expand`` is an argument their search tool does not take, so
    preflight refuses and names it. Unmapped the way the shipped file says to
    (``--set argument_map.search.expand=null``), everything else resolves.
    """

    from pheasant_lab.pheasant.capabilities import resolve

    tools = _pheasant_tools(version)
    refused = resolve(_shipped_pheasant_file(), tools)
    assert refused.has("search") is False
    assert "expand" in (refused.resolutions["search"].reason or "")

    from pheasant_lab.settings import PheasantFile

    body = _shipped_pheasant_file().model_dump()
    body["argument_map"]["search"]["expand"] = None
    unmapped = PheasantFile.model_validate(body)
    assert "expand" not in unmapped.argument_map["search"]
    capabilities = resolve(unmapped, tools)
    unusable = {r.name: r.reason for r in capabilities.resolutions.values() if not r.usable}
    # Optional and newer than either release; absent is the expected answer.
    assert unusable == {
        "index_queue": "'get_index_queue' is not in tools/list",
        "source_inventory": "'describe_source' is not in tools/list",
    }


def test_a_pin_the_region_does_not_accept_is_refused_at_preflight():
    """The same map against a region whose search tool has no pin (pre-0.12.6)."""

    from pheasant_lab.pheasant.capabilities import resolve

    tools = _pheasant_tools()
    search = dict(tools["search_context"])
    schema = dict(search["inputSchema"])
    schema["properties"] = {
        k: v for k, v in schema["properties"].items() if k not in {"snapshot_id", "as_of"}
    }
    search["inputSchema"] = schema
    capabilities = resolve(_shipped_pheasant_file(), {**tools, "search_context": search})
    assert capabilities.has("search") is False
    assert "snapshot_id" in (capabilities.resolutions["search"].reason or "")


# -- graph expansion, declared like the pin ------------------------------------


def test_expansion_is_sent_only_when_the_region_declares_a_name_for_it():
    from pheasant_lab.pheasant.retrieval import SearchRequest

    request = SearchRequest(
        run_id="run-1",
        arm_id="P0",
        question_id="q-1",
        query="dsup",
        namespace="pheasant-lab",
        expand={"depth": 2},
    )
    unmapped = {"query": "query", "max_results": "max_results", "mode": "mode", "memory": "memory"}
    assert "expand" not in request.as_arguments(unmapped, "knowledge_base", "pheasant-lab")
    assert request.expand_sent(unmapped) is False

    mapped = {**unmapped, "expand": "expand"}
    assert request.as_arguments(mapped, "knowledge_base", "pheasant-lab")["expand"] == {"depth": 2}
    assert request.expand_sent(mapped) is True

    request.expand = None
    assert "expand" not in request.as_arguments(mapped, "knowledge_base", "pheasant-lab")
    assert request.expand_sent(mapped) is False


def test_the_response_records_the_expansion_the_region_reported(config, mock_server, tracer):
    from pheasant_lab.pheasant.capabilities import resolve
    from pheasant_lab.pheasant.retrieval import Retriever, SearchRequest

    config.pheasant.argument_map["search"] = {
        **config.pheasant.argument_map["search"],
        "expand": "expand",
    }
    client = PheasantClient.in_process(config.pheasant, mock_server, tracer=tracer)
    retriever = Retriever(client, resolve(config.pheasant, client.connect().tools), config.pheasant)
    assert retriever.supports_graph_expansion is True
    ingestor = ingestor_for(config, mock_server, tracer)
    ingestor.submit([request_for("body about dsup")])
    ingestor.sync()

    def search(expand):
        return retriever.search(
            SearchRequest(
                run_id="run-1",
                arm_id="P0",
                question_id="q-1",
                query="dsup",
                namespace="pheasant-lab",
                expand=expand,
            )
        ).as_record()

    expanded = search(True)
    assert expanded["expand_sent"] is True
    assert expanded["expansion"]["seeds"] == 1
    assert expanded["results"][0]["graph_neighbors"] == []
    plain = search(None)
    assert plain["expand_sent"] is False and plain["expansion"] is None


def test_doctor_refuses_an_expansion_the_pheasant_file_cannot_send(config, capsys):
    from pheasant_lab.cli import main

    demo = str(_repo() / "configs/demo.yaml")
    common = ["doctor", "--config", demo, "--env-file", "/nonexistent", "--offline"]
    asked = [*common, "--set", "replay.graph_expansion=2"]
    unmapped = [*asked, "--set", "argument_map.search.expand=null"]
    assert main(unmapped) != 0
    assert "does not map `expand`" in capsys.readouterr().out

    # Only this finding is asserted on: the rest of doctor's verdict depends
    # on providers and the machine, which this test is not about.
    main(asked)
    out = capsys.readouterr().out
    assert "capability search -> search_context: ok" in out
    assert "does not map `expand`" not in out


def test_graph_expansion_is_validated_when_the_config_loads():
    from pydantic import ValidationError

    from pheasant_lab.settings import ReplaySection

    assert ReplaySection(graph_expansion=False).expansion is None
    assert ReplaySection(graph_expansion=0).expansion is None
    assert ReplaySection(graph_expansion=True).expansion is True
    assert ReplaySection(graph_expansion={"depth": 2}).expansion == {"depth": 2}
    with pytest.raises(ValidationError, match="from 1 to 3"):
        ReplaySection(graph_expansion=5)
    with pytest.raises(ValidationError, match="does not take hops"):
        ReplaySection(graph_expansion={"hops": 2})


def _repo():
    from pathlib import Path

    return Path(__file__).resolve().parents[2]
