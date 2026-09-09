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


def request_for(text: str, source_id: str = "source-1") -> IngestRequest:
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
    )


def ingestor_for(config, server, tracer=None) -> Ingestor:
    from pheasant_lab.pheasant.capabilities import resolve

    client = PheasantClient.in_process(config.pheasant, server, tracer=tracer)
    session = client.connect()
    return Ingestor(
        client,
        resolve(config.pheasant, session.tools),
        config.pheasant,
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
    assert receipts[0].error_code == "EMPTY_DOCUMENT"
    assert receipts[0].retryable is False


def test_reconcile_reports_silent_loss_not_a_difference_of_totals(config, mock_server, tracer):
    ingestor = ingestor_for(config, mock_server, tracer)
    ingestor.submit([request_for("body one"), request_for("body two", "source-2")])
    ingestor.sync()
    # One artifact disappears from the region while its receipt remains.
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
    with pytest.raises(McpToolError, match="SNAPSHOT_DRIFTED"):
        retriever.search(request)


# -- the pin the region may not offer --------------------------------------


def test_the_pin_is_not_sent_when_the_region_declares_no_name_for_it(config):
    """pheasant >= 0.12 exposes the snapshot pin on HTTP and not on MCP.

    The lab must not send an argument the tool does not accept, and must not
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


def test_the_shipped_example_config_claims_no_pin_that_pheasant_lacks():
    """A regression guard on the configuration itself.

    `services/retrieval.SearchRequest` carries `snapshot_id` and `as_of`, and
    pheasant's HTTP surface passes both — but `search_context` accepts
    neither. Mapping them here would make every live run fail at P0's first
    search, and the mock region accepts them, so no test that only exercises
    the mock could see it.
    """

    from pathlib import Path

    import yaml

    repo = Path(__file__).resolve().parents[2]
    body = yaml.safe_load((repo / "configs/pheasant-mcp.example.yaml").read_text())
    search_map = (body.get("argument_map") or {}).get("search") or {}
    assert "snapshot_id" not in search_map
    assert "as_of" not in search_map
