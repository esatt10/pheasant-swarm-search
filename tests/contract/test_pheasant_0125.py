"""Adapter regressions from a real Pheasant 0.12.5 Docker setup run.

These tests keep the observed wire shapes offline. The live, opt-in workflow
is scripts/check_pheasant_setup.py, outside the default pytest suite.
"""

from types import SimpleNamespace

import pytest

from pheasant_lab.pheasant.ingestion import Ingestor
from pheasant_lab.pheasant.protocol import ProtocolError, parse_tool_result
from pheasant_lab.pheasant.question_memory import BenchmarkQuestionPublisher
from pheasant_lab.pheasant.receipts import IngestReceipt, parse_receipts
from pheasant_lab.pheasant.retrieval import Retriever, SearchRequest, normalise_results


class RecordedClient:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def call(self, tool, arguments, **kwargs):
        self.calls.append((tool, arguments, kwargs))
        body = self.responses[tool]
        if callable(body):
            body = body(arguments)
        return SimpleNamespace(
            result=parse_tool_result(tool, {"structuredContent": body}),
            duration_ms=1,
            server_trace_id=None,
            warnings=[],
            partial=False,
            status="succeeded",
        )


def capabilities(*names):
    return SimpleNamespace(has=lambda name: name in names, tool=lambda name: name)


def search_request():
    return SearchRequest(
        run_id="run",
        arm_id="P0",
        question_id="q",
        query="DNA repair",
        namespace="kb",
        principal="reader",
    )


def live_hit():
    return {
        "node_id": "file:collection:paper.md:branch=none",
        "rank": 1,
        "path": "/state/uploads/collection/paper.md",
        "relative_path": "paper.md",
        "score": 0.016393,
        "retrieved_by": "text",
        "chunks": [{"chunk_id": "chunk:1", "text_preview": "Truncated abstract..."}],
        "provenance": {"source_id": "collection", "source_type": "document_folder"},
    }


def test_split_receipts_keep_rejections_and_verify_the_bare_sha256():
    result = parse_receipts(
        {
            "accepted": [
                {"idempotency_key": "a", "disposition": "accepted", "content_sha256": "abc"}
            ],
            "rejected": [
                {"idempotency_key": "b", "disposition": "rejected", "error_code": "DENIED"}
            ],
            "indexed": 0,
        },
        run_id="run",
        key_to_source={"a": "source-a", "b": "source-b"},
        requested_digests={"a": "sha256:abc"},
    )
    assert [(r.source_id, r.status) for r in result] == [
        ("source-a", "accepted"),
        ("source-b", "rejected"),
    ]
    assert result[0].digest_matches is True
    assert result[0].indexed is False
    assert result[1].error_code == "DENIED"


def test_receipt_totals_are_not_item_receipts():
    assert parse_receipts({"accepted": 3, "rejected": 0}, run_id="r", key_to_source={}) == []


@pytest.mark.parametrize("acknowledged", [0, 1])
@pytest.mark.parametrize("disposition", ["indexed", "accepted", None])
def test_numeric_acknowledgement_reads_the_actual_item_status(config, acknowledged, disposition):
    client = RecordedClient(
        {
            "ingest_acknowledge": {"acknowledged": acknowledged, "still_accepted": 0},
            "ingest_status": {"receipts": [{"idempotency_key": "key", "disposition": disposition}]},
        }
    )
    ingestor = Ingestor(client, capabilities("ingest_acknowledge"), config.pheasant, run_id="run")
    ingestor.ledger.record(
        IngestReceipt(
            receipt_id="receipt",
            run_id="run",
            source_id="source",
            idempotency_key="key",
            submission_id="submission",
            status="accepted",
        )
    )
    ingestor.acknowledge()
    receipt = ingestor.ledger.get("key")
    assert receipt.status == (disposition or "unknown")
    assert receipt.submissions == 1
    assert client.calls[1][1]["idempotency_key"] == "key"


def test_new_submission_directory_is_registered_before_sync(config):
    client = RecordedClient({"list_sources": {"sources": []}, "register_source": {}, "sync": {}})
    ingestor = Ingestor(
        client, capabilities("list_sources", "register_source", "sync"), config.pheasant, run_id="r"
    )
    ingestor._submission_directory = "/state/uploads/lab"
    ingestor.sync()
    assert [c[0] for c in client.calls] == ["list_sources", "register_source", "sync"]
    assert client.calls[1][1]["path"] == "/state/uploads/lab"
    assert client.calls[1][1]["source_type"] == "document_folder"


def test_registration_refuses_an_existing_source_with_another_path(config):
    client = RecordedClient(
        {
            "list_sources": {
                "sources": [{"name": config.pheasant.source_name, "path": "/user/documents"}]
            }
        }
    )
    ingestor = Ingestor(
        client, capabilities("list_sources", "register_source", "sync"), config.pheasant, run_id="r"
    )
    ingestor._submission_directory = "/state/uploads/lab"
    with pytest.raises(RuntimeError, match="already indexes another path"):
        ingestor.sync()
    assert [c[0] for c in client.calls] == ["list_sources"]


def test_retrieval_preserves_previews_and_links_collection_hits_to_paper_receipts():
    hit = live_hit()
    results = normalise_results(
        {"results": [hit]},
        search_request(),
        artifact_sources={hit["node_id"]: "source-paper"},
    )
    assert results[0].matched_text == "Truncated abstract..."
    assert results[0].source_id == "source-paper"
    assert results[0].retrieval_arm == "text"


def test_source_name_conflicts_are_checked_beyond_the_first_page(config):
    def page(arguments):
        if arguments["offset"] == 0:
            return {"sources": [{"name": f"existing-{i}"} for i in range(100)]}
        return {"sources": [{"name": config.pheasant.source_name, "path": "/user/documents"}]}

    client = RecordedClient({"list_sources": page})
    ingestor = Ingestor(
        client,
        capabilities("list_sources", "register_source", "sync"),
        config.pheasant,
        run_id="r",
    )
    ingestor._submission_directory = "/state/uploads/lab"
    with pytest.raises(RuntimeError, match="already indexes another path"):
        ingestor.sync()
    assert [call[1]["offset"] for call in client.calls] == [0, 100]


def test_a_fetch_without_principal_support_cannot_broaden_a_scoped_search(config):
    config.pheasant.argument_map["fetch"] = {"path": "path"}
    client = RecordedClient({"search": {"results": [live_hit()]}})
    retriever = Retriever(client, capabilities("search", "fetch"), config.pheasant)
    response = retriever.search(search_request())
    assert response.results[0].matched_text == "Truncated abstract..."
    assert [call[0] for call in client.calls] == ["search"]


def test_full_file_reads_use_the_returned_path_source_and_same_principal(config):
    hit = live_hit()
    config.pheasant.argument_map["fetch"] = {
        "path": "path",
        "source_name": "source_name",
        "principal": "principal",
    }
    client = RecordedClient(
        {
            "search": {"results": [hit]},
            "fetch": {"content": "Complete abstract: DNA repair evidence."},
        }
    )
    retriever = Retriever(
        client,
        capabilities("search", "fetch"),
        config.pheasant,
        artifact_sources={hit["node_id"]: "source-paper"},
    )
    response = retriever.search(search_request())
    assert response.results[0].matched_text == "Complete abstract: DNA repair evidence."
    assert response.results[0].source_id == "source-paper"
    assert client.calls[1][1] == {
        "knowledge_base": config.pheasant.knowledge_base,
        "path": "paper.md",
        "source_name": "collection",
        "principal": "reader",
    }
    assert client.calls[1][2]["question_id"] == "q"


def test_benchmark_questions_become_memories_without_answer_material(config, question):
    prompt, _expected_fact = question
    client = RecordedClient(
        {
            "write_memory": {
                "record": {"record_id": "mem-question"},
                "created": True,
                "outcome": "created",
            }
        }
    )
    publisher = BenchmarkQuestionPublisher(
        client,
        capabilities("write_memory"),
        config.pheasant,
        run_id="run-1",
        benchmark_version="bench-1",
    )
    published = publisher.publish([prompt])
    arguments = client.calls[0][1]
    assert published[0].record_id == "mem-question"
    assert prompt.text in arguments["text"]
    assert "Dsup reduced" not in arguments["text"]
    assert "matcher" not in arguments["text"].lower()
    assert arguments["scope"] == "org"
    assert arguments["sync"] is True
    assert "benchmark-question" in arguments["tags"]


def test_question_publication_refuses_a_write_without_a_record_id(config, question):
    prompt, _expected_fact = question
    publisher = BenchmarkQuestionPublisher(
        RecordedClient({"write_memory": {"created": True, "outcome": "created"}}),
        capabilities("write_memory"),
        config.pheasant,
        run_id="run-1",
        benchmark_version="bench-1",
    )

    with pytest.raises(ProtocolError, match="returned no record_id"):
        publisher.publish([prompt])
