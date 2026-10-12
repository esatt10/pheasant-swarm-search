"""Submission batches obey the same encoded-byte ceiling as the HTTP server."""

from __future__ import annotations

import base64
import hashlib
import json

import httpx
import pytest
from pydantic import ValidationError

from pheasant_lab.pheasant import protocol
from pheasant_lab.pheasant.capabilities import resolve
from pheasant_lab.pheasant.client import PheasantClient, StreamableHttpTransport, TransportError
from pheasant_lab.pheasant.ingestion import Ingestor, IngestPayload, IngestRequest, IngestSource
from pheasant_lab.pheasant.mock import MockPheasantServer
from pheasant_lab.settings import PheasantFile


def request_for(text="body", source_id="source-1"):
    return IngestRequest(
        run_id="run-1",
        topic_id="topic-1",
        source_id=source_id,
        namespace="pheasant-lab",
        source=IngestSource(title="t", stable_identifier=f"10.1/{source_id}"),
        payload=IngestPayload(text=text),
    )


class SizeLimitedServer(MockPheasantServer):
    """Enforce a wire ceiling and validate base64 bytes, without parsing PDFs."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bodies = []

    def handle(self, message):
        if (
            message.get("method") == "tools/call"
            and message["params"]["name"] == "submit_documents"
        ):
            body = json.dumps(message).encode("utf-8")
            if len(body) > self.config.ingest_max_request_bytes:
                raise TransportError("HTTP 413: Request body too large", status_code=413)
            self.bodies.append(body)
        return super().handle(message)

    def _tool_submit_documents(self, arguments):
        result = super()._tool_submit_documents(arguments)
        if arguments.get("content_encoding") == "base64":
            # Mirror the real region's receipt digest of original bytes. This
            # fixture does not claim to extract text from a binary document.
            for document in arguments["documents"]:
                key = document["idempotency_key"]
                sha = hashlib.sha256(base64.b64decode(document["text"], validate=True)).hexdigest()
                self.receipts[key]["content_sha256"] = sha
                for receipt in result["accepted"]:
                    if receipt["idempotency_key"] == key:
                        receipt["content_sha256"] = sha
        return result


def ingestor_for(config, tracer=None):
    server = SizeLimitedServer(config.pheasant)
    client = PheasantClient.in_process(config.pheasant, server, tracer=tracer)
    session = client.connect()
    session.tools["submit_documents"]["inputSchema"]["properties"]["content_encoding"] = {
        "type": "string"
    }
    return Ingestor(
        client,
        resolve(config.pheasant, session.tools),
        config.pheasant,
        run_id="run-1",
        tracer=tracer,
    ), server


def request_size(ingestor, requests):
    arguments = ingestor._submission_arguments(
        [request.as_document(ingestor._argument_map) for request in requests],
        "submission-" + "0" * 16,
    )
    return len(
        json.dumps(
            protocol.request(
                "tools/call", {"name": "submit_documents", "arguments": arguments}, 2**64 - 1
            )
        ).encode("utf-8")
    )


def batches(server):
    return [json.loads(body)["params"]["arguments"] for body in server.bodies]


def test_pdf_and_text_batches_count_base64_metadata_and_envelope(config, tmp_path, tracer):
    config.pheasant.argument_map["ingest"]["content_encoding"] = "content_encoding"
    ingestor, server = ingestor_for(config, tracer)
    originals = [
        b"%PDF-1.7\n" + bytes(range(256)) * 6,
        b"markdown evidence" * 95,
        b"third document" * 130,
    ]
    requests = []
    for index, content in enumerate(originals):
        request = request_for(source_id=f"source-{index}")
        if index == 0:
            path = tmp_path / "original.pdf"
            path.write_bytes(content)
            request.payload = IngestPayload(
                media_type="application/pdf", local_artifact_ref=str(path)
            )
        else:
            request.payload.text = content.decode("utf-8")
        requests.append(request)
    config.pheasant.ingest_max_request_bytes = request_size(ingestor, requests[:2])
    assert sum(len(content) for content in originals) < config.pheasant.ingest_max_request_bytes
    assert request_size(ingestor, requests) > config.pheasant.ingest_max_request_bytes

    receipts = ingestor.submit(requests)
    submitted = batches(server)
    assert [len(batch["documents"]) for batch in submitted] == [2, 1]
    documents = [document for batch in submitted for document in batch["documents"]]
    assert [base64.b64decode(document["text"]) for document in documents] == originals
    assert documents[0]["relative_path"].endswith(".pdf")
    assert all(receipt.accepted and receipt.digest_matches for receipt in receipts)
    assert all(len(body) <= config.pheasant.ingest_max_request_bytes for body in server.bodies)
    # Identical retries retain the batches' submission ids and per-file keys.
    ingestor.submit(requests)
    assert [batch["submission_id"] for batch in batches(server)[2:]] == [
        batch["submission_id"] for batch in submitted
    ]
    assert len(server.documents) == len(requests)
    ingestor.sync()
    ingestor.acknowledge()
    assert all(receipt.indexed for receipt in ingestor.ledger.receipts)


def test_boundary_and_json_escaping_count_with_no_base64(config):
    ingestor, server = ingestor_for(config)
    requests = [request_for('"\\\n雪' * 50, source_id=f"source-{i}") for i in range(3)]
    for request in requests:
        request.source.title = "研究 " * 20
    config.pheasant.ingest_max_request_bytes = request_size(ingestor, requests[:2])
    ingestor.submit(requests)
    assert [len(batch["documents"]) for batch in batches(server)] == [2, 1]
    server.bodies.clear()
    config.pheasant.ingest_max_request_bytes -= 1
    ingestor.submit(requests)
    assert [len(batch["documents"]) for batch in batches(server)] == [1, 1, 1]


def test_count_cap_still_bounds_small_document_batches(config):
    ingestor, server = ingestor_for(config)
    ingestor.submit([request_for(source_id=f"source-{i}") for i in range(21)])
    assert [len(batch["documents"]) for batch in batches(server)] == [20, 1]


def test_oversized_pdf_fails_locally_without_encoding_or_losing_other_documents(
    config, tmp_path, monkeypatch, tracer
):
    config.pheasant.argument_map["ingest"]["content_encoding"] = "content_encoding"
    config.pheasant.ingest_max_request_bytes = 2500
    path = tmp_path / "large.pdf"
    path.write_bytes(b"%PDF-1.7\n" + b"x" * 3000)
    large = request_for(source_id="large")
    large.payload = IngestPayload(media_type="application/pdf", local_artifact_ref=str(path))
    real = IngestRequest.as_document

    def encode(request, argument_map):
        assert request is not large, "an oversized binary must not be base64-allocated"
        return real(request, argument_map)

    monkeypatch.setattr(IngestRequest, "as_document", encode)
    ingestor, server = ingestor_for(config, tracer)
    receipts = ingestor.submit(
        [request_for(source_id="before"), large, request_for(source_id="after")]
    )
    assert [receipt.source_id for receipt in receipts] == ["before", "large", "after"]
    assert receipts[0].accepted and receipts[2].accepted
    failure = receipts[1]
    assert failure.status == "failed" and failure.error_code == "REQUEST_TOO_LARGE"
    assert failure.retryable is False and failure.submission_id is None
    assert "not sent" in failure.error_message
    assert len(server.documents) == 2
    assert ingestor.ledger.get(large.idempotency_key) is failure
    tracer.sync()
    rows = [
        json.loads(line)
        for line in (tracer.paths.raw / "ingest-receipts.jsonl").read_text().splitlines()
    ]
    assert any(row["error_code"] == "REQUEST_TOO_LARGE" for row in rows)


@pytest.mark.parametrize(
    "mapping",
    [
        {},
        {
            "content_encoding": "encoding",
            "document_text": "body",
            "document_path": "path",
            "document_key": "key",
            "document_metadata": "info",
        },
    ],
)
@pytest.mark.parametrize("length", [0, 1, 2, 3, 4, 1024])
def test_document_size_matches_actual_wire_encoding(mapping, length):
    request = request_for("雪" * length)
    request.source.title = 'A "quoted" title\\'
    assert request.encoded_document_size(mapping) == len(
        json.dumps(request.as_document(mapping)).encode("utf-8")
    )


def test_http_413_is_not_retried_even_with_an_idempotency_key(config):
    bodies = []

    def refuse(request):
        bodies.append(request.content)
        return httpx.Response(413, text="Request body too large")

    transport = StreamableHttpTransport("http://region.test/mcp")
    transport._client = httpx.Client(transport=httpx.MockTransport(refuse))
    client = PheasantClient(config.pheasant, transport=transport)
    try:
        with pytest.raises(TransportError, match="413"):
            client.call("submit_documents", {"documents": []}, idempotency_key="submission-1")
        assert len(bodies) == 1 and client.calls[-1].attempts == 1
    finally:
        transport.close()


@pytest.mark.parametrize("value", [0, -1])
def test_request_limit_must_be_positive(value):
    with pytest.raises(ValidationError, match="greater than 0"):
        PheasantFile(transport="mock", ingest_max_request_bytes=value)


def test_empty_submission_and_invalid_count_cap(config):
    ingestor, server = ingestor_for(config)
    assert ingestor.submit([]) == [] and server.bodies == []
    with pytest.raises(ValueError, match="positive"):
        ingestor.submit([], batch_size=0)
