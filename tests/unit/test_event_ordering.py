"""Concurrent workers must append events in their allocated sequence order."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

from pheasant_lab.redaction import Redactor
from pheasant_lab.tracing.events import EventLog, read_jsonl


def test_a_slow_first_writer_cannot_be_overtaken(tmp_path, monkeypatch):
    path = tmp_path / "events.jsonl"
    log = EventLog(path, run_id="run", config_digest="config", redactor=Redactor())
    first_writing = threading.Event()
    second_started = threading.Event()
    append = log._writer.append

    def delayed(record):
        if record["sequence"] == 1:
            first_writing.set()
            assert second_started.wait(timeout=5)
            time.sleep(0.05)
        append(record)

    monkeypatch.setattr(log._writer, "append", delayed)

    def emit_second():
        assert first_writing.wait(timeout=5)
        second_started.set()
        return log.emit("second", trace_id="trace", span_id="span")

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(log.emit, "first", trace_id="trace", span_id="span")
            second = pool.submit(emit_second)
            first.result(timeout=5)
            second.result(timeout=5)
    finally:
        log.close()
    assert [row["sequence"] for row in read_jsonl(path)] == [1, 2]
