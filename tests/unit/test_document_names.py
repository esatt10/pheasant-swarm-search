"""Documents land in the region under a name a person can read.

The file name is also what pheasant shows as a hit's title, weights most in
BM25, and what a memory record links to - so it must stay a slug a
preference rule and a Markdown link both survive, it must not change for a
source already submitted, and it must never be shared by two sources.
"""

from __future__ import annotations

import re
import threading
from types import SimpleNamespace

from pheasant_lab.pheasant.ingestion import (
    Ingestor,
    IngestSource,
    build_requests,
    document_name,
)
from pheasant_lab.settings import PheasantFile


def _item(source_id: str, title: str, **extra):
    return {
        "source_id": source_id,
        "title": title,
        "stable_identifier": extra.pop("stable_identifier", f"10.5555/{source_id}"),
        "authors": extra.pop("authors", ["Wei Chen", "Nadia Haddad"]),
        "published_at": extra.pop("published_at", "2018-02-11"),
        "text": extra.pop("text", f"body of {source_id}"),
        **extra,
    }


def _requests(*items):
    return build_requests(items, run_id="run-1", topic_id="topic-x", namespace="kb")


def _ingestor() -> Ingestor:
    return Ingestor(None, None, PheasantFile(transport="mock", knowledge_base="kb"), run_id="run-1")  # type: ignore[arg-type]


def test_a_document_is_named_by_its_title_then_first_author_and_year():
    (request,) = _requests(
        _item("source-1", "Structural basis of nucleosome binding by the Dsup protein")
    )
    assert request.path() == (
        "topic-x/structural-basis-of-nucleosome-binding-by-the-dsup-protein-chen-2018.md"
    )


def test_a_name_survives_pheasant_s_rule_and_link_grammars():
    name = document_name(
        IngestSource(
            title="Dsup, a nucleosome-binding protein (from R. varieornatus): résumé & notes",
            authors=["Haddad, Nadia"],
            published_at="2016",
        ),
        "source-1",
    )
    assert name == "dsup-a-nucleosome-binding-protein-from-r-varieornatus-resume-notes-haddad-2016"
    assert re.fullmatch(r"[a-z0-9-]+", name), "commas, spaces and parentheses break rules and links"


def test_a_long_title_is_cut_at_a_word_and_an_untitled_source_keeps_its_identifier():
    long = document_name(IngestSource(title="word " * 60, authors=[]), "source-1")
    assert len(long) <= 96 and not long.endswith("-")
    assert (
        document_name(IngestSource(title="", stable_identifier="10.5555/x.1"), "s") == "10.5555_x.1"
    )


def test_two_sources_never_share_a_path():
    ingestor = _ingestor()
    first, second = _requests(_item("source-1", "Same title"), _item("source-2", "Same title"))
    assert ingestor.assign_path(first) == "topic-x/same-title-chen-2018.md"
    assert ingestor.assign_path(second) == "topic-x/same-title-chen-2018-2.md"


def test_a_source_keeps_its_path_when_resubmitted_and_after_a_resume():
    ingestor = _ingestor()
    (request,) = _requests(_item("source-1", "A title"))
    path = ingestor.assign_path(request)
    (again,) = _requests(_item("source-1", "A title, corrected"))
    assert ingestor.assign_path(again) == path

    resumed = _ingestor()
    resumed.restore_paths([{"source_id": "source-1", "relative_path": "topic-x/old-name.md"}])
    (after,) = _requests(_item("source-1", "A title"))
    assert resumed.assign_path(after) == "topic-x/old-name.md", "a resume renamed a document"
    (other,) = _requests(_item("source-2", "Old name", authors=[], published_at=None))
    assert resumed.assign_path(other) == "topic-x/old-name-2.md"


def test_concurrent_branches_never_assign_one_path_twice():
    ingestor = _ingestor()
    requests = _requests(*[_item(f"source-{i}", "Shared title") for i in range(40)])
    threads = [
        threading.Thread(target=ingestor.assign_path, args=(request,)) for request in requests
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len({request.path() for request in requests}) == 40


def test_submit_sends_the_assigned_path_and_records_it():
    sent: list[dict] = []
    appended: list[tuple[str, dict]] = []

    def call(tool, arguments, **_kwargs):
        sent.extend(arguments["documents"])
        return SimpleNamespace(result=SimpleNamespace(payload=lambda: {"accepted": []}))

    ingestor = Ingestor(
        SimpleNamespace(call=call),  # type: ignore[arg-type]
        SimpleNamespace(tool=lambda name: "submit_documents", has=lambda name: True),  # type: ignore[arg-type]
        PheasantFile(transport="mock", knowledge_base="kb"),
        run_id="run-1",
        tracer=SimpleNamespace(
            append=lambda name, row: appended.append((name, row)), emit=lambda *a, **k: None
        ),
    )
    first, second = _requests(_item("source-1", "Same title"), _item("source-2", "Same title"))
    ingestor.submit([first, second])
    paths = [document["relative_path"] for document in sent]
    assert paths == ["topic-x/same-title-chen-2018.md", "topic-x/same-title-chen-2018-2.md"]
    assert [
        row["relative_path"] for name, row in appended if name == "ingest-requests.jsonl"
    ] == paths
