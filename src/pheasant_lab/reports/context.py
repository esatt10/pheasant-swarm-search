"""What a reader needs beside the numbers: names, questions, answers, evidence.

The metrics name everything by id - ``q-2da8504b21b694cb``,
``source-3ebf4e29b1dbbb08``, ``file:swarm-lab-literature:…:branch=none`` -
because ids are what join. A person reading a report needs the question, the
document's title and what each arm actually said. This module reads those
from the run directory once, so every report labels things the same way, and
a report written without it (an old run, a partial one) still renders with
ids where names would have been.

It reads; it never computes a number a report states. Every figure in the
reports still comes from ``metrics/``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ..lifecycle import RunPaths
from ..textkit import truncate
from ..tracing.events import read_jsonl

#: What each arm *is*, in a phrase. The ids stay beside these everywhere,
#: because the metrics, the config and the trace all use the ids.
ARM_NAMES: dict[str, str] = {
    "S0": "source-aware specialist",
    "C0": "model prior only, no retrieval",
    "P0": "Pheasant corpus search",
    "P1": "Pheasant search + memory",
    "P2": "Pheasant search + memory + tuned queries",
}

#: What each comparison asks, in the words a reader would ask it.
COMPARISON_QUESTIONS: dict[tuple[str, str], str] = {
    ("P0", "C0"): "Does searching the Pheasant corpus beat the model's own prior?",
    ("P1", "P0"): "Does memory help?",
    ("P2", "P1"): "Does tuning the search queries help?",
    ("P1", "S0"): "Does Pheasant with memory rival the specialist who built the collection?",
    ("P2", "S0"): "Does tuned Pheasant search rival the specialist?",
}


def arm(arm_id: str) -> str:
    """``P1 (Pheasant search + memory)`` - an id a reader can place."""

    name = ARM_NAMES.get(arm_id)
    return f"{arm_id} ({name})" if name else arm_id


@dataclass
class RunContext:
    questions: dict[str, dict[str, Any]] = field(default_factory=dict)
    facts: dict[str, dict[str, Any]] = field(default_factory=dict)
    evidence: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Any id a document goes by - the lab's source id and the region's
    #: artifact id - mapped to the source record that names it.
    documents: dict[str, dict[str, Any]] = field(default_factory=dict)
    answers: dict[tuple[str, str, int], dict[str, Any]] = field(default_factory=dict)
    #: The `tuning.strategy` P2 ran with, or None when the run recorded none.
    tuning: dict[str, Any] | None = None
    #: The `memory.seeded` summary P1 ran with, or None.
    memory: dict[str, Any] | None = None
    #: Seeded memory records by record id, for naming a memory hit or citation.
    memory_records: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls, paths: RunPaths) -> RunContext:
        context = cls()
        benchmark = paths.benchmark
        for row in _rows(benchmark / "questions.jsonl") or _rows(paths.raw_file("questions.jsonl")):
            context.questions[str(row.get("question_id"))] = row
        for row in _rows(benchmark / "expected-facts.jsonl"):
            context.facts[str(row.get("fact_id"))] = row
        for row in _rows(benchmark / "expected-evidence.jsonl"):
            context.evidence[str(row.get("question_id"))] = row
        requests = {
            str(row.get("source_id")): row for row in _rows(paths.raw_file("ingest-requests.jsonl"))
        }
        for row in _rows(paths.raw_file("sources.jsonl")):
            source = dict(row)
            request = requests.get(str(source.get("source_id")))
            if request and request.get("relative_path"):
                source.setdefault("relative_path", request["relative_path"])
            for key in ("source_id", "artifact_id"):
                if source.get(key):
                    context.documents[str(source[key])] = source
        for row in _rows(paths.raw_file("answers.jsonl")):
            key = (
                str(row.get("arm_id")),
                str(row.get("question_id")),
                int(row.get("repetition") or 1),
            )
            context.answers[key] = row
        for row in _rows(paths.raw_file("memory-records.jsonl")):
            if row.get("record_id"):
                context.memory_records[str(row["record_id"])] = row
        for event in _rows(paths.raw_file("events.jsonl")):
            kind = event.get("event_type")
            if kind == "tuning.strategy":
                context.tuning = dict(event.get("payload") or {})
            elif kind == "memory.seeded":
                context.memory = dict(event.get("payload") or {})
        return context

    # -- names -----------------------------------------------------------------
    def question_text(self, question_id: str, limit: int | None = None) -> str:
        text = str((self.questions.get(question_id) or {}).get("text") or question_id)
        return truncate(text, limit) if limit else text

    def document(self, any_id: str) -> str:
        """``Title`` (first author, year) for a source or artifact id; the id if unknown."""

        source = self.documents.get(str(any_id))
        if not source or not source.get("title"):
            memory = _MEMORY_ID.search(str(any_id))
            if memory:
                record = self.memory_records.get(memory.group("record")) or {}
                about = record.get("subject")
                kind = {"fact": "retrieval proof"}.get(str(record.get("kind")), record.get("kind"))
                if about:
                    return f"memory {kind or 'record'} for “{truncate(str(about), 90)}”"
                return f"memory record `{memory.group('record')}`"
            return f"`{any_id}`"
        authors = list(source.get("authors") or [])
        byline = _surname(authors[0]) if authors else ""
        if len(authors) > 1:
            byline += " et al."
        year = str(source.get("published_at") or "")[:4]
        tail = ", ".join(part for part in (byline, year) if part)
        title = truncate(" ".join(str(source["title"]).split()), 110)
        return f"*{title}*" + (f" ({tail})" if tail else "")

    def same_document(self, left: str, right: str) -> bool:
        a, b = self.documents.get(left), self.documents.get(right)
        if a is not None and b is not None:
            return a.get("source_id") == b.get("source_id")
        return left == right

    # -- the expected answer ---------------------------------------------------
    def required_facts(self, question_id: str) -> list[dict[str, Any]]:
        question = self.questions.get(question_id) or {}
        return [
            self.facts[fact_id]
            for fact_id in question.get("required_fact_ids") or []
            if fact_id in self.facts
        ]

    def expected_documents(self, question_id: str) -> list[str]:
        """Ids of the documents that hold this question's answer."""

        row = self.evidence.get(question_id) or {}
        out = [str(a) for a in row.get("acceptable_artifact_ids") or []]
        out += [str(s) for s in row.get("acceptable_source_ids") or []]
        return out

    def is_abstention(self, question_id: str) -> bool:
        return (self.questions.get(question_id) or {}).get("type") == "abstention"

    # -- what an arm did -------------------------------------------------------
    def answer(self, arm_id: str, question_id: str, repetition: int = 1) -> dict[str, Any] | None:
        return self.answers.get((arm_id, question_id, repetition))

    def cited(self, answer: Mapping[str, Any] | None) -> list[str]:
        """Distinct ids the answer cited, in the order it first cited them."""

        out: list[str] = []
        for claim in (answer or {}).get("claims") or []:
            for citation in claim.get("citations") or []:
                if citation and not any(self.same_document(citation, seen) for seen in out):
                    out.append(str(citation))
        return out

    def retrieved(self, answer: Mapping[str, Any] | None) -> list[tuple[str, int | None, str]]:
        """``(id, rank, query)`` for every distinct document the arm's searches returned."""

        out: list[tuple[str, int | None, str]] = []
        for call in (answer or {}).get("search_calls") or []:
            for result in call.get("results") or []:
                for key in (result.get("artifact_id"), result.get("source_id")):
                    if key and not any(self.same_document(str(key), seen) for seen, _, _ in out):
                        out.append((str(key), result.get("rank"), str(call.get("query") or "")))
                        break
        return out

    def outcome(self, answer: Mapping[str, Any] | None) -> str:
        if answer is None:
            return "did not run"
        if answer.get("status") == "failed":
            return "failed"
        if answer.get("abstained"):
            return "abstained"
        return "answered"

    @property
    def tuning_applied(self) -> bool | None:
        """Did P2 run with any tuning at all? ``None`` when the run did not say."""

        if self.tuning is None:
            return None
        return bool(self.tuning.get("expansions") or self.tuning.get("drop_terms"))


#: A memory record's region path: ``…:org/mem-<id>.md…``.
_MEMORY_ID = re.compile(r"(?:^|[:/])(?P<record>mem-[A-Za-z0-9-]+)\.md")


def listing(items: list[str], limit: int = 3) -> str:
    """``a; b; c and 2 more`` - a list a sentence can carry."""

    if len(items) <= limit:
        return "; ".join(items)
    return "; ".join(items[:limit]) + f" and {len(items) - limit} more"


def _surname(author: str) -> str:
    author = str(author or "").strip()
    if "," in author:
        return author.split(",", 1)[0]
    return author.split()[-1] if author.split() else ""


def _rows(path: Any) -> Iterable[dict[str, Any]]:
    try:
        return list(read_jsonl(path)) if path.is_file() else []
    except (OSError, ValueError):
        return []
