"""Reports a person can read, over evaluations that measure what they claim.

Two halves. The scoring fixes: a citation of the right document names that
document however the region spelled its id, and a forward-reaching citation
is caught by ``temporal_validity``. The report fixes: every id a reader meets
comes with a name, a comparison that applied no treatment says so instead of
reporting a measured zero, and a regression comes with both answers and a
cause read off the trace.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pheasant_lab.benchmark.question_types import (
    ExpectedEvidence,
    ExpectedFact,
    FactMatcher,
    Question,
)
from pheasant_lab.evaluation import answer_metrics
from pheasant_lab.lifecycle import RunPaths
from pheasant_lab.reports import context as report_context
from pheasant_lab.reports.context import RunContext
from pheasant_lab.reports.regressions import _from_answers
from pheasant_lab.reports.summary import _no_treatment, _plain_words, health_vector
from pheasant_lab.settings import AnswerMatchingSection
from pheasant_lab.textkit import document_body, prose
from pheasant_lab.tracing.lineage import LineageIndex

ARTIFACT = "file:swarm-lab-literature:topic/10.5555_synthetic.dsup.2018.009.md:branch=none"
LATER = "file:swarm-lab-literature:topic/10.5555_synthetic.dsup.2021.022.md:branch=none"
SOURCE, LATER_SOURCE = "source-7ee04b58a1c1d981", "source-25ae995c00b19ca9"
DOCUMENT = (
    "---\nlab_source_id: source-7ee04b58a1c1d981\ntitle: Structural basis\n---\n\n"
    "# Structural basis of nucleosome binding\n\nA cryo-EM structure shows Dsup engages the "
    "acidic patch. Binding does not distort the octamer.\n"
)


# -- prose -------------------------------------------------------------------


def test_prose_reads_what_a_document_says_not_its_metadata_or_markup():
    assert prose(DOCUMENT) == [
        "A cryo-EM structure shows Dsup engages the acidic patch.",
        "Binding does not distort the octamer.",
    ]
    assert "lab_source_id" not in document_body("---\nlab_source_id: x\ntitle: cut off mid-pre")
    memory = "- Dsup binds nucleosomes. [1][2]\n\n[1] [A title](topic/a.md)\n    > A quote here."
    assert prose(memory) == ["Dsup binds nucleosomes.", "A quote here."]


def test_the_extractive_demo_answerer_never_quotes_front_matter_or_a_question():
    from pheasant_lab.models.base import ModelRequest
    from pheasant_lab.models.replay import ReplayProvider

    request = ModelRequest(
        role="test_agent",
        schema="answer",
        system="",
        user="",
        context={
            "question": "What does the structure show about Dsup and the acidic patch?",
            "question_type": "atomic_fact",
            "passages": [
                {"artifact_id": ARTIFACT, "text": DOCUMENT},
                {
                    "artifact_id": "mem",
                    "text": "**Question:** What does Dsup do to the acidic patch?",
                },
            ],
        },
    )
    # The answer schema reads only the request; no provider settings needed.
    claims = ReplayProvider.__new__(ReplayProvider)._schema_answer(request)["claims"]
    assert claims and all("lab_source_id" not in c["text"] for c in claims)
    assert not any(c["text"].rstrip().endswith("?") for c in claims)


# -- scoring -----------------------------------------------------------------


def _question(**overrides) -> Question:
    base = {
        "question_id": "q-1",
        "topic_id": "topic",
        "type": "atomic_fact",
        "text": "Which source establishes that Dsup binds the acidic patch? Name it.",
        "difficulty": "direct",
        "cohorts": ["anchor"],
        "required_fact_ids": ["fact-1"],
    }
    base.update(overrides)
    return Question(**base)  # type: ignore[arg-type]


def _fact(kind="source", **matcher) -> ExpectedFact:
    defaults = (
        {"identifiers": ["10.5555/synthetic.dsup.2018.009", "Structural basis", SOURCE]}
        if kind == "source"
        else matcher
    )
    return ExpectedFact(
        fact_id="fact-1",
        text="the source",
        matcher=FactMatcher(kind=kind, **defaults),  # type: ignore[arg-type]
        source_ids=[SOURCE],
    )


def _answer(text: str, citations: list[str]) -> dict:
    return {
        "answer_id": "a-1",
        "arm_id": "S0",
        "answer_text": text,
        "abstained": False,
        "claims": [{"text": text, "citations": citations}],
        "read_passages": [{"artifact_id": c, "text": text} for c in citations],
    }


def _value(results, metric):
    return next(r.value for r in results if r.metric == metric)


def test_a_correct_citation_names_the_source_however_the_region_spelled_its_id():
    answer = _answer("A cryo-EM structure shows Dsup engages the acidic patch.", [ARTIFACT])
    args = (_question(), [_fact()], ExpectedEvidence(question_id="q-1"), answer)
    blind = answer_metrics.compute(*args, AnswerMatchingSection(), run_id="r")
    joined = answer_metrics.compute(
        *args, AnswerMatchingSection(), run_id="r", citation_sources={ARTIFACT: SOURCE}
    )
    assert _value(blind, "fact_recall") == 0.0, "the DOI is spelled with `_` in the id"
    assert _value(joined, "fact_recall") == 1.0
    # Validity still reads the citation as given.
    assert _value(joined, "citation_validity") == 1.0


def test_temporal_validity_sees_a_citation_of_a_later_source():
    question = _question(
        type="temporal", as_of="2019-01-01", known_negative_source_ids=[LATER_SOURCE]
    )
    fact = ExpectedFact(
        fact_id="fact-1",
        text="Dsup engages the acidic patch",
        matcher=FactMatcher(kind="all_of", groups=[["acidic"], ["patch"]]),
        source_ids=[SOURCE],
    )
    answer = _answer(
        "Dsup engages the acidic patch, as a later study confirmed.", [ARTIFACT, LATER]
    )
    results = answer_metrics.compute(
        question,
        [fact],
        ExpectedEvidence(question_id="q-1"),
        answer,
        AnswerMatchingSection(),
        run_id="r",
        citation_sources={ARTIFACT: SOURCE, LATER: LATER_SOURCE},
    )
    assert _value(results, "temporal_validity") == 0.0


# -- the run context ---------------------------------------------------------


@pytest.fixture
def run_dir(tmp_path: Path) -> RunPaths:
    paths = RunPaths(tmp_path / "run-1")
    for directory in (paths.raw, paths.benchmark, paths.reports):
        directory.mkdir(parents=True, exist_ok=True)

    def write(path: Path, rows: list[dict]) -> None:
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    write(
        paths.raw_file("sources.jsonl"),
        [
            {
                "source_id": SOURCE,
                "artifact_id": ARTIFACT,
                "title": "Structural basis of nucleosome binding by Dsup",
                "authors": ["Wei Chen", "Nadia Haddad"],
                "published_at": "2018-02-11",
            }
        ],
    )
    write(
        paths.benchmark / "questions.jsonl",
        [
            {
                "question_id": "q-1",
                "text": "What does Dsup bind?",
                "type": "atomic_fact",
                "cohorts": ["learned"],
                "required_fact_ids": ["fact-1"],
            },
            {
                "question_id": "q-2",
                "text": "What about the zzq protocol?",
                "type": "abstention",
                "cohorts": ["control"],
                "required_fact_ids": [],
            },
        ],
    )
    write(
        paths.benchmark / "expected-facts.jsonl",
        [{"fact_id": "fact-1", "text": "Dsup binds nucleosomes.", "source_ids": [SOURCE]}],
    )
    write(
        paths.benchmark / "expected-evidence.jsonl",
        [
            {
                "question_id": "q-1",
                "acceptable_artifact_ids": [ARTIFACT],
                "acceptable_source_ids": [SOURCE],
            }
        ],
    )
    write(
        paths.raw_file("memory-records.jsonl"),
        [{"record_id": "mem-abc123", "kind": "fact", "subject": "What does Dsup bind?"}],
    )
    write(
        paths.raw_file("events.jsonl"),
        [{"event_type": "tuning.strategy", "payload": {"expansions": {}, "drop_terms": []}}],
    )
    write(
        paths.raw_file("answers.jsonl"),
        [
            {
                "arm_id": "S0",
                "question_id": "q-1",
                "repetition": 1,
                "answer_text": "Dsup binds nucleosomes.",
                "claims": [{"text": "Dsup binds nucleosomes.", "citations": [SOURCE]}],
            },
            {
                "arm_id": "P1",
                "question_id": "q-1",
                "repetition": 1,
                "answer_text": "Unrelated.",
                "claims": [
                    {
                        "text": "Unrelated.",
                        "citations": ["file:agent-memory:org/mem-abc123.md:branch=none"],
                    }
                ],
                "search_calls": [
                    {"query": "dsup", "results": [{"artifact_id": ARTIFACT, "rank": 2}]}
                ],
            },
            {
                "arm_id": "P0",
                "question_id": "q-1",
                "repetition": 1,
                "answer_text": "Something.",
                "claims": [],
                "search_calls": [{"query": "dsup", "results": []}],
            },
            {
                "arm_id": "P1",
                "question_id": "q-2",
                "repetition": 1,
                "answer_text": "It is X.",
                "abstained": False,
                "claims": [{"text": "It is X.", "citations": [ARTIFACT]}],
            },
        ],
    )
    return paths


def test_every_id_a_reader_meets_has_a_name(run_dir):
    context = RunContext.load(run_dir)
    named = "*Structural basis of nucleosome binding by Dsup* (Chen et al., 2018)"
    assert context.document(SOURCE) == named
    assert context.document(ARTIFACT) == named, "the region's id and the lab's are one document"
    assert context.document("file:agent-memory:org/mem-abc123.md:branch=none") == (
        "memory retrieval proof for “What does Dsup bind?”"
    )
    assert context.document("source-unknown") == "`source-unknown`"
    assert context.question_text("q-1") == "What does Dsup bind?"
    assert context.tuning_applied is False
    assert report_context.arm("P1") == "P1 (Pheasant search + memory)"


def test_a_regression_names_its_cause_from_the_answers(run_dir):
    context = RunContext.load(run_dir)
    unused = _from_answers(context, "q-1", "P1", "S0")
    assert unused and "retrieved *Structural basis" in unused and "did not use it" in unused
    missed = _from_answers(context, "q-1", "P0", "S0")
    assert missed and "never returned a document holding the expected fact" in missed
    answered = _from_answers(context, "q-2", "P1", "S0")
    assert answered and "should have abstained" in answered


def test_the_per_question_report_reads_as_questions_and_titles(run_dir):
    from pheasant_lab.reports.query_detail import write_query_detail

    rows = [
        {
            "question_id": "q-1",
            "arm_id": arm_id,
            "repetition": 1,
            "metric": "fact_recall",
            "value": value,
            "numerator": value,
            "denominator": 1.0,
            "status": "ok",
        }
        for arm_id, value in (("S0", 1.0), ("P1", 0.0))
    ]
    context = RunContext.load(run_dir)
    _csv, markdown = write_query_detail(run_dir, rows, context=context)
    text = markdown.read_text()
    assert "## 1. What does Dsup bind?" in text
    assert "Dsup binds nucleosomes. — from *Structural basis" in text
    assert "| S0 (source-aware specialist) | answered | 1 of 1 |" in text
    assert "file:swarm-lab-literature" not in text and SOURCE not in text


def test_an_untuned_p2_is_reported_as_not_measured(run_dir):
    context = RunContext.load(run_dir)
    assert _no_treatment("P2", "P1", context)
    assert _no_treatment("P1", "P0", context) is None
    row = SimpleNamespace(absolute_delta=0.0)
    result = SimpleNamespace(
        classification_for=lambda *a, **k: row,
        aggregates={},
        gates=SimpleNamespace(sets={}, verdict=SimpleNamespace(value="PASS")),
        non_inferiority={},
    )
    vector = health_vector(
        result,
        run_status="complete",
        collection_sufficient=True,
        receipt_rate=1.0,
        cost_usd=0.0,
        tuning_applied=False,
    )
    assert vector["P2_vs_P1_tuning_gain"] is None, "no treatment is not a measured zero"
    assert vector["P1_vs_P0_memory_gain"] == 0.0


def test_plain_words_say_what_each_comparison_found():
    from pheasant_lab.evaluation.classification import Classification, Status

    rows = {
        ("fact_f1", "P1", "P0"): Classification(
            metric="fact_f1",
            baseline_arm="P0",
            treatment_arm="P1",
            status=Status.improved,
            n=12,
            baseline_mean=0.5,
            treatment_mean=0.75,
            absolute_delta=0.25,
            coverage=1.0,
            relative_delta=0.5,
            threshold=0.05,
            higher_is_better=True,
        ),
        ("fact_recall", "P1", "P0"): Classification(
            metric="fact_recall",
            baseline_arm="P0",
            treatment_arm="P1",
            status=Status.improved,
            n=12,
            baseline_mean=0.5,
            treatment_mean=0.7,
            absolute_delta=0.2,
            coverage=1.0,
            relative_delta=0.4,
            threshold=0.05,
            higher_is_better=True,
        ),
    }
    result = SimpleNamespace(classification_for=lambda m, t, b, c=None: rows.get((m, t, b)))
    text = "\n".join(_plain_words(result, None))
    assert (
        "**Does memory help?** Better, by more than the practical threshold: P1 found 70%" in text
    )
    assert "answer F1 0.50 → 0.75" in text


def test_a_memory_hit_is_not_a_broken_lineage_link(run_dir):
    run_dir.raw_file("questions.jsonl").write_text(
        json.dumps({"question_id": "q-1", "text": "x"}) + "\n", encoding="utf-8"
    )
    answers = [
        {
            "answer_id": "a",
            "arm_id": "P1",
            "question_id": "q-1",
            "search_calls": [{"results": [{"artifact_id": "m", "memory_ids": ["mem-abc123"]}]}],
        }
    ]
    run_dir.raw_file("answers.jsonl").write_text(
        "".join(json.dumps(a) + "\n" for a in answers), encoding="utf-8"
    )
    chains = LineageIndex(run_dir).chain_for_question("q-1")
    assert chains and not [link for chain in chains for link in chain.broken_links]
    assert any(node.kind == "memory_record" for chain in chains for node in chain.nodes)
