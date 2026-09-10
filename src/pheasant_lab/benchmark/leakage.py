"""Leakage checks.

A leakage finding invalidates the affected comparison until the benchmark is
re-frozen. Five checks, each answering a different way the question set can
already contain what the arms are supposed to discover:

1. the question's own wording contains its expected answer;
2. a question matches a research prompt, a memory record or a tuned alias;
3. a holdout question created the memory or tuning it is used to test;
4. an abstention case has retrievable supporting evidence;
5. benchmark answer files reached the Pheasant namespace.

Check 3 is the one that matters most and is the easiest to lose: without it,
``learned - holdout`` stops being a memorisation detector and becomes a
restatement of the learned number.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..textkit import containment, content_terms, normalise, tokens
from .question_types import ExpectedFact, Question

SEVERITIES = ("invalidating", "material", "advisory")
FUZZY_THRESHOLD = 0.85

# Words a question uses to ask rather than to name its subject. They are
# excluded from the abstention check, because "what does the corpus contain
# about X" would otherwise never match a document - no document says "corpus"
# - and the check would pass on every abstention case regardless.
QUESTION_FRAMING = frozenset(
    {
        "corpus",
        "knowledge",
        "base",
        "region",
        "contain",
        "contains",
        "collected",
        "literature",
        "establish",
        "establishes",
        "database",
        "index",
        "indexed",
    }
)
# How many of a question's distinctive terms one document must carry before
# the abstention case is called answerable.
ABSTENTION_TERMS = 3


@dataclass
class LeakageFinding:
    check: str
    severity: str
    question_id: str | None
    detail: str
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def invalidating(self) -> bool:
        return self.severity == "invalidating"

    def as_dict(self) -> dict[str, Any]:
        return {
            "check": self.check,
            "severity": self.severity,
            "question_id": self.question_id,
            "detail": self.detail,
            "evidence": self.evidence,
        }


@dataclass
class LeakageReport:
    findings: list[LeakageFinding] = field(default_factory=list)
    checked: dict[str, int] = field(default_factory=dict)

    @property
    def clean(self) -> bool:
        return not any(finding.invalidating for finding in self.findings)

    @property
    def invalidated_questions(self) -> set[str]:
        return {f.question_id for f in self.findings if f.invalidating and f.question_id}

    def as_dict(self) -> dict[str, Any]:
        return {
            "clean": self.clean,
            "checked": dict(self.checked),
            "findings": [f.as_dict() for f in self.findings],
            "invalidated_questions": sorted(q for q in self.invalidated_questions if q),
        }


def check_leakage(
    questions: Sequence[Question],
    facts: Mapping[str, ExpectedFact],
    *,
    research_prompts: Iterable[str] = (),
    memory_records: Iterable[Mapping[str, Any]] = (),
    tuned_aliases: Iterable[str] = (),
    corpus_texts: Mapping[str, str] | None = None,
    namespace_paths: Iterable[str] = (),
    benchmark_filenames: Iterable[str] = (),
) -> LeakageReport:
    report = LeakageReport()
    prompts = [normalise(p) for p in research_prompts]
    records = list(memory_records)
    aliases = [normalise(a) for a in tuned_aliases]

    report.checked = {
        "questions": len(questions),
        "research_prompts": len(prompts),
        "memory_records": len(records),
        "tuned_aliases": len(aliases),
        "corpus_documents": len(corpus_texts or {}),
    }

    for question in questions:
        folded = normalise(question.text)
        question_tokens = set(tokens(question.text))

        # 1. the answer inside the question
        for fact_id in question.required_fact_ids:
            fact = facts.get(fact_id)
            if fact is None or fact.matcher.kind == "abstain":
                continue
            if fact.matcher.matches(question.text):
                report.findings.append(
                    LeakageFinding(
                        "answer_in_question",
                        "invalidating",
                        question.question_id,
                        "the question's own wording satisfies its expected fact",
                        {"fact_id": fact_id},
                    )
                )

        # 2. the question already exists as a prompt, memory or alias
        for prompt in prompts:
            if folded and folded in prompt:
                report.findings.append(
                    LeakageFinding(
                        "question_in_research_prompt",
                        "material",
                        question.question_id,
                        "a research prompt contains this question verbatim",
                    )
                )
                break
        for alias in aliases:
            if alias and containment(tokens(alias), question_tokens) >= FUZZY_THRESHOLD:
                report.findings.append(
                    LeakageFinding(
                        "question_matches_tuned_alias",
                        "material",
                        question.question_id,
                        f"a tuned alias covers this question's terms: {alias!r}",
                    )
                )

        # 3. a holdout question created what it is used to test
        if "temporal_holdout" in question.cohorts or "control" in question.cohorts:
            for record in records:
                origin = str(record.get("originating_question_id") or "")
                if origin == question.question_id:
                    report.findings.append(
                        LeakageFinding(
                            "holdout_created_treatment",
                            "invalidating",
                            question.question_id,
                            "a holdout or control question created the memory it is used to test",
                            {"record_id": record.get("record_id")},
                        )
                    )
                text = normalise(str(record.get("text") or ""))
                if text and containment(tokens(text), question_tokens) >= FUZZY_THRESHOLD:
                    report.findings.append(
                        LeakageFinding(
                            "holdout_covered_by_memory",
                            "invalidating" if "control" in question.cohorts else "material",
                            question.question_id,
                            "a memory record covers this holdout or control question's terms",
                            {"record_id": record.get("record_id")},
                        )
                    )

        # 4. an abstention case with retrievable support
        if question.type == "abstention" and corpus_texts:
            terms = [
                term
                for term in content_terms(question.text)
                if len(term) > 4 and term not in QUESTION_FRAMING
            ][:ABSTENTION_TERMS]
            for artifact_id, body in corpus_texts.items():
                haystack = set(tokens(body))
                if len(terms) == ABSTENTION_TERMS and all(term in haystack for term in terms):
                    report.findings.append(
                        LeakageFinding(
                            "abstention_answerable",
                            "invalidating",
                            question.question_id,
                            "an indexed document carries this abstention case's terms",
                            {"artifact_id": artifact_id},
                        )
                    )
                    break

    # 5. benchmark files in the namespace
    benchmark_names = set(benchmark_filenames) or {
        "questions.jsonl",
        "expected-facts.jsonl",
        "expected-evidence.jsonl",
        "abstention-cases.jsonl",
        "benchmark-manifest.json",
    }
    for path in namespace_paths:
        leaf = str(path).rsplit("/", 1)[-1]
        if leaf in benchmark_names:
            report.findings.append(
                LeakageFinding(
                    "benchmark_in_namespace",
                    "invalidating",
                    None,
                    f"a benchmark file reached the Pheasant namespace: {path}",
                    {"path": str(path)},
                )
            )
    return report
