"""Freezing the benchmark.

Questions are frozen and content-hashed **before the first test arm runs**.
The manifest digests every file, the source ledger, the generator's model and
configuration, the seed, and the Pheasant snapshot at freeze time - so a
result can be checked against the question set that produced it rather than
against whatever the directory holds now.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..hashing import digest, digest_file, digest_text
from ..lifecycle import RunPaths, isonow
from ..settings import LabConfig
from .builder import BuiltBenchmark
from .question_types import ExpectedEvidence, ExpectedFact, Question

FILES = (
    "questions.jsonl",
    "expected-facts.jsonl",
    "expected-evidence.jsonl",
    "exclusions.jsonl",
    "abstention-cases.jsonl",
    "cohort-membership.jsonl",
)
MANIFEST = "benchmark-manifest.json"


@dataclass
class FreezePackage:
    directory: Path
    manifest: dict[str, Any]
    questions: list[Question] = field(default_factory=list)
    facts: dict[str, ExpectedFact] = field(default_factory=dict)
    evidence: dict[str, ExpectedEvidence] = field(default_factory=dict)

    @property
    def version(self) -> str:
        return str(self.manifest.get("benchmark_version", ""))

    @property
    def digest(self) -> str:
        return str(self.manifest.get("package_digest", ""))

    def question(self, question_id: str) -> Question:
        for question in self.questions:
            if question.question_id == question_id:
                return question
        raise KeyError(question_id)

    def facts_for(self, question: Question) -> list[ExpectedFact]:
        return [self.facts[f] for f in question.required_fact_ids if f in self.facts]

    def evidence_for(self, question: Question) -> ExpectedEvidence:
        return self.evidence.get(
            question.question_id, ExpectedEvidence(question_id=question.question_id)
        )

    def by_cohort(self, cohort: str) -> list[Question]:
        return [q for q in self.questions if cohort in q.cohorts]


def freeze(
    built: BuiltBenchmark,
    paths: RunPaths,
    config: LabConfig,
    *,
    run_id: str,
    source_ledger_digest: str,
    snapshot: dict[str, Any] | None = None,
    prompt_digests: dict[str, str] | None = None,
) -> FreezePackage:
    directory = paths.benchmark
    directory.mkdir(parents=True, exist_ok=True)

    # The frozen files carry no run stamp. They are content-addressed
    # artifacts: two runs over the same corpus must produce byte-identical
    # questions, or the "same question in two runs" that the trend line rests
    # on is not the same question. Which run produced them is in the manifest.
    _write(directory / "questions.jsonl", (q.as_dict() for q in built.questions))
    _write(directory / "expected-facts.jsonl", (f.as_dict() for f in built.facts.values()))
    _write(directory / "expected-evidence.jsonl", (e.as_dict() for e in built.evidence.values()))
    _write(directory / "exclusions.jsonl", (x.as_dict() for x in built.exclusions))
    _write(directory / "abstention-cases.jsonl", (q.as_dict() for q in built.abstention_questions))
    _write(directory / "cohort-membership.jsonl", built.cohort_membership())

    file_digests = {name: digest_file(directory / name) for name in FILES}
    manifest = {
        "schema_version": 1,
        "benchmark_version": built.version,
        "run_id": run_id,
        "topic_id": built.topic_id,
        "created_at": isonow(),
        "files": file_digests,
        "source_ledger_digest": source_ledger_digest,
        "generator": {
            "role": "benchmark_builder",
            "provider": config.role("benchmark_builder").provider,
            "model": config.role("benchmark_builder").model,
            "temperature": config.role("benchmark_builder").temperature,
            "prompt_digests": dict(prompt_digests or {}),
        },
        "seed": config.experiment.seed,
        "composition_requested": dict(built.composition_requested),
        "composition_built": dict(built.composition_built),
        "counts": built.as_summary(),
        "pheasant": {
            "knowledge_base": config.pheasant.knowledge_base,
            "snapshot": snapshot or {},
        },
        "frozen": True,
    }
    # The package digest covers the file digests and the manifest's own
    # content-bearing fields, not its creation stamp: an id has to be a
    # function of exactly the thing it names.
    manifest["package_digest"] = digest(
        {
            "files": file_digests,
            "version": built.version,
            "ledger": source_ledger_digest,
            "composition": manifest["composition_built"],
        }
    )
    (directory / MANIFEST).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    return FreezePackage(
        directory=directory,
        manifest=manifest,
        questions=list(built.questions),
        facts=dict(built.facts),
        evidence=dict(built.evidence),
    )


def load_frozen(paths: RunPaths) -> FreezePackage:
    directory = paths.benchmark
    manifest_path = directory / MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"no frozen benchmark at {directory}. Run `pheasant-lab freeze-benchmark` first: "
            "an evaluation against an unfrozen question set cannot be checked afterwards."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    questions = [Question.from_dict(row) for row in _read(directory / "questions.jsonl")]
    facts = {
        row["fact_id"]: ExpectedFact.from_dict(row)
        for row in _read(directory / "expected-facts.jsonl")
    }
    evidence = {
        row["question_id"]: ExpectedEvidence.from_dict(row)
        for row in _read(directory / "expected-evidence.jsonl")
    }
    return FreezePackage(
        directory=directory, manifest=manifest, questions=questions, facts=facts, evidence=evidence
    )


def verify_freeze(paths: RunPaths) -> list[str]:
    """Re-derive every digest in the manifest. Returns findings, not a bool."""

    findings: list[str] = []
    directory = paths.benchmark
    manifest_path = directory / MANIFEST
    if not manifest_path.is_file():
        return [f"no benchmark manifest at {manifest_path}"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, recorded in (manifest.get("files") or {}).items():
        path = directory / name
        if not path.is_file():
            findings.append(f"benchmark file {name} is named in the manifest and absent")
            continue
        recomputed = digest_file(path)
        if recomputed != recorded:
            findings.append(
                f"benchmark file {name} changed after freeze: {recorded} -> {recomputed}"
            )
    recomputed_package = digest(
        {
            "files": manifest.get("files", {}),
            "version": manifest.get("benchmark_version"),
            "ledger": manifest.get("source_ledger_digest"),
            "composition": manifest.get("composition_built", {}),
        }
    )
    if recomputed_package != manifest.get("package_digest"):
        findings.append("package digest does not match the manifest's own fields")
    return findings


def source_ledger_digest(records: Iterable[dict[str, Any]]) -> str:
    """Digest the evidence ledger a benchmark was built from."""

    return digest(sorted(str(row.get("claim_id") or row.get("source_id") or "") for row in records))


def _write(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    lines = [json.dumps(row, sort_keys=True, ensure_ascii=False, default=str) for row in rows]
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def question_digest(question: Question) -> str:
    return digest_text(question.text)
