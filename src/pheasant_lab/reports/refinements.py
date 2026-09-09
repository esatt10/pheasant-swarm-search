"""Refinement candidates.

Errors and regressions projected into things somebody could actually change,
each carrying the evidence behind it and a frequency. A candidate below the
configured minimum frequency is not emitted: a "pattern" seen once is a
coincidence with a ticket number.

These are a **report artifact** by default. Submitting them into a Pheasant
namespace is a separate, explicit act, and never into the namespace ordinary
retrieval reads - a region that can retrieve its own diagnostics can answer a
question with its own report.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import ids
from ..evaluation.engine import EvaluationResult
from ..hashing import digest
from ..lifecycle import RunPaths

TARGETS = (
    "ingestion",
    "chunking",
    "metadata",
    "graph",
    "retrieval",
    "memory",
    "mcp_adapter",
    "orchestration",
)
TRIGGERS = (
    "repeated_error",
    "retrieval_miss",
    "unsupported_claim",
    "negative_exposure",
    "latency",
    "duplication",
    "drift",
)
RISK = ("low", "medium", "high")


@dataclass
class RefinementCandidate:
    candidate_id: str
    target: str
    trigger_type: str
    evidence_refs: list[str] = field(default_factory=list)
    frequency: int = 0
    affected_query_ids: list[str] = field(default_factory=list)
    affected_source_ids: list[str] = field(default_factory=list)
    suspected_cause: str = ""
    proposed_change: str = ""
    expected_metric: str = ""
    risk_class: str = "low"
    status: str = "proposed"

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "target": self.target,
            "trigger_type": self.trigger_type,
            "evidence_refs": list(self.evidence_refs),
            "frequency": self.frequency,
            "affected_query_ids": list(self.affected_query_ids),
            "affected_source_ids": list(self.affected_source_ids),
            "suspected_cause": self.suspected_cause,
            "proposed_change": self.proposed_change,
            "expected_metric": self.expected_metric,
            "risk_class": self.risk_class,
            "status": self.status,
        }


def build_candidates(
    result: EvaluationResult,
    *,
    errors: Sequence[Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]] = (),
    duplicate_rate: float | None = None,
    minimum_frequency: int = 2,
) -> list[RefinementCandidate]:
    candidates: list[RefinementCandidate] = []

    # 1. repeated errors, clustered by (component, operation, class)
    clusters: Counter = Counter()
    evidence: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for error in errors:
        key = (str(error.get("component")), str(error.get("operation")), str(error.get("class")))
        clusters[key] += 1
        evidence[key].append(str(error.get("error_id")))
    for (component, operation, error_class), count in clusters.most_common():
        if count < minimum_frequency:
            continue
        candidates.append(
            _candidate(
                target="mcp_adapter" if component.startswith("pheasant") else "orchestration",
                trigger="repeated_error",
                frequency=count,
                evidence=evidence[(component, operation, error_class)][:20],
                cause=f"{count} `{error_class}` failures in {component}.{operation}",
                change=(
                    f"inspect {component}.{operation}: either the operation is not idempotent and is "
                    "being retried, or the region is refusing an argument this adapter sends"
                ),
                metric="mcp_success_rate",
                risk="low",
            )
        )

    # 2. retrieval misses: the known positive was never surfaced
    misses: dict[str, list[str]] = defaultdict(list)
    for row in result.per_query:
        if row.get("metric") != "known_positive_hit_at_k":
            continue
        if row.get("value") == 0.0:
            misses[str(row.get("arm_id"))].append(str(row.get("question_id")))
    for arm, questions in sorted(misses.items()):
        if len(questions) < minimum_frequency:
            continue
        candidates.append(
            _candidate(
                target="retrieval",
                trigger="retrieval_miss",
                frequency=len(questions),
                evidence=questions[:20],
                cause=f"`{arm}` surfaced no known positive on {len(questions)} questions",
                change=(
                    "check whether the missed documents are indexed at all before tuning ranking: a "
                    "document the arms never saw is not a fusion failure, and no ranking parameter "
                    "reaches it"
                ),
                metric="known_positive_recall_at_k",
                risk="medium",
                queries=questions,
            )
        )

    # 3. unsupported claims
    unsupported = [
        row
        for row in result.per_query
        if row.get("metric") == "unsupported_claim_rate" and (row.get("value") or 0) > 0.5
    ]
    if len(unsupported) >= minimum_frequency:
        candidates.append(
            _candidate(
                target="retrieval",
                trigger="unsupported_claim",
                frequency=len(unsupported),
                evidence=[str(row.get("metric_result_id")) for row in unsupported][:20],
                cause=f"{len(unsupported)} answers rested more than half on material they did not cite",
                change=(
                    "raise the passage budget or the over-fetch before changing the answerer: an arm "
                    "with nothing to cite will answer from its prior"
                ),
                metric="evidence_support_rate",
                risk="medium",
                queries=[str(row.get("question_id")) for row in unsupported],
            )
        )

    # 4. negative exposure moving the wrong way
    exposure = result.classification_for("negative_exposure_at_k", "P1", "P0")
    if exposure is not None and (exposure.absolute_delta or 0) > 0:
        candidates.append(
            _candidate(
                target="memory",
                trigger="negative_exposure",
                frequency=exposure.n,
                evidence=["classification:negative_exposure_at_k:P1-P0"],
                cause=f"memory raised known-negative exposure by {exposure.absolute_delta:+.4f}",
                change=(
                    "review the seeded steering records: a preference rule that fires on the wrong "
                    "subject promotes the wrong material without changing the corpus"
                ),
                metric="negative_exposure_at_k",
                risk="high",
            )
        )

    # 5. ingest failures
    failed = [
        receipt
        for receipt in receipts
        if str(receipt.get("status")) not in {"accepted", "indexed", "verified"}
    ]
    if len(failed) >= minimum_frequency:
        reasons = Counter(str(receipt.get("error_code") or "unknown") for receipt in failed)
        candidates.append(
            _candidate(
                target="ingestion",
                trigger="repeated_error",
                frequency=len(failed),
                evidence=[str(receipt.get("receipt_id")) for receipt in failed][:20],
                cause=f"{len(failed)} submissions carry no acceptance: {dict(reasons)}",
                change="check the submission shape against the region's ingest schema before re-running",
                metric="ingest_receipt_rate",
                risk="high",
                sources=[str(receipt.get("source_id")) for receipt in failed],
            )
        )

    # 6. duplication
    if duplicate_rate is not None and duplicate_rate > 0.25:
        candidates.append(
            _candidate(
                target="orchestration",
                trigger="duplication",
                frequency=int(duplicate_rate * 100),
                evidence=["collection_metrics:duplicate_rate"],
                cause=f"{duplicate_rate:.0%} of acquired items were duplicates",
                change=(
                    "widen the terminology cluster rather than deepening rounds: a high duplicate "
                    "rate means the plan is searching one vocabulary"
                ),
                metric="duplicate_rate",
                risk="low",
            )
        )
    return candidates


def _candidate(
    *,
    target: str,
    trigger: str,
    frequency: int,
    evidence: Sequence[str],
    cause: str,
    change: str,
    metric: str,
    risk: str,
    queries: Sequence[str] = (),
    sources: Sequence[str] = (),
) -> RefinementCandidate:
    return RefinementCandidate(
        candidate_id=ids.candidate_id(target, trigger, digest(sorted(evidence))),
        target=target,
        trigger_type=trigger,
        evidence_refs=list(evidence),
        frequency=frequency,
        affected_query_ids=sorted(set(queries)),
        affected_source_ids=sorted(set(sources)),
        suspected_cause=cause,
        proposed_change=change,
        expected_metric=metric,
        risk_class=risk,
    )


def write_refinements(
    paths: RunPaths, candidates: Sequence[RefinementCandidate], *, submitted: bool = False
) -> list[Path]:
    jsonl = paths.metrics / "refinement-candidates.jsonl"
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    import json

    with jsonl.open("w", encoding="utf-8") as handle:
        for candidate in candidates:
            handle.write(json.dumps(candidate.as_dict(), sort_keys=True) + "\n")

    lines = ["# Refinement candidates", ""]
    lines.append(
        "Each candidate is a *proposal* with its evidence attached. Nothing here has been tested, "
        "and nothing here has been written into the knowledge region "
        + (
            "(submission to an isolated diagnostic namespace was enabled for this run)"
            if submitted
            else "(submission is off, which is the default)"
        )
        + "."
    )
    lines.append("")
    if not candidates:
        lines.append("_this run produced no candidate above the configured minimum frequency_")
        lines.append("")
        lines.append(
            "That is a result, not an absence of one: a run that proposes nothing has said the "
            "failures it saw are not reachable by the changes it knows how to suggest."
        )
        return [jsonl, _write(paths.reports / "refinement-candidates.md", lines)]

    for candidate in sorted(candidates, key=lambda c: (-c.frequency, c.candidate_id)):
        lines.append(
            f"## `{candidate.candidate_id}` — {candidate.target} / {candidate.trigger_type}"
        )
        lines.append("")
        lines.append(f"- frequency: **{candidate.frequency}**")
        lines.append(f"- risk: `{candidate.risk_class}` · status: `{candidate.status}`")
        lines.append(f"- expected to move: `{candidate.expected_metric}`")
        lines.append(f"- suspected cause: {candidate.suspected_cause}")
        lines.append(f"- proposed change: {candidate.proposed_change}")
        if candidate.affected_query_ids:
            lines.append(
                f"- affected questions ({len(candidate.affected_query_ids)}): "
                f"{', '.join('`' + q + '`' for q in candidate.affected_query_ids[:8])}"
            )
        if candidate.affected_source_ids:
            lines.append(f"- affected sources ({len(candidate.affected_source_ids)})")
        lines.append(f"- evidence: {', '.join('`' + e + '`' for e in candidate.evidence_refs[:6])}")
        lines.append("")
    return [jsonl, _write(paths.reports / "refinement-candidates.md", lines)]


def _write(path: Path, lines: Sequence[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path
