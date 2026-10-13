"""Per-question detail.

The acceptance test for this repository is that a reader can start at one
sentence in ``summary.md`` and walk down to a single Pheasant result and its
ingest receipt. This file is the step in the middle: every question, every
arm, every operand, and every exclusion that touched a denominator - and, so
that a person can read it, the question in words, the expected answer and the
documents that hold it by title, and what each arm actually said and cited.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..lifecycle import RunPaths
from ..textkit import truncate
from .context import ARM_NAMES, RunContext, arm

COLUMNS = (
    "question_id",
    "question_type",
    "difficulty",
    "cohort",
    "arm_id",
    "repetition",
    "metric",
    "value",
    "numerator",
    "denominator",
    "status",
    "excluded",
    "answer_id",
    "metric_result_id",
)


def write_query_detail(
    paths: RunPaths,
    per_query: Sequence[Mapping[str, Any]],
    *,
    questions: Mapping[str, Mapping[str, Any]] | None = None,
    lineage: Any = None,
    context: RunContext | None = None,
) -> list[Path]:
    csv_path = paths.reports / "query-detail.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in per_query:
            writer.writerow({column: row.get(column) for column in COLUMNS})

    markdown = paths.reports / "query-detail.md"
    context = context or RunContext(questions={k: dict(v) for k, v in (questions or {}).items()})
    lines = ["# Per-question detail", ""]
    lines.append(
        "One block per question: what was asked, what the right answer is and where it lives, "
        "what each arm answered and cited, and every score behind the aggregates in "
        "`arm-comparison.md`. A score shown as — could not be measured for that arm on that "
        "question; it is not a zero. `query-detail.csv` has the same operands, one per row."
    )
    lines.append("")
    lines.append("Arms: " + " · ".join(f"**{a}** {name}" for a, name in ARM_NAMES.items()))
    lines.append("")

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in per_query:
        grouped.setdefault(str(row.get("question_id")), []).append(row)

    ordered = sorted(grouped.items(), key=lambda item: (context.question_text(item[0]), item[0]))
    for number, (question_id, rows) in enumerate(ordered, start=1):
        lines.extend(_question_block(number, question_id, rows, context))
        if lineage is not None:
            chains = lineage.chain_for_question(question_id)
            broken = [link for chain in chains for link in chain.broken_links]
            if broken:
                lines.append("**Broken lineage links:**")
                for link in sorted(set(broken)):
                    lines.append(f"- {link}")
                lines.append("")
    markdown.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return [csv_path, markdown]


def _question_block(
    number: int, question_id: str, rows: Sequence[Mapping[str, Any]], context: RunContext
) -> list[str]:
    question = context.questions.get(question_id) or {}
    lines = [f"## {number}. {context.question_text(question_id)}", ""]
    facts = [
        f"`{question_id}`",
        str(question.get("type") or "").replace("_", " "),
        f"difficulty {question.get('difficulty')}" if question.get("difficulty") else "",
        "cohort " + ", ".join(question.get("cohorts") or []) if question.get("cohorts") else "",
        f"as of {question['as_of']}" if question.get("as_of") else "",
    ]
    lines.append(" · ".join(part for part in facts if part))
    lines.append("")

    # -- the right answer
    if context.is_abstention(question_id):
        lines.append(
            "**Expected:** the corpus does not answer this. The right response is to say so "
            "(abstain), not to answer."
        )
    else:
        required = context.required_facts(question_id)
        if required:
            lines.append(f"**Expected answer** — {len(required)} required fact(s):")
            for fact in required:
                where = "; ".join(context.document(s) for s in fact.get("source_ids") or [])
                lines.append(f"- {fact.get('text')}" + (f" — from {where}" if where else ""))
        documents = context.expected_documents(question_id)
        if documents and not required:
            lines.append(
                "**Expected evidence:** " + "; ".join(context.document(d) for d in documents)
            )
    lines.append("")

    # -- what each arm did, first repetition
    arms = sorted({str(r.get("arm_id")) for r in rows}, key=_arm_order)
    lines.append("**What each arm answered** (first repetition):")
    lines.append("")
    lines.append(
        "| arm | outcome | required facts found | claims backed by what it cited | documents cited |"
    )
    lines.append("|---|---|---|---|---|")
    texts: list[str] = []
    for arm_id in arms:
        answer = context.answer(arm_id, question_id)
        outcome = context.outcome(answer)
        if context.is_abstention(question_id) and answer is not None:
            outcome += " ✓" if answer.get("abstained") else " ✗ (should have abstained)"
        cited = "; ".join(context.document(c) for c in context.cited(answer)) or "—"
        lines.append(
            f"| {arm(arm_id)} | {outcome} | {_fraction(rows, arm_id, 'fact_recall')} | "
            f"{_fraction(rows, arm_id, 'evidence_support_rate')} | {cited} |"
        )
        if answer and answer.get("answer_text"):
            texts.append(
                f"- **{arm_id}:** {truncate(' '.join(str(answer['answer_text']).split()), 500)}"
            )
        elif answer and answer.get("abstention_reason"):
            texts.append(f"- **{arm_id}:** _abstained — {answer['abstention_reason']}_")
    lines.append("")
    if texts:
        lines.append("<details><summary>Answer text</summary>")
        lines.append("")
        lines.extend(texts)
        lines.append("")
        lines.append("</details>")
        lines.append("")

    # -- every score, metric by arm
    metrics = sorted({str(r.get("metric")) for r in rows})
    lines.append("<details><summary>All scores (mean over repetitions)</summary>")
    lines.append("")
    lines.append("| metric | " + " | ".join(arms) + " |")
    lines.append("|---|" + "---:|" * len(arms))
    for metric in metrics:
        cells = [_mean_cell(rows, arm_id, metric) for arm_id in arms]
        lines.append(f"| `{metric}` | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("</details>")
    lines.append("")
    return lines


def _first(rows: Sequence[Mapping[str, Any]], arm_id: str, metric: str) -> Mapping[str, Any] | None:
    matching = [r for r in rows if r.get("arm_id") == arm_id and r.get("metric") == metric]
    matching.sort(key=lambda r: int(r.get("repetition") or 1))
    return matching[0] if matching else None


def _fraction(rows: Sequence[Mapping[str, Any]], arm_id: str, metric: str) -> str:
    row = _first(rows, arm_id, metric)
    if row is None or row.get("value") is None or not row.get("denominator"):
        return "—"
    return f"{float(row.get('numerator') or 0):g} of {float(row['denominator']):g}"


def _mean_cell(rows: Sequence[Mapping[str, Any]], arm_id: str, metric: str) -> str:
    values = [
        float(r["value"])
        for r in rows
        if r.get("arm_id") == arm_id and r.get("metric") == metric and r.get("value") is not None
    ]
    if not values:
        present = any(r.get("arm_id") == arm_id and r.get("metric") == metric for r in rows)
        return "—" if present else ""
    mean = sum(values) / len(values)
    return f"{mean:.3g}" + (f" (n={len(values)})" if len(values) > 1 else "")


def _arm_order(arm_id: str) -> tuple[int, str]:
    order = list(ARM_NAMES)
    return (order.index(arm_id) if arm_id in order else len(order), arm_id)
