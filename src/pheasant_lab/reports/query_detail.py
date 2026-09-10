"""Per-question detail.

The acceptance test for this repository is that a reader can start at one
sentence in ``summary.md`` and walk down to a single Pheasant result and its
ingest receipt. This file is the step in the middle: every question, every
arm, every operand, and every exclusion that touched a denominator.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..lifecycle import RunPaths

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
) -> list[Path]:
    csv_path = paths.reports / "query-detail.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in per_query:
            writer.writerow({column: row.get(column) for column in COLUMNS})

    markdown = paths.reports / "query-detail.md"
    lines = ["# Per-question detail", ""]
    lines.append(
        "One block per question. Every number here is the operand of an aggregate in "
        "`arm-comparison.md`; a number with `insufficient_evidence` has no value by design, and "
        "is not a zero."
    )
    lines.append("")

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in per_query:
        grouped.setdefault(str(row.get("question_id")), []).append(row)

    for question_id, rows in sorted(grouped.items()):
        question = (questions or {}).get(question_id, {})
        lines.append(f"## `{question_id}`")
        lines.append("")
        if question:
            lines.append(f"> {question.get('text')}")
            lines.append("")
            lines.append(
                f"- type `{question.get('type')}` · difficulty `{question.get('difficulty')}` · "
                f"cohorts {question.get('cohorts')}"
            )
            if question.get("as_of"):
                lines.append(f"- as_of: `{question['as_of']}`")
            lines.append(f"- required facts: {len(question.get('required_fact_ids') or [])}")
            lines.append("")
        lines.append("| arm | metric | value | numerator | denominator | status |")
        lines.append("|---|---|---:|---:|---:|---|")
        for row in sorted(rows, key=lambda r: (str(r.get("arm_id")), str(r.get("metric")))):
            value = row.get("value")
            lines.append(
                f"| `{row.get('arm_id')}` | `{row.get('metric')}` | "
                f"{'—' if value is None else f'{float(value):.4g}'} | "
                f"{row.get('numerator')} | {row.get('denominator')} | {row.get('status')} |"
            )
        lines.append("")
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
