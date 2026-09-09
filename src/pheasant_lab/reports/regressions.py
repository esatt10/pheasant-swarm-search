"""``worst-regressions.md`` and ``errors-and-retries.md``.

The regressions report is ordered by how much a question got *worse*, not by
how badly it scored: a question both arms fail is not a regression, and
putting it at the top of this list buries the ones that are.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..evaluation.engine import EvaluationResult
from ..evaluation.pairing import pair_arms
from ..lifecycle import RunPaths


def write_regressions(
    paths: RunPaths,
    result: EvaluationResult,
    *,
    questions: Mapping[str, Mapping[str, Any]],
    metric: str = "fact_f1",
    limit: int = 20,
) -> Path:
    lines = ["# Worst regressions", ""]
    lines.append(
        f"Per-question deltas on `{metric}`, worst first. A question both arms fail is not a "
        "regression and is not listed here."
    )
    lines.append("")

    for treatment, baseline in (("P1", "P0"), ("P2", "P1"), ("P1", "S0"), ("P2", "S0")):
        pairing = pair_arms(
            result.per_query,
            metric=metric,
            baseline_arm=baseline,
            treatment_arm=treatment,
            questions=questions,
        )
        if not pairing.samples:
            continue
        worst = sorted(pairing.samples, key=lambda sample: sample.delta)[:limit]
        worst = [sample for sample in worst if sample.delta < 0]
        lines.append(f"## `{treatment}` − `{baseline}`")
        lines.append("")
        if not worst:
            lines.append("_no question regressed on this comparison_")
            lines.append("")
            continue
        lines.append("| question | cohort | type | baseline | treatment | Δ |")
        lines.append("|---|---|---|---:|---:|---:|")
        for sample in worst:
            lines.append(
                f"| `{sample.question_id}` | {sample.cohort} | {sample.question_type} | "
                f"{sample.baseline:.4g} | {sample.treatment:.4g} | {sample.delta:+.4g} |"
            )
        lines.append("")
        lines.append("**Likely causes, from the trace:**")
        for sample in worst[:5]:
            lines.append(
                f"- `{sample.question_id}`: {_diagnose(result, sample.question_id, treatment, baseline)}"
            )
        lines.append("")
    return _write(paths.reports / "worst-regressions.md", lines)


def _diagnose(result: EvaluationResult, question_id: str, treatment: str, baseline: str) -> str:
    """A trace-backed guess at *why*, stated as a guess."""

    def value(metric: str, arm: str) -> float | None:
        rows = [
            row
            for row in result.per_query
            if row.get("question_id") == question_id
            and row.get("arm_id") == arm
            and row.get("metric") == metric
        ]
        values = [float(row["value"]) for row in rows if row.get("value") is not None]
        return sum(values) / len(values) if values else None

    recall_t, recall_b = (
        value("known_positive_recall_at_k", treatment),
        value("known_positive_recall_at_k", baseline),
    )
    support_t, support_b = (
        value("evidence_support_rate", treatment),
        value("evidence_support_rate", baseline),
    )
    exposure_t, exposure_b = (
        value("negative_exposure_at_k", treatment),
        value("negative_exposure_at_k", baseline),
    )

    if recall_t is not None and recall_b is not None and recall_t < recall_b:
        return "retrieval reached less of the known-positive set (a retrieval cause, not an answering one)"
    if exposure_t is not None and exposure_b is not None and exposure_t > exposure_b:
        return "more known-negative material was shown, which is consistent with a steering rule firing wrongly"
    if support_t is not None and support_b is not None and support_t < support_b:
        return "retrieval was comparable but less of the answer rested on what was retrieved"
    return "no single stage explains this from the trace; the per-question detail has the operands"


def write_errors_report(
    paths: RunPaths,
    errors: Sequence[Mapping[str, Any]],
    *,
    mcp_reliability: Mapping[str, Any] | None = None,
    latency: Mapping[str, Any] | None = None,
) -> Path:
    lines = ["# Errors, retries and exclusions", ""]
    if not errors:
        lines.append("_no error was recorded in this run_")
        lines.append("")
    else:
        by_class = Counter(str(error.get("class")) for error in errors)
        by_stage = Counter(str(error.get("stage")) for error in errors)
        lines.append("## Counts")
        lines.append("")
        lines.append("| class | count |")
        lines.append("|---|---:|")
        for name, count in sorted(by_class.items()):
            lines.append(f"| `{name}` | {count} |")
        lines.append("")
        lines.append("| stage | count |")
        lines.append("|---|---:|")
        for name, count in sorted(by_stage.items()):
            lines.append(f"| `{name}` | {count} |")
        lines.append("")

        retried = [error for error in errors if error.get("resolution") == "retried"]
        lines.append(
            f"{len(retried)} attempt(s) were retried and are recorded here even where the retry "
            "succeeded: a run that only recorded terminal failures would look healthier the "
            "flakier it was."
        )
        lines.append("")

        invalidating = [
            error
            for error in errors
            if (error.get("impact") or {}).get("comparability") == "invalidated"
        ]
        if invalidating:
            lines.append("## Comparability-invalidating errors")
            lines.append("")
            for error in invalidating:
                lines.append(
                    f"- `{error.get('error_id')}` — {error.get('component')}.{error.get('operation')}: "
                    f"{error.get('message_redacted')}"
                )
            lines.append("")

        excluded = [
            error for error in errors if (error.get("impact") or {}).get("excluded_from_metrics")
        ]
        lines.append("## Exclusions from metric denominators")
        lines.append("")
        if not excluded:
            lines.append("_no error removed a question from a denominator_")
        for error in excluded:
            queries = (error.get("impact") or {}).get("affected_queries") or []
            lines.append(
                f"- `{error.get('error_id')}` ({error.get('stage')}/{error.get('class')}): "
                f"{len(queries)} question(s) affected"
            )
        lines.append("")

        lines.append("## Every recorded attempt")
        lines.append("")
        lines.append(
            "| id | stage | class | component | operation | attempt | retryable | resolution |"
        )
        lines.append("|---|---|---|---|---|---:|---|---|")
        for error in errors[:200]:
            lines.append(
                f"| `{error.get('error_id')}` | {error.get('stage')} | {error.get('class')} | "
                f"{error.get('component')} | {error.get('operation')} | {error.get('attempt')} | "
                f"{error.get('retryable')} | {error.get('resolution')} |"
            )
        lines.append("")

    if mcp_reliability:
        lines.append("## MCP reliability")
        lines.append("")
        for key, value in sorted(mcp_reliability.items()):
            lines.append(f"- `{key}`: {value}")
        lines.append("")
        lines.append(
            "> `partial_rate` is reported on its own. A partial response is partial; folding it "
            "into success is how a truncated result set becomes a corpus gap in a report."
        )
        lines.append("")

    if latency:
        lines.append("## Latency by operation")
        lines.append("")
        lines.append("| operation | count | p50 ms | p95 ms | max ms |")
        lines.append("|---|---:|---:|---:|---:|")
        for name, stats in sorted(latency.items()):
            lines.append(
                f"| `{name}` | {stats.get('count'):.0f} | {_ms(stats.get('p50_ms'))} | "
                f"{_ms(stats.get('p95_ms'))} | {_ms(stats.get('max_ms'))} |"
            )
        lines.append("")
    return _write(paths.reports / "errors-and-retries.md", lines)


def _ms(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f}"


def _write(path: Path, lines: Sequence[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path
