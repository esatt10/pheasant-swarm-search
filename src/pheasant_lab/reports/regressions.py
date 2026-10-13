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
from ..textkit import truncate
from .context import RunContext, arm, listing


def write_regressions(
    paths: RunPaths,
    result: EvaluationResult,
    *,
    questions: Mapping[str, Mapping[str, Any]],
    metric: str = "fact_f1",
    limit: int = 20,
    context: RunContext | None = None,
) -> Path:
    context = context or RunContext(questions={k: dict(v) for k, v in questions.items()})
    lines = ["# Worst regressions", ""]
    lines.append(
        f"The questions that got worse on `{metric}`, worst first, with what each arm said and "
        "the most likely reason from the trace. A question both arms fail is not a regression "
        "and is not listed here."
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
        lines.append(f"## {arm(treatment)} vs {arm(baseline)}")
        lines.append("")
        if not worst:
            lines.append("_no question regressed on this comparison_")
            lines.append("")
            continue
        lines.append(f"| question | cohort | type | {baseline} | {treatment} | Δ |")
        lines.append("|---|---|---|---:|---:|---:|")
        for sample in worst:
            lines.append(
                f"| {context.question_text(sample.question_id, 90)} | {sample.cohort} | "
                f"{str(sample.question_type).replace('_', ' ')} | {sample.baseline:.4g} | "
                f"{sample.treatment:.4g} | {sample.delta:+.4g} |"
            )
        lines.append("")
        for sample in worst[:5]:
            lines.extend(_case(result, context, sample.question_id, treatment, baseline))
    return _write(paths.reports / "worst-regressions.md", lines)


def _case(
    result: EvaluationResult,
    context: RunContext,
    question_id: str,
    treatment: str,
    baseline: str,
) -> list[str]:
    lines = [f"### {context.question_text(question_id)}", "", f"`{question_id}`", ""]
    if context.is_abstention(question_id):
        lines.append("- **expected:** abstain — the corpus does not answer this")
    for fact in context.required_facts(question_id):
        lines.append(f"- **expected:** {fact.get('text')}")
    for arm_id in (baseline, treatment):
        answer = context.answer(arm_id, question_id)
        outcome = context.outcome(answer)
        said = ""
        if answer and answer.get("answer_text") and not answer.get("abstained"):
            said = f": {truncate(' '.join(str(answer['answer_text']).split()), 300)}"
        cited = listing([context.document(c) for c in context.cited(answer)], limit=4)
        lines.append(f"- **{arm_id}** {outcome}{said}" + (f" — cited {cited}" if cited else ""))
    lines.append(
        f"- **likely cause:** {_diagnose(result, question_id, treatment, baseline, context)}"
    )
    lines.append("")
    return lines


def _diagnose(
    result: EvaluationResult,
    question_id: str,
    treatment: str,
    baseline: str,
    context: RunContext | None = None,
) -> str:
    """A trace-backed guess at *why*, stated as a guess.

    The answers themselves come first - did the treatment fail, abstain, or
    answer what it should have declined; did its searches reach a document
    holding the expected fact, and did it then use it. Those read straight off
    the trace for any pair of arms, including against the specialist, which
    has no retrieval metrics. The metric comparison is the fallback.
    """

    if context is not None:
        reason = _from_answers(context, question_id, treatment, baseline)
        if reason:
            return reason

    def value(metric: str, arm_id: str) -> float | None:
        rows = [
            row
            for row in result.per_query
            if row.get("question_id") == question_id
            and row.get("arm_id") == arm_id
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


def _from_answers(
    context: RunContext, question_id: str, treatment: str, baseline: str
) -> str | None:
    answer = context.answer(treatment, question_id)
    other = context.answer(baseline, question_id)
    if answer is None:
        return None
    if answer.get("status") == "failed":
        return f"{treatment}'s answer failed: {answer.get('error') or 'no error recorded'}"
    if context.is_abstention(question_id):
        if not answer.get("abstained"):
            cited = listing([context.document(c) for c in context.cited(answer)]) or "nothing"
            return (
                f"{treatment} answered a question the corpus cannot answer, where it should have "
                f"abstained (it cited {cited})"
            )
        return None
    if answer.get("abstained") and other is not None and not other.get("abstained"):
        reason = answer.get("abstention_reason")
        return f"{treatment} abstained where {baseline} answered" + (
            f" ({reason})" if reason else ""
        )
    expected = context.expected_documents(question_id)
    if not expected or not answer.get("search_calls"):
        return None
    retrieved = context.retrieved(answer)
    found = [
        (key, rank, query)
        for key, rank, query in retrieved
        if any(context.same_document(key, e) for e in expected)
    ]
    titles = "; ".join(context.document(e) for e in _distinct(context, expected)[:2])
    if not found:
        return (
            f"{treatment}'s searches never returned a document holding the expected fact "
            f"({titles}) - a retrieval miss, not an answering one"
        )
    cited = context.cited(answer)
    if not any(context.same_document(c, key) for c in cited for key, _, _ in found):
        key, rank, query = found[0]
        return (
            f'{treatment} retrieved {context.document(key)} (rank {rank} for "{query}") but its '
            "answer did not use it - an answering cause, not a retrieval one"
        )
    return (
        f"{treatment} retrieved and cited the right document, but its answer did not state the "
        "expected fact in matchable words"
    )


def _distinct(context: RunContext, ids: Sequence[str]) -> list[str]:
    out: list[str] = []
    for key in ids:
        if not any(context.same_document(key, seen) for seen in out):
            out.append(key)
    return out


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
