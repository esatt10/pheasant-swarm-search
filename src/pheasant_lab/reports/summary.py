"""``summary.md``, ``collection.md`` and ``arm-comparison.md``.

For every important delta the human report states: what changed, the baseline
and treatment, the numerator and denominator, the absolute and relative
change, the pairing and evidence coverage, the uncertainty diagnostics where
they are eligible, the worst regressions, the likely trace-backed cause, and -
last and most important - what the evidence supports and what it does not.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..evaluation.classification import Classification
from ..evaluation.engine import COMPARISONS, EvaluationResult
from ..evaluation.gates import Verdict
from ..lifecycle import RunPaths, isonow

HEALTH_KEYS = (
    "run_status",
    "collection_sufficient",
    "evidence_coverage",
    "pheasant_ingest_receipt_rate",
    "P0_vs_C0_fact_f1_delta",
    "P1_vs_P0_memory_gain",
    "P2_vs_P1_tuning_gain",
    "P2_specialist_gap",
    "specialist_noninferiority",
    "negative_exposure_delta",
    "unsupported_claim_rate",
    "control_regression_rate",
    "hard_gates",
    "cost_usd",
)


def health_vector(
    result: EvaluationResult,
    *,
    run_status: str,
    collection_sufficient: bool,
    receipt_rate: float | None,
    cost_usd: float,
) -> dict[str, Any]:
    """The compact vector at the top of the report. Never one score."""

    def delta(
        metric: str, treatment: str, baseline: str, cohort: str | None = None
    ) -> float | None:
        row = result.classification_for(metric, treatment, baseline, cohort)
        return row.absolute_delta if row else None

    coverage = None
    proof = result.aggregates.get("proof") or {}
    judged = proof.get("judged_targets")
    events = proof.get("events")
    if judged is not None and events:
        coverage = round(1.0 - (proof.get("unknown_events", 0) / events), 4)

    unsupported = (result.aggregates.get("by_arm", {}).get("P1", {}) or {}).get(
        "unsupported_claim_rate", {}
    )
    control_gate = None
    for gate_set in result.gates.sets.values():
        for outcome in gate_set.outcomes:
            if outcome.gate == "control_regression":
                control_gate = outcome.observed

    noninferiority = result.non_inferiority.get("P2") or result.non_inferiority.get("P1") or {}

    return {
        "run_status": run_status,
        "collection_sufficient": collection_sufficient,
        "evidence_coverage": coverage,
        "pheasant_ingest_receipt_rate": receipt_rate,
        "P0_vs_C0_fact_f1_delta": delta("fact_f1", "P0", "C0"),
        "P1_vs_P0_memory_gain": delta("fact_f1", "P1", "P0"),
        "P2_vs_P1_tuning_gain": delta("fact_f1", "P2", "P1"),
        "P2_specialist_gap": delta("fact_f1", "P2", "S0"),
        "specialist_noninferiority": noninferiority.get("status", "insufficient_evidence"),
        "negative_exposure_delta": delta("negative_exposure_at_k", "P1", "P0"),
        "unsupported_claim_rate": unsupported.get("mean"),
        "control_regression_rate": control_gate,
        "hard_gates": result.gates.verdict.value,
        "cost_usd": round(cost_usd, 4),
    }


def write_summary(
    paths: RunPaths,
    result: EvaluationResult,
    *,
    manifest: Mapping[str, Any],
    vector: Mapping[str, Any],
    collection: Mapping[str, Any] | None = None,
    limitations: Sequence[str] = (),
) -> Path:
    lines: list[str] = []
    lines.append(f"# {manifest.get('experiment_name', 'run')} — summary")
    lines.append("")
    lines.append(f"Run `{manifest.get('run_id')}` · rendered {isonow()}")
    lines.append("")
    lines.append("```yaml")
    for key in HEALTH_KEYS:
        lines.append(f"{key}: {_yaml(vector.get(key))}")
    lines.append("```")
    lines.append("")

    lines.append("## What this run can and cannot say")
    lines.append("")
    lines.append(
        "This report answers five separate questions and does not blend them. "
        "Corpus similarity is not truth, exposure is not success, and a score computed over "
        "sparse evidence is not accuracy — where the evidence was not there, the number below is "
        "`insufficient_evidence` with no value, never `0.0`."
    )
    lines.append("")

    verdict = result.gates.verdict
    lines.append(f"**Hard gates: {verdict.value}.**")
    for name, gate_set in sorted(result.gates.sets.items()):
        lines.append(f"- `{name}`: {gate_set.heading()}")
        for outcome in gate_set.outcomes:
            if outcome.passed is False:
                lines.append(f"  - **FAIL** `{outcome.gate}`: {outcome.detail}")
            elif not outcome.evaluated:
                lines.append(f"  - skipped `{outcome.gate}`: {outcome.skip_reason}")
    if verdict is Verdict.incomplete:
        lines.append("")
        lines.append(
            "> An `INCOMPLETE` verdict is not a pass. A skipped gate and a failed gate are equally "
            "disqualifying for a result somebody will publish."
        )
    lines.append("")

    lines.append("## Does a Pheasant-only agent rival the specialist?")
    lines.append("")
    if not result.non_inferiority:
        lines.append("Not evaluated: the specialist arm or a Pheasant arm did not run.")
    for arm, decision in sorted(result.non_inferiority.items()):
        answer = decision.get("rivals_specialist")
        label = {True: "yes", False: "no", None: "insufficient evidence"}[answer]
        lines.append(f"### `{arm}` vs `S0`: **{label}**")
        lines.append("")
        for metric, body in sorted((decision.get("metrics") or {}).items()):
            overall = (body or {}).get("overall") or {}
            holdout = (body or {}).get("holdout") or {}
            lines.append(
                f"- `{metric}` (margin {body.get('margin')}): overall {overall.get('non_inferior')} "
                f"— {overall.get('reason')}; holdout {holdout.get('non_inferior')} — {holdout.get('reason')}"
            )
        for reason in decision.get("reasons") or []:
            lines.append(f"- blocked by: {reason}")
        lines.append("")

    lines.append("## Headline comparisons")
    lines.append("")
    lines.append(_comparison_table(result))
    lines.append("")

    lines.append("## Limitations of this run")
    lines.append("")
    for limitation in list(result.limitations) + list(limitations):
        lines.append(f"- {limitation}")
    lines.append(
        "- Learned-cohort improvement is **recall of learned experience**, not generalization. "
        "The temporal holdout is the cohort that speaks to generalization, and it is reported "
        "separately everywhere it appears."
    )
    lines.append(
        "- Model-based judging is off. Nothing in the primary metrics rests on a model's opinion "
        "of another model's answer."
    )
    lines.append("")

    if collection:
        lines.append("## Collection")
        lines.append("")
        lines.append(
            f"- decision: **{collection.get('decision', {}).get('outcome')}** — "
            f"{collection.get('decision', {}).get('reason')}"
        )
        summary = collection.get("summary") or {}
        lines.append(
            f"- {summary.get('sources_retained')} retained sources, {summary.get('claims_eligible')} "
            f"eligible claims, {summary.get('rounds')} rounds"
        )
        lines.append("- see `collection.md` for the facet-by-facet picture")
        lines.append("")

    lines.append("## Where to look next")
    lines.append("")
    lines.append("- `arm-comparison.md` — every paired delta with its operands and diagnostics")
    lines.append("- `worst-regressions.md` — the questions that got worse, with their traces")
    lines.append("- `errors-and-retries.md` — every failure, retry and exclusion")
    lines.append(
        "- `refinement-candidates.md` — what this run suggests changing, and on what evidence"
    )
    lines.append("- `../metrics/per-query.jsonl` — every operand behind every number above")
    lines.append("")

    return _write(paths.reports / "summary.md", lines)


def _comparison_table(result: EvaluationResult) -> str:
    rows = [
        "| comparison | metric | n | baseline | treatment | Δ | rel Δ | status | interval |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    wanted = {(treatment, baseline) for treatment, baseline, _ in COMPARISONS}
    for row in result.classifications:
        if row.cohort is not None or (row.treatment_arm, row.baseline_arm) not in wanted:
            continue
        interval = (
            f"[{row.interval.lower:+.3g}, {row.interval.upper:+.3g}]" if row.interval else "—"
        )
        rows.append(
            f"| `{row.treatment_arm}` − `{row.baseline_arm}` | `{row.metric}` | {row.n} | "
            f"{_num(row.baseline_mean)} | {_num(row.treatment_mean)} | {_num(row.absolute_delta, sign=True)} | "
            f"{_pct(row.relative_delta)} | {row.status.value} | {interval} |"
        )
    return "\n".join(rows) if len(rows) > 2 else "_no paired comparison produced enough evidence_"


def write_arm_comparison(paths: RunPaths, result: EvaluationResult) -> Path:
    lines = ["# Arm comparison", ""]
    lines.append(
        "Every delta below is **paired per question**: a question enters a comparison only when "
        "both arms completed it. The practical threshold that decided each status is printed with "
        'it, because "improved" without a threshold has said less than it appears to.'
    )
    lines.append("")
    for treatment, baseline, meaning in COMPARISONS:
        rows = [
            row
            for row in result.classifications
            if row.treatment_arm == treatment and row.baseline_arm == baseline
        ]
        if not rows:
            continue
        lines.append(f"## `{treatment}` − `{baseline}`")
        lines.append("")
        lines.append(f"_{meaning}_")
        lines.append("")
        for row in sorted(rows, key=lambda r: (r.metric, r.cohort or "")):
            lines.extend(_classification_block(row))
        lines.append("")
    return _write(paths.reports / "arm-comparison.md", lines)


def _classification_block(row: Classification) -> list[str]:
    cohort = f" · cohort `{row.cohort}`" if row.cohort else " · all cohorts"
    lines = [f"### `{row.metric}`{cohort} — **{row.status.value}**", ""]
    lines.append(f"- baseline `{row.baseline_arm}` mean: {_num(row.baseline_mean)}")
    lines.append(f"- treatment `{row.treatment_arm}` mean: {_num(row.treatment_mean)}")
    lines.append(
        f"- absolute delta: {_num(row.absolute_delta, sign=True)}"
        + (
            " percentage points"
            if row.metric.endswith(("rate", "recall", "precision", "f1"))
            else ""
        )
    )
    lines.append(
        f"- relative delta: {_pct(row.relative_delta)}"
        + (
            ""
            if row.relative_delta is not None
            else " (`not_applicable`: baseline magnitude below epsilon)"
        )
    )
    lines.append(
        f"- paired questions: {row.n}" + (f" · coverage {row.coverage:.0%}" if row.coverage else "")
    )
    lines.append(
        f"- practical threshold: {row.threshold} ({'higher' if row.higher_is_better else 'lower'} is better)"
    )
    if row.interval:
        lines.append(
            f"- {row.interval.level:.0%} paired bootstrap interval: "
            f"[{row.interval.lower:+.4g}, {row.interval.upper:+.4g}] "
            f"({'excludes' if row.interval.excludes_zero else 'includes'} zero, "
            f"{row.interval.resamples} resamples)"
        )
    for test in row.tests:
        lines.append(
            f"- {test.test}: statistic {test.statistic}, p = {_num(test.p_value)}, n = {test.n}"
        )
    if row.effect:
        lines.append(
            f"- wins/ties/losses: {row.effect.get('wins')}/{row.effect.get('ties')}/{row.effect.get('losses')}"
            f" · dz = {_num(row.effect.get('cohens_dz'))}"
        )
    regressed = [s for s in row.subgroups if s.status == "regressed"]
    if regressed:
        lines.append("- **protected subgroups that regressed:**")
        for subgroup in regressed:
            lines.append(
                f"  - {subgroup.key} = `{subgroup.value}` ({subgroup.n} questions): {subgroup.delta:+.4g}"
            )
    for reason in row.reasons:
        lines.append(f"- {reason}")
    lines.append("")
    return lines


def write_collection_report(
    paths: RunPaths,
    collection: Mapping[str, Any],
    collection_metrics: Sequence[Mapping[str, Any]] = (),
) -> Path:
    decision = collection.get("decision") or {}
    audit = collection.get("audit") or {}
    lines = ["# Collection", ""]
    lines.append(f"**Decision: `{decision.get('outcome')}`** — {decision.get('reason')}")
    lines.append("")
    lines.append(
        "`sufficient` is only reachable when every hard condition passes. Budget or clock "
        "exhaustion reports itself as such and is never renamed."
    )
    lines.append("")
    lines.append("## Hard conditions")
    lines.append("")
    lines.append("| condition | met | detail | threshold |")
    lines.append("|---|---|---|---|")
    for condition in decision.get("conditions") or []:
        lines.append(
            f"| `{condition['id']}` | {'yes' if condition['met'] else '**no**'} | "
            f"{condition['detail']} | {condition.get('threshold')} |"
        )
    lines.append("")

    lines.append("## Facet coverage")
    lines.append("")
    lines.append(
        "| facet | weight | sources | families | peer-reviewed | authoritative | claims "
        "| meets minimum |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for row in audit.get("facet_coverage") or []:
        lines.append(
            f"| {row['facet_id']} | {row['weight']} | {row['sources']} | {row['families']} | "
            f"{row['peer_reviewed']} | {row.get('authoritative', row['peer_reviewed'])} | "
            f"{row.get('claims', 0)} | "
            f"{'yes' if row['meets_minimum'] else '**no** — ' + ', '.join(row.get('unmet', []))} |"
        )
    lines.append("")
    lines.append(
        "> Six papers from one lab are one family. A source count that looks healthy beside a "
        "family count of 1 is concentration, not coverage."
    )
    lines.append("")

    gaps = audit.get("quality_gaps") or []
    if gaps:
        lines.append("## Source-quality gaps")
        lines.append("")
        for gap in gaps:
            lines.append(f"- `{gap['facet_id']}` — **{gap['kind']}**: {gap['detail']}")
        lines.append("")

    yields = audit.get("marginal_claim_yield") or []
    if yields:
        lines.append("## Saturation")
        lines.append("")
        lines.append(
            f"Marginal unique-claim yield by round: {', '.join(f'{y:.3f}' for y in yields)}"
        )
        lines.append("")
        lines.append(
            "> A declining yield means this search direction is exhausted. It is not evidence that "
            "the collection is good."
        )
        lines.append("")

    if collection_metrics:
        lines.append("## Collection metrics")
        lines.append("")
        for metric in collection_metrics:
            lines.extend(_metric_block(metric))
    return _write(paths.reports / "collection.md", lines)


def _metric_block(metric: Mapping[str, Any]) -> list[str]:
    value = metric.get("value")
    lines = [
        f"### `{metric['metric']}` — "
        + (f"{value:.4g}" if isinstance(value, int | float) else f"`{metric.get('status')}`"),
        "",
        f"- formula: `{metric.get('formula')}`",
        f"- substituted: `{metric.get('substituted') or 'n/a'}`",
        f"- numerator / denominator: {metric.get('numerator')} / {metric.get('denominator')}",
    ]
    if metric.get("excluded"):
        lines.append(f"- excluded: {metric['excluded']} ({metric.get('exclusion_reasons')})")
    lines.append(f"- supports: {metric.get('claim_supported')}")
    lines.append(f"- does **not** support: {metric.get('claim_not_supported')}")
    lines.append(f"- limitation: {metric.get('limitation')}")
    lines.append("")
    return lines


def _write(path: Path, lines: Sequence[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def _num(value: float | None, *, sign: bool = False) -> str:
    if value is None:
        return "—"
    return f"{value:+.4g}" if sign else f"{value:.4g}"


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:+.1%}"


def _yaml(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
