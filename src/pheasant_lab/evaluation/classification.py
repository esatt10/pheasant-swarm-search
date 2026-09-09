"""Turning a paired delta into a sentence somebody can act on.

Six statuses, and the boundaries between them are configured rather than
inferred at report time. Every classification names the practical threshold it
used - a report that says "improved" without saying "by more than 5 points"
has said less than it appears to.

``mixed`` is the status that earns its place: an aggregate that improved while
a protected subgroup regressed is not an improvement, and calling it one is
how a change ships that makes the hard questions worse.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from ..settings import ClassificationSection, StatisticsSection
from .pairing import Pairing
from .statistics import (
    Interval,
    TestResult,
    bootstrap_interval,
    effect_size,
    mcnemar,
    wilcoxon_signed_rank,
)


class Status(StrEnum):
    improved = "improved"
    regressed = "regressed"
    unchanged = "unchanged"
    mixed = "mixed"
    insufficient_evidence = "insufficient_evidence"
    not_comparable = "not_comparable"


@dataclass
class SubgroupResult:
    key: str
    value: str
    n: int
    delta: float
    status: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "n": self.n,
            "delta": self.delta,
            "status": self.status,
        }


@dataclass
class Classification:
    metric: str
    baseline_arm: str
    treatment_arm: str
    status: Status
    n: int
    coverage: float | None
    baseline_mean: float | None
    treatment_mean: float | None
    absolute_delta: float | None
    relative_delta: float | None
    threshold: float
    higher_is_better: bool
    interval: Interval | None = None
    tests: list[TestResult] = field(default_factory=list)
    effect: dict[str, Any] = field(default_factory=dict)
    subgroups: list[SubgroupResult] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    cohort: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "baseline_arm": self.baseline_arm,
            "treatment_arm": self.treatment_arm,
            "cohort": self.cohort,
            "status": self.status.value,
            "n": self.n,
            "coverage": self.coverage,
            "baseline_mean": self.baseline_mean,
            "treatment_mean": self.treatment_mean,
            "absolute_delta": self.absolute_delta,
            "relative_delta": self.relative_delta,
            "practical_threshold": self.threshold,
            "higher_is_better": self.higher_is_better,
            "interval": self.interval.as_dict() if self.interval else None,
            "tests": [test.as_dict() for test in self.tests],
            "effect": dict(self.effect),
            "subgroups": [subgroup.as_dict() for subgroup in self.subgroups],
            "reasons": list(self.reasons),
        }


def classify(
    pairing: Pairing,
    *,
    classification: ClassificationSection,
    statistics_config: StatisticsSection,
    relative_lift_epsilon: float = 0.02,
    gate_failures: Sequence[str] = (),
    comparability_findings: Sequence[str] = (),
    cohort: str | None = None,
) -> Classification:
    metric = pairing.metric
    threshold = classification.threshold_for(metric)
    higher_is_better = classification.higher_is_better(metric)
    n = pairing.n
    coverage = pairing.coverage

    baseline_mean = sum(s.baseline for s in pairing.samples) / n if n else None
    treatment_mean = sum(s.treatment for s in pairing.samples) / n if n else None
    absolute = (treatment_mean - baseline_mean) if (n and baseline_mean is not None) else None
    relative = (
        absolute / abs(baseline_mean)
        if absolute is not None
        and baseline_mean is not None
        and abs(baseline_mean) >= relative_lift_epsilon
        else None
    )

    result = Classification(
        metric=metric,
        baseline_arm=pairing.baseline_arm,
        treatment_arm=pairing.treatment_arm,
        status=Status.insufficient_evidence,
        n=n,
        coverage=coverage,
        baseline_mean=baseline_mean,
        treatment_mean=treatment_mean,
        absolute_delta=absolute,
        relative_delta=relative,
        threshold=threshold,
        higher_is_better=higher_is_better,
        cohort=cohort,
    )

    if comparability_findings:
        result.status = Status.not_comparable
        result.reasons = [f"not comparable: {finding}" for finding in comparability_findings]
        return result

    if n < classification.minimum_paired_questions:
        result.reasons.append(
            f"{n} paired questions is below the configured minimum of "
            f"{classification.minimum_paired_questions}"
        )
        return result
    if coverage is not None and coverage < classification.minimum_pairing_coverage:
        result.reasons.append(
            f"pairing coverage {coverage:.0%} is below the configured minimum of "
            f"{classification.minimum_pairing_coverage:.0%}"
        )
        return result

    deltas = pairing.deltas
    result.effect = effect_size(deltas)
    if statistics_config.enabled and n >= statistics_config.minimum_n_for_tests:
        result.interval = bootstrap_interval(
            deltas,
            resamples=statistics_config.bootstrap_resamples,
            level=statistics_config.confidence_level,
            seed=statistics_config.bootstrap_seed,
        )
        if "wilcoxon" in statistics_config.paired_tests:
            result.tests.append(wilcoxon_signed_rank(deltas))
        if "mcnemar" in statistics_config.paired_tests and _binary(pairing):
            result.tests.append(
                mcnemar(
                    [sample.baseline >= 0.5 for sample in pairing.samples],
                    [sample.treatment >= 0.5 for sample in pairing.samples],
                )
            )

    # Direction is normalised so the rest of this function reads the same for
    # both directions; the report prints the raw signed delta.
    directed = absolute if higher_is_better else -(absolute or 0.0)
    assert directed is not None

    if gate_failures:
        result.status = Status.regressed
        result.reasons.append(f"hard gate failure: {', '.join(sorted(gate_failures))}")
        return result

    result.subgroups = _subgroups(pairing, classification, threshold, higher_is_better)
    regressed_subgroups = [s for s in result.subgroups if s.status == "regressed"]

    if directed <= -threshold:
        result.status = Status.regressed
        result.reasons.append(f"paired delta {absolute:+.4g} is at or below -{threshold}")
        return result

    if directed >= threshold:
        if (
            classification.require_interval_excludes_zero
            and result.interval
            and not result.interval.excludes_zero
        ):
            result.status = Status.unchanged
            result.reasons.append(
                f"paired delta {absolute:+.4g} clears the threshold but the "
                f"{result.interval.level:.0%} interval "
                f"[{result.interval.lower:+.4g}, {result.interval.upper:+.4g}] includes zero"
            )
            # A regressed protected subgroup is reported whatever the
            # aggregate did. Returning `unchanged` here would let a real
            # subgroup regression leave the report entirely, because the
            # aggregate happened not to clear its interval.
            if regressed_subgroups:
                result.status = Status.mixed
                result.reasons.append(
                    "and a protected subgroup regressed: "
                    + ", ".join(f"{s.key}={s.value} ({s.delta:+.4g})" for s in regressed_subgroups)
                )
            return result
        if regressed_subgroups:
            result.status = Status.mixed
            result.reasons.append(
                "aggregate improved but "
                + ", ".join(
                    f"{s.key}={s.value} regressed ({s.delta:+.4g})" for s in regressed_subgroups
                )
            )
            return result
        result.status = Status.improved
        result.reasons.append(f"paired delta {absolute:+.4g} is at or above +{threshold}")
        return result

    result.status = Status.unchanged
    result.reasons.append(
        f"|paired delta {absolute:+.4g}| is below the practical threshold {threshold}"
    )
    if regressed_subgroups:
        result.status = Status.mixed
        result.reasons.append(
            "and a protected subgroup regressed: "
            + ", ".join(f"{s.key}={s.value} ({s.delta:+.4g})" for s in regressed_subgroups)
        )
    return result


def _binary(pairing: Pairing) -> bool:
    return all(
        value in (0.0, 1.0)
        for sample in pairing.samples
        for value in (sample.baseline, sample.treatment)
    )


def _subgroups(
    pairing: Pairing,
    classification: ClassificationSection,
    threshold: float,
    higher_is_better: bool,
) -> list[SubgroupResult]:
    out: list[SubgroupResult] = []
    for key in classification.protected_subgroups:
        for value, samples in pairing.by_subgroup(key).items():
            if len(samples) < 3:
                # Three questions is not a subgroup finding; it is three
                # questions. Reported as `unknown` rather than as a regression.
                out.append(
                    SubgroupResult(
                        key, value, len(samples), _mean(samples), "insufficient_evidence"
                    )
                )
                continue
            delta = _mean(samples)
            directed = delta if higher_is_better else -delta
            status = (
                "regressed"
                if directed <= -threshold
                else ("improved" if directed >= threshold else "unchanged")
            )
            out.append(SubgroupResult(key, value, len(samples), delta, status))
    return out


def _mean(samples: Sequence[Any]) -> float:
    return sum(sample.delta for sample in samples) / len(samples) if samples else 0.0


def non_inferior(
    classification_result: Classification,
    *,
    margin: float,
) -> tuple[bool | None, str]:
    """Is the treatment no worse than the baseline by more than ``margin``?

    Deliberately not "did it score higher": the question this lab asks is
    whether a Pheasant-only agent *rivals* the specialist, and a numeric win is
    neither required nor sufficient. ``None`` means the interval needed to
    answer was not available.
    """

    if classification_result.status is Status.not_comparable:
        return (None, "arms are not comparable")
    if classification_result.interval is None:
        return (None, "no confidence interval was computed (too few paired questions)")
    lower = (
        classification_result.interval.lower
        if classification_result.higher_is_better
        else -classification_result.interval.upper
    )
    if lower >= -margin:
        return (True, f"lower paired bound {lower:+.4g} is no worse than the margin -{margin}")
    return (False, f"lower paired bound {lower:+.4g} is worse than the margin -{margin}")


def apply_multiple_comparison(
    results: Sequence[Classification],
    *,
    statistics_config: StatisticsSection,
) -> dict[str, bool]:
    """Benjamini-Hochberg across the family of reported comparisons."""

    from .statistics import benjamini_hochberg

    if statistics_config.multiple_comparison_correction != "benjamini_hochberg":
        return {}
    p_values: list[float | None] = []
    keys: list[str] = []
    for result in results:
        p = next((test.p_value for test in result.tests if test.p_value is not None), None)
        p_values.append(p)
        keys.append(
            f"{result.metric}:{result.baseline_arm}->{result.treatment_arm}:{result.cohort or 'all'}"
        )
    survives = benjamini_hochberg(p_values, fdr=statistics_config.false_discovery_rate)
    return dict(zip(keys, survives, strict=True))


def comparability_findings(
    manifest_a: Mapping[str, Any],
    manifest_b: Mapping[str, Any],
) -> list[str]:
    """Enumerate every differing resolved field between two runs.

    Two runs are comparable only when this is empty, or when a reader has read
    what it lists. It is deliberately a list rather than a boolean.
    """

    findings: list[str] = []
    for field_name in ("config_digest", "seed", "arms", "package_version"):
        if manifest_a.get(field_name) != manifest_b.get(field_name):
            findings.append(
                f"{field_name}: {manifest_a.get(field_name)!r} vs {manifest_b.get(field_name)!r}"
            )
    models_a = manifest_a.get("models") or {}
    models_b = manifest_b.get("models") or {}
    for role in sorted(set(models_a) | set(models_b)):
        if models_a.get(role) != models_b.get(role):
            findings.append(f"models.{role} differs")
    return findings
