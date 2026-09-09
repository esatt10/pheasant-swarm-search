"""Pairing.

A question enters a paired comparison only when **both** arms completed under
the pairing policy. Partial runs exclude both sides consistently: an arm that
failed on the hard questions would otherwise be compared on the easy ones and
look better for having failed.

Repetitions are collapsed before pairing, by median, so a metric's paired unit
is the *question* rather than the run - otherwise three repetitions of one
question would count as three independent observations in every interval and
every test.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

PairingPolicy = Literal["complete_pairs_only", "available_pairs"]


@dataclass
class PairedSample:
    question_id: str
    cohort: str | None
    baseline: float
    treatment: float
    baseline_answer_ids: list[str] = field(default_factory=list)
    treatment_answer_ids: list[str] = field(default_factory=list)
    question_type: str | None = None
    difficulty: str | None = None

    @property
    def delta(self) -> float:
        return self.treatment - self.baseline


@dataclass
class Pairing:
    metric: str
    baseline_arm: str
    treatment_arm: str
    samples: list[PairedSample] = field(default_factory=list)
    excluded: dict[str, list[str]] = field(default_factory=dict)
    eligible_questions: int = 0

    @property
    def n(self) -> int:
        return len(self.samples)

    @property
    def coverage(self) -> float | None:
        if not self.eligible_questions:
            return None
        return self.n / self.eligible_questions

    @property
    def deltas(self) -> list[float]:
        return [sample.delta for sample in self.samples]

    def by_subgroup(self, key: str) -> dict[str, list[PairedSample]]:
        buckets: dict[str, list[PairedSample]] = defaultdict(list)
        for sample in self.samples:
            value = getattr(sample, key, None) or (sample.cohort if key == "cohort" else None)
            buckets[str(value)].append(sample)
        return dict(sorted(buckets.items()))

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "baseline_arm": self.baseline_arm,
            "treatment_arm": self.treatment_arm,
            "n": self.n,
            "eligible_questions": self.eligible_questions,
            "coverage": self.coverage,
            "excluded": {reason: sorted(ids) for reason, ids in sorted(self.excluded.items())},
        }


def collapse_repetitions(
    per_query: Sequence[Mapping[str, Any]], metric: str, arm_id: str
) -> dict[str, tuple[float, list[str]]]:
    """One value per question, from however many repetitions ran.

    Median rather than mean: a single failed repetition should not drag a
    question's value in a way the other two disagree with, and the median of
    an even count is still deterministic given a fixed ordering.
    """

    buckets: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for row in per_query:
        if row.get("metric") != metric or row.get("arm_id") != arm_id:
            continue
        if row.get("status") != "ok" or row.get("value") is None:
            continue
        buckets[str(row["question_id"])].append(
            (float(row["value"]), str(row.get("answer_id") or ""))
        )
    out: dict[str, tuple[float, list[str]]] = {}
    for question_id, values in buckets.items():
        ordered = sorted(values, key=lambda pair: (pair[0], pair[1]))
        out[question_id] = (
            statistics.median(value for value, _ in ordered),
            [answer_id for _value, answer_id in ordered if answer_id],
        )
    return out


def pair_arms(
    per_query: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    baseline_arm: str,
    treatment_arm: str,
    questions: Mapping[str, Mapping[str, Any]],
    policy: PairingPolicy = "complete_pairs_only",
    restrict_to: Sequence[str] | None = None,
) -> Pairing:
    baseline = collapse_repetitions(per_query, metric, baseline_arm)
    treatment = collapse_repetitions(per_query, metric, treatment_arm)
    wanted = set(restrict_to) if restrict_to is not None else set(questions)

    pairing = Pairing(metric=metric, baseline_arm=baseline_arm, treatment_arm=treatment_arm)
    pairing.eligible_questions = len(wanted)

    for question_id in sorted(wanted):
        has_baseline = question_id in baseline
        has_treatment = question_id in treatment
        if not has_baseline or not has_treatment:
            reason = (
                "missing_in_both"
                if not has_baseline and not has_treatment
                else ("missing_in_baseline" if not has_baseline else "missing_in_treatment")
            )
            if policy == "complete_pairs_only":
                pairing.excluded.setdefault(reason, []).append(question_id)
                continue
            pairing.excluded.setdefault(reason, []).append(question_id)
            continue
        question = questions.get(question_id, {})
        pairing.samples.append(
            PairedSample(
                question_id=question_id,
                cohort=(question.get("cohorts") or [None])[0],
                baseline=baseline[question_id][0],
                treatment=treatment[question_id][0],
                baseline_answer_ids=baseline[question_id][1],
                treatment_answer_ids=treatment[question_id][1],
                question_type=question.get("type"),
                difficulty=question.get("difficulty"),
            )
        )
    return pairing
