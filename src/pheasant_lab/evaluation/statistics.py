"""Paired statistical diagnostics.

Pure Python, no SciPy: the arithmetic is small, and a dependency that changes
a p-value between versions is a dependency this repository cannot audit.

These are **diagnostics**. They do not rescue low evidence coverage or
violated independence, and no classification in this lab is decided by a
p-value alone - the practical threshold and the gates come first.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass
class Interval:
    lower: float
    upper: float
    level: float
    method: str
    resamples: int

    @property
    def excludes_zero(self) -> bool:
        return self.lower > 0.0 or self.upper < 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "lower": self.lower,
            "upper": self.upper,
            "level": self.level,
            "method": self.method,
            "resamples": self.resamples,
            "excludes_zero": self.excludes_zero,
        }


@dataclass
class TestResult:
    test: str
    statistic: float | None
    p_value: float | None
    n: int
    detail: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "test": self.test,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "n": self.n,
            **self.detail,
        }


def bootstrap_interval(
    deltas: Sequence[float],
    *,
    resamples: int = 2000,
    level: float = 0.95,
    seed: int = 0,
    statistic: str = "mean",
) -> Interval | None:
    """Seeded paired bootstrap over the per-question deltas.

    Seeded so the interval is reproducible: an interval that moves between two
    runs of the same numbers is not a diagnostic, it is noise with a decimal
    point.
    """

    values = [float(delta) for delta in deltas]
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    aggregate = (lambda sample: sum(sample) / len(sample)) if statistic == "mean" else _median
    draws: list[float] = []
    size = len(values)
    for _ in range(resamples):
        draws.append(aggregate([values[rng.randrange(size)] for _ in range(size)]))
    draws.sort()
    tail = (1.0 - level) / 2.0
    lower = draws[max(0, int(tail * len(draws)) - 1)]
    upper = draws[min(len(draws) - 1, int((1.0 - tail) * len(draws)))]
    return Interval(
        lower=lower,
        upper=upper,
        level=level,
        method=f"paired_bootstrap_{statistic}",
        resamples=resamples,
    )


def _median(sample: Sequence[float]) -> float:
    ordered = sorted(sample)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def mcnemar(baseline: Sequence[bool], treatment: Sequence[bool]) -> TestResult:
    """Exact McNemar over paired binary outcomes.

    The exact binomial rather than the chi-square approximation: the
    discordant counts here are routinely single digits, which is exactly where
    the approximation is worst.
    """

    if len(baseline) != len(treatment):
        raise ValueError("paired sequences must be the same length")
    b = sum(1 for base, treat in zip(baseline, treatment, strict=True) if base and not treat)
    c = sum(1 for base, treat in zip(baseline, treatment, strict=True) if treat and not base)
    n = b + c
    if n == 0:
        return TestResult(
            "mcnemar_exact",
            None,
            None,
            len(baseline),
            {"b": b, "c": c, "note": "no discordant pairs"},
        )
    smaller = min(b, c)
    tail = sum(math.comb(n, k) for k in range(smaller + 1)) / (2.0**n)
    p_value = min(1.0, 2.0 * tail)
    return TestResult(
        "mcnemar_exact",
        statistic=float(smaller),
        p_value=p_value,
        n=len(baseline),
        detail={"b_baseline_only": b, "c_treatment_only": c, "discordant": n},
    )


def wilcoxon_signed_rank(deltas: Sequence[float]) -> TestResult:
    """Wilcoxon signed-rank, normal approximation with tie correction.

    Zero differences are dropped (Wilcoxon's own convention) and the count
    after dropping is reported, because it - not the sample size - is what the
    test actually used.
    """

    nonzero = [float(delta) for delta in deltas if delta != 0.0]
    n = len(nonzero)
    if n < 6:
        return TestResult(
            "wilcoxon_signed_rank",
            None,
            None,
            n,
            {"note": f"{n} non-zero differences is too few for the normal approximation"},
        )
    ordered = sorted(nonzero, key=abs)
    ranks: list[float] = [0.0] * n
    index = 0
    while index < n:
        stop = index
        while stop + 1 < n and abs(ordered[stop + 1]) == abs(ordered[index]):
            stop += 1
        average = (index + stop + 2) / 2.0
        for position in range(index, stop + 1):
            ranks[position] = average
        index = stop + 1

    positive = sum(rank for value, rank in zip(ordered, ranks, strict=True) if value > 0)
    negative = sum(rank for value, rank in zip(ordered, ranks, strict=True) if value < 0)
    statistic = min(positive, negative)
    mean = n * (n + 1) / 4.0
    tie_correction = 0.0
    index = 0
    while index < n:
        stop = index
        while stop + 1 < n and abs(ordered[stop + 1]) == abs(ordered[index]):
            stop += 1
        group = stop - index + 1
        tie_correction += group**3 - group
        index = stop + 1
    variance = (n * (n + 1) * (2 * n + 1) - tie_correction / 2.0) / 24.0
    if variance <= 0:
        return TestResult("wilcoxon_signed_rank", statistic, None, n, {"note": "zero variance"})
    z = (statistic - mean + 0.5) / math.sqrt(variance)
    p_value = 2.0 * _normal_cdf(z)
    return TestResult(
        "wilcoxon_signed_rank",
        statistic=statistic,
        p_value=min(1.0, max(0.0, p_value)),
        n=n,
        detail={"positive_rank_sum": positive, "negative_rank_sum": negative, "z": z},
    )


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def effect_size(deltas: Sequence[float]) -> dict[str, Any]:
    """Wins, ties, losses and a standardised effect, reported together.

    Raw counts are published beside the standardised number because a Cohen's
    d over twelve questions is a number with a false air of precision, and the
    counts are what a reader can check.
    """

    values = [float(delta) for delta in deltas]
    wins = sum(1 for delta in values if delta > 0)
    losses = sum(1 for delta in values if delta < 0)
    ties = len(values) - wins - losses
    mean = sum(values) / len(values) if values else 0.0
    if len(values) > 1:
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        deviation = math.sqrt(variance)
    else:
        deviation = 0.0
    return {
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "mean_delta": mean,
        "sd_delta": deviation,
        "cohens_dz": (mean / deviation) if deviation else None,
        "rank_biserial": ((wins - losses) / len(values)) if values else None,
    }


def benjamini_hochberg(p_values: Sequence[float | None], *, fdr: float = 0.10) -> list[bool]:
    """Return, per input, whether it survives BH at ``fdr``.

    ``None`` p-values (a test that could not run) are never "significant" and
    do not enter the ranking - counting them would change every other
    threshold in the family.
    """

    indexed = [(index, p) for index, p in enumerate(p_values) if p is not None]
    if not indexed:
        return [False] * len(p_values)
    indexed.sort(key=lambda pair: pair[1])
    total = len(indexed)
    cutoff_rank = 0
    for rank, (_index, p) in enumerate(indexed, start=1):
        if p <= fdr * rank / total:
            cutoff_rank = rank
    survivors = {index for index, _p in indexed[:cutoff_rank]}
    return [index in survivors for index in range(len(p_values))]
