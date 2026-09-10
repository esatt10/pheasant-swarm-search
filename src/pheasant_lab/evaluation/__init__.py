"""The evaluation plane.

Measurements are neither observations nor knowledge. Nothing this package
produces is written back into the Pheasant namespace the arms read: a region
must not be able to answer a question with its own report.
"""

from .classification import Classification, classify
from .engine import EvaluationEngine, EvaluationResult
from .gates import GateOutcome, GateSet, GateSetResult, Verdict, evaluate_gates
from .metric import MetricResult, MetricStatus, insufficient
from .pairing import PairedSample, Pairing, pair_arms
from .statistics import (
    benjamini_hochberg,
    bootstrap_interval,
    mcnemar,
    wilcoxon_signed_rank,
)

__all__ = [
    "Classification",
    "EvaluationEngine",
    "EvaluationResult",
    "GateOutcome",
    "GateSet",
    "GateSetResult",
    "MetricResult",
    "MetricStatus",
    "PairedSample",
    "Pairing",
    "Verdict",
    "benjamini_hochberg",
    "bootstrap_interval",
    "classify",
    "evaluate_gates",
    "insufficient",
    "mcnemar",
    "pair_arms",
    "wilcoxon_signed_rank",
]
