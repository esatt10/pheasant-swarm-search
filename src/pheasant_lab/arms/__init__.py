"""The comparison arms.

Five ship: ``S0`` source-aware specialist, ``C0`` prior-only control, ``P0``
Pheasant corpus baseline, ``P1`` Pheasant with memory and steering, ``P2``
tuned-search replay.

The isolation rules are enforced in :mod:`pheasant_lab.arms.base`, not left to
each arm to remember: a Pheasant arm receives the question, the response
schema and the capability map, and nothing else. Not the source URLs, not the
topic plan, not the specialist's notes, not the expected answers, not another
arm's response or tool trace.
"""

from .base import Answer, AnswerClaim, Arm, ArmContext, IsolationError
from .pheasant_corpus import PheasantCorpusArm
from .pheasant_memory import MemorySeeder, PheasantMemoryArm
from .prior_control import PriorControlArm
from .specialist import SpecialistArm
from .tuned_replay import QueryTuner, TunedReplayArm

__all__ = [
    "Answer",
    "AnswerClaim",
    "Arm",
    "ArmContext",
    "IsolationError",
    "MemorySeeder",
    "PheasantCorpusArm",
    "PheasantMemoryArm",
    "PriorControlArm",
    "QueryTuner",
    "SpecialistArm",
    "TunedReplayArm",
    "build_arm",
]


def build_arm(arm_id: str, context: ArmContext) -> Arm:
    """Construct one arm by id."""

    factories = {
        "S0": SpecialistArm,
        "C0": PriorControlArm,
        "P0": PheasantCorpusArm,
        "P1": PheasantMemoryArm,
        "P2": TunedReplayArm,
    }
    try:
        factory = factories[arm_id]
    except KeyError as exc:
        raise ValueError(f"unknown arm '{arm_id}'; known: {sorted(factories)}") from exc
    return factory(context)
