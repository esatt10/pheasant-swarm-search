"""The research swarm: plan, branch, extract, audit, decide."""

from .auditor import CoverageAudit, CoverageAuditor
from .orchestrator import CollectionResult, Orchestrator
from .planner import Planner, Subtopic
from .researcher import BranchResult, Researcher
from .state import ClaimRecord, CollectionState, ContradictionRecord, SourceRecord, SourceState
from .stopping import StopDecision, StoppingCalculus

__all__ = [
    "BranchResult",
    "ClaimRecord",
    "CollectionResult",
    "CollectionState",
    "ContradictionRecord",
    "CoverageAudit",
    "CoverageAuditor",
    "Orchestrator",
    "Planner",
    "Researcher",
    "SourceRecord",
    "SourceState",
    "StopDecision",
    "StoppingCalculus",
    "Subtopic",
]
