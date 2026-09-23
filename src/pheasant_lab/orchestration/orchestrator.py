"""The collection state machine.

Rounds of bounded, concurrent research branches, an audit after each, and a
stop decision that can only say ``sufficient`` when every hard condition
passes. The orchestrator owns three things nothing else may decide: how much
budget a round may commit, whether the evaluation reserve is still intact, and
what the run is allowed to call the result.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from ..budget import BudgetExceeded, CostLedger
from ..models import ModelProvider
from ..pheasant.ingestion import Ingestor
from ..providers.base import LiteratureProvider
from ..settings import LabConfig, Topic
from .auditor import CoverageAudit, CoverageAuditor
from .planner import Planner, Subtopic
from .researcher import BranchResult, Researcher
from .state import CollectionState, RoundRecord
from .stopping import StopDecision, StoppingCalculus


@dataclass
class CollectionResult:
    topic_id: str
    state: CollectionState
    decision: StopDecision
    audit: CoverageAudit
    rounds: list[dict[str, Any]] = field(default_factory=list)
    branches: list[dict[str, Any]] = field(default_factory=list)
    subtopics: list[dict[str, Any]] = field(default_factory=list)

    @property
    def sufficient(self) -> bool:
        return self.decision.sufficient

    def as_dict(self) -> dict[str, Any]:
        return {
            "topic_id": self.topic_id,
            "summary": self.state.summary(),
            "decision": self.decision.as_dict(),
            "audit": self.audit.as_dict(),
            "rounds": list(self.rounds),
            "branches": list(self.branches),
            "subtopics": list(self.subtopics),
        }


class Orchestrator:
    def __init__(
        self,
        config: LabConfig,
        *,
        run_id: str,
        tracer: Any,
        ledger: CostLedger,
        providers: list[LiteratureProvider],
        planner_model: ModelProvider,
        researcher_model: ModelProvider,
        auditor_model: ModelProvider | None,
        ingestor: Ingestor | None,
        started_at: float | None = None,
    ) -> None:
        self.config = config
        self.run_id = run_id
        self.tracer = tracer
        self.ledger = ledger
        self.providers = providers
        self.planner_model = planner_model
        self.researcher_model = researcher_model
        self.auditor_model = auditor_model
        self.ingestor = ingestor
        self.started_at = started_at if started_at is not None else time.monotonic()

    # -- the loop ----------------------------------------------------------
    def collect(self, topic: Topic) -> CollectionResult:
        state = CollectionState(self.run_id, topic.id)
        planner = Planner(self.config, self.planner_model, self.ledger, tracer=self.tracer)
        auditor = CoverageAuditor(
            self.config, topic, model=self.auditor_model, ledger=self.ledger, tracer=self.tracer
        )
        calculus = StoppingCalculus(
            self.config.stopping,
            topic,
            authoritative_types=self.config.collection.authoritative_source_types,
        )

        branches: list[dict[str, Any]] = []
        subtopics: list[Subtopic] = []
        audit = auditor.audit(state)
        decision = StopDecision(outcome="continue", reason="collection has not started")
        ordinal = 0

        with self.tracer.span("collection", topic_id=topic.id, stage="discovery"):
            subtopics = planner.plan(
                topic, max_subtopics=self.config.collection.max_research_agents, run_id=self.run_id
            )
            for round_number in range(1, self.config.collection.max_depth + 1):
                if not subtopics:
                    break
                before = len(state.eligible_claims())
                acquired_before = len(state.sources)
                cost_before = self.ledger.committed_usd

                results, ordinal = self._run_round(topic, subtopics, state, round_number, ordinal)
                branches.extend(result.as_dict() for result in results)

                duplicates_after, _ = state.duplicate_rate()
                state.rounds.append(
                    RoundRecord(
                        number=round_number,
                        subtopics=[s.subtopic_id for s in subtopics],
                        new_eligible_claims=len(state.eligible_claims()) - before,
                        eligible_claims_before=before,
                        acquired=len(state.sources) - acquired_before,
                        duplicates=duplicates_after,
                        cost_usd=self.ledger.committed_usd - cost_before,
                    )
                )

                assigned = {facet for subtopic in subtopics for facet in subtopic.facet_ids}
                audit = auditor.audit(state, assigned_facets=assigned)
                decision = self._decide(
                    calculus,
                    state,
                    audit,
                    round_number=round_number,
                    round_limit=self.config.collection.max_depth,
                )
                self.tracer.emit(
                    "collection.round",
                    payload={
                        "round": round_number,
                        **state.rounds[-1].as_dict(),
                        "decision": decision.outcome,
                        "unmet": decision.unmet,
                        "budget": self.ledger.snapshot(),
                    },
                    topic_id=topic.id,
                )
                if decision.outcome != "continue":
                    break

                widened = planner.replan(
                    topic,
                    gaps=[gap.facet_id for gap in audit.quality_gaps]
                    + audit.unassigned_critical_gaps,
                    existing=subtopics,
                    run_id=self.run_id,
                )
                subtopics = widened or subtopics

        result = CollectionResult(
            topic_id=topic.id,
            state=state,
            decision=decision,
            audit=audit,
            rounds=[r.as_dict() for r in state.rounds],
            branches=branches,
            subtopics=[s.as_dict() for s in subtopics],
        )
        self.tracer.emit(
            "collection.finished",
            status="succeeded" if result.sufficient else "partial",
            payload=result.as_dict(),
            topic_id=topic.id,
        )
        return result

    # -- one round ---------------------------------------------------------
    def _run_round(
        self,
        topic: Topic,
        subtopics: list[Subtopic],
        state: CollectionState,
        round_number: int,
        ordinal: int,
    ) -> tuple[list[BranchResult], int]:
        researcher = Researcher(
            self.config,
            topic=topic,
            providers=self.providers,
            model=self.researcher_model,
            ledger=self.ledger,
            ingestor=self.ingestor,
            tracer=self.tracer,
            run_id=self.run_id,
        )
        assignments: list[tuple[Subtopic, int]] = []
        for subtopic in subtopics[: self.config.collection.max_research_agents]:
            ordinal += 1
            assignments.append((subtopic, ordinal))

        results = {
            subtopic.subtopic_id: BranchResult(
                subtopic_id=subtopic.subtopic_id, agent_id=researcher.agent_for(branch_ordinal)
            )
            for subtopic, branch_ordinal in assignments
        }
        workers = max(1, min(self.config.collection.max_concurrent_agents, len(assignments)))
        max_sources = self.config.collection.max_sources_per_subtopic

        # Phase 1 - discovery, concurrent. Each branch talks to the providers
        # and touches nothing shared.
        candidates: dict[str, list[Any]] = {}
        self._parallel(
            workers,
            [
                (
                    subtopic,
                    lambda subtopic=subtopic: candidates.__setitem__(
                        subtopic.subtopic_id,
                        researcher.search_candidates(
                            subtopic,
                            results[subtopic.subtopic_id],
                            max_rounds=self.config.collection.max_search_rounds_per_agent,
                            max_sources=max_sources,
                        ),
                    ),
                )
                for subtopic, _ordinal in assignments
            ],
            stage="discovery",
            results=results,
        )

        # Phase 2 - admission: sequential, in a fixed order, and **round
        # robin** across subtopics. This is the only phase that reads and
        # writes the shared registry, so the order decides which subtopic owns
        # a candidate two branches both found. Draining one subtopic before
        # starting the next is deterministic and unfair: a broad subtopic
        # claims everything a narrow one also needs, and the narrow facet
        # reports as uncovered when the corpus in fact covers it.
        ordered = sorted(assignments, key=lambda pair: pair[0].subtopic_id)
        fresh: dict[str, list[Any]] = {subtopic.subtopic_id: [] for subtopic, _ in ordered}
        depth = max((len(candidates.get(s.subtopic_id, [])) for s, _ in ordered), default=0)
        for index in range(depth):
            for subtopic, _ordinal in ordered:
                pool = candidates.get(subtopic.subtopic_id, [])
                if index >= len(pool) or len(fresh[subtopic.subtopic_id]) >= max_sources:
                    continue
                record = researcher.admit_one(
                    subtopic,
                    pool[index],
                    state,
                    results[subtopic.subtopic_id],
                    round_=round_number,
                )
                if record is not None:
                    fresh[subtopic.subtopic_id].append(record)

        # Phase 3 - extraction and ingestion, concurrent again: by now each
        # branch owns its own records.
        self._parallel(
            workers,
            [
                (
                    subtopic,
                    lambda subtopic=subtopic: researcher.extract_and_persist(
                        subtopic,
                        state,
                        results[subtopic.subtopic_id],
                        fresh.get(subtopic.subtopic_id, []),
                        round_=round_number,
                    ),
                )
                for subtopic, _ordinal in assignments
            ],
            stage="extraction",
            results=results,
        )
        return sorted(results.values(), key=lambda r: r.subtopic_id), ordinal

    def _parallel(
        self,
        workers: int,
        work: list[tuple[Subtopic, Any]],
        *,
        stage: str,
        results: dict[str, BranchResult],
    ) -> None:
        """Run one phase across branches, recording whatever fails.

        A branch that raises is recorded and the round continues: one failed
        subtopic is a coverage gap the audit will report, not a reason to
        throw away the other branches' work.
        """

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="branch") as pool:
            futures = {pool.submit(action): subtopic for subtopic, action in work}
            for future in as_completed(futures):
                subtopic = futures[future]
                try:
                    future.result()
                except BudgetExceeded as exc:
                    self.tracer.errors.record(
                        exc,
                        stage=stage,
                        component="orchestration.orchestrator",
                        operation=f"branch:{subtopic.subtopic_id}",
                        resolution="terminal",
                        error_class="budget",
                        retryable=False,
                    )
                    results[subtopic.subtopic_id].errors += 1
                except Exception as exc:
                    self.tracer.errors.record(
                        exc,
                        stage=stage,
                        component="orchestration.orchestrator",
                        operation=f"branch:{subtopic.subtopic_id}",
                        resolution="terminal",
                    )
                    results[subtopic.subtopic_id].errors += 1

    # -- the decision ------------------------------------------------------
    def _decide(
        self,
        calculus: StoppingCalculus,
        state: CollectionState,
        audit: CoverageAudit,
        *,
        round_number: int,
        round_limit: int,
    ) -> StopDecision:
        reserve = self.config.stopping.evaluation_budget_reserve_fraction
        intact = self.ledger.evaluation_reserve_intact(reserve)
        receipts = (
            self.ingestor.ledger.receipt_rate()
            if self.ingestor is not None
            else (len(state.retained_sources()), len(state.retained_sources()))
        )
        elapsed_minutes = (time.monotonic() - self.started_at) / 60.0
        return calculus.evaluate(
            state,
            receipt_rate=receipts,
            evaluation_reserve_intact=intact,
            unassigned_critical_gaps=audit.unassigned_critical_gaps,
            budget_exhausted=not intact,
            time_exhausted=elapsed_minutes >= self.config.experiment.runtime_budget_minutes,
            round_limit_reached=round_number >= round_limit,
        )
