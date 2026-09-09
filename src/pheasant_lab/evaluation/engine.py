"""The evaluation engine.

Runs the arms in isolation, turns what they did into typed proof, computes
every metric with its operands, pairs the arms, classifies each delta and
evaluates the gates - in that order, because a gate that runs after
aggregation is a gate a good score can hide.

Arm order is randomised from a recorded seed within each independent group.
The groups exist because two arms depend on what an earlier arm produced: P1
is measured with memory seeded from the first pass, and P2 with a query
strategy derived from it. Those dependencies are real and the ordering
therefore cannot be fully randomised; what *is* randomised is stated, and what
is not is stated too.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..arms import Arm, build_arm
from ..arms.base import ArmContext
from ..benchmark.freezer import FreezePackage
from ..benchmark.leakage import LeakageReport
from ..budget import CostLedger
from ..lifecycle import RunPaths, write_json
from ..settings import LabConfig
from . import answer_metrics, operational_metrics, retrieval_metrics
from .classification import Classification, apply_multiple_comparison, classify, non_inferior
from .evidence import ProofLedger
from .gates import GateSetResult, Verdict, evaluate_gates
from .metric import MetricResult
from .pairing import pair_arms

# The comparisons this lab is for, and what each one means.
COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("P0", "C0", "value of Pheasant corpus retrieval beyond the model prior"),
    ("P1", "P0", "attributable effect of memory and steering"),
    ("P2", "P1", "attributable effect of first-pass search tuning"),
    ("P1", "S0", "gap between Pheasant-only and the source-aware specialist"),
    ("P2", "S0", "post-tuning specialist gap"),
)

# Arms that can run against anything, and arms that need an earlier pass.
INDEPENDENT_GROUP = ("S0", "C0", "P0")
DEPENDENT_ORDER = ("P1", "P2")


@dataclass
class EvaluationResult:
    per_query: list[dict[str, Any]] = field(default_factory=list)
    aggregates: dict[str, Any] = field(default_factory=dict)
    classifications: list[Classification] = field(default_factory=list)
    gates: GateSetResult = field(default_factory=GateSetResult)
    proof: list[dict[str, Any]] = field(default_factory=list)
    answers: list[dict[str, Any]] = field(default_factory=list)
    arm_order: list[dict[str, Any]] = field(default_factory=list)
    non_inferiority: dict[str, Any] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)

    def metric(self, name: str, arm: str, cohort: str | None = None) -> dict[str, Any] | None:
        scope = self.aggregates.get("by_arm", {}).get(arm, {})
        if cohort:
            scope = self.aggregates.get("by_arm_cohort", {}).get(f"{arm}:{cohort}", {})
        return scope.get(name)

    def classification_for(
        self, metric: str, treatment: str, baseline: str, cohort: str | None = None
    ) -> Classification | None:
        for row in self.classifications:
            if (
                row.metric == metric
                and row.treatment_arm == treatment
                and row.baseline_arm == baseline
                and row.cohort == cohort
            ):
                return row
        return None


class EvaluationEngine:
    def __init__(
        self,
        config: LabConfig,
        *,
        run_id: str,
        paths: RunPaths,
        tracer: Any,
        ledger: CostLedger,
        package: FreezePackage,
        arm_contexts: Mapping[str, ArmContext],
        proof_ledger: ProofLedger,
        leakage: LeakageReport | None = None,
        memory_seeder: Any = None,
        query_tuner: Any = None,
        region_artifact_ids: Sequence[str] = (),
        snapshot_verification: Mapping[str, Any] | None = None,
        reconcile: Mapping[str, Any] | None = None,
        receipt_rate: tuple[int, int] = (0, 0),
    ) -> None:
        self.config = config
        self.run_id = run_id
        self.paths = paths
        self.tracer = tracer
        self.ledger = ledger
        self.package = package
        self.arm_contexts = dict(arm_contexts)
        self.proof = proof_ledger
        self.leakage = leakage
        self.memory_seeder = memory_seeder
        self.query_tuner = query_tuner
        self.region_artifact_ids = set(region_artifact_ids)
        self.snapshot_verification = dict(snapshot_verification or {})
        self.reconcile = dict(reconcile or {})
        self.receipt_rate = receipt_rate
        self.result = EvaluationResult()

    # -- the run -----------------------------------------------------------
    def run(self) -> EvaluationResult:
        questions = sorted(self.package.questions, key=lambda q: q.question_id)
        if self.leakage is not None and self.leakage.invalidated_questions:
            invalidated = self.leakage.invalidated_questions
            questions = [q for q in questions if q.question_id not in invalidated]
            self.result.limitations.append(
                f"{len(invalidated)} question(s) were excluded by an invalidating leakage finding"
            )

        answers: dict[str, list[dict[str, Any]]] = defaultdict(list)
        rng = random.Random(self.config.experiment.seed)

        with self.tracer.span("evaluation", stage="evaluation"):
            first_pass = self._run_group(INDEPENDENT_GROUP, questions, answers, rng)
            self._seed_and_tune(first_pass, questions)
            self._run_group(DEPENDENT_ORDER, questions, answers, rng, randomise=False)

            self._score(questions, answers)
            self._aggregate(answers)
            self._compare(questions)
            self._evaluate_gates(questions, answers)
            self._decide_non_inferiority()
            self._operational(answers)

        self.result.answers = [row for rows in answers.values() for row in rows]
        self.result.proof = self.proof.as_records()
        self._persist()
        return self.result

    def _run_group(
        self,
        group: Sequence[str],
        questions: Sequence[Any],
        answers: dict[str, list[dict[str, Any]]],
        rng: random.Random,
        *,
        randomise: bool = True,
    ) -> list[dict[str, Any]]:
        active = [
            arm_id for arm_id in group if arm_id in self.config.arms and arm_id in self.arm_contexts
        ]
        first_pass: list[dict[str, Any]] = []
        for repetition in range(1, self.config.replay.repetitions + 1):
            for question in questions:
                order = list(active)
                if randomise and self.config.replay.randomize_arm_order:
                    rng.shuffle(order)
                self.result.arm_order.append(
                    {
                        "question_id": question.question_id,
                        "repetition": repetition,
                        "order": list(order),
                    }
                )
                for arm_id in order:
                    arm = build_arm(arm_id, self.arm_contexts[arm_id])
                    record = self._answer(arm, question, repetition)
                    answers[arm_id].append(record)
                    if arm_id == "P0" and repetition == 1:
                        first_pass.append(
                            {
                                "question_id": question.question_id,
                                "question_text": question.text,
                                "answer_text": record.get("answer_text"),
                                "abstained": record.get("abstained"),
                                "queries_used": record.get("queries_used"),
                                "retrieved_artifact_ids": record.get("retrieved_artifact_ids"),
                                "search_calls": record.get("search_calls"),
                            }
                        )
        return first_pass

    def _answer(self, arm: Arm, question: Any, repetition: int) -> dict[str, Any]:
        try:
            answer = arm.answer(question, repetition=repetition)
            record = answer.as_record(store_text=self.config.privacy.store_response_text)
        except Exception as exc:
            self.tracer.errors.record(
                exc,
                stage="answer",
                component=f"arms.{arm.arm_id}",
                operation="answer",
                resolution="terminal",
            )
            record = {
                "answer_id": f"answer-failed-{arm.arm_id}-{question.question_id}-{repetition}",
                "run_id": self.run_id,
                "question_id": question.question_id,
                "arm_id": arm.arm_id,
                "repetition": repetition,
                "status": "failed",
                "error": str(exc),
                "claims": [],
                "search_calls": [],
            }
        self._record_proof(question, record)
        return record

    # -- proof -------------------------------------------------------------
    def _record_proof(self, question: Any, answer: Mapping[str, Any]) -> None:
        """Turn what happened into typed proof.

        ``served`` carries weight zero and is recorded anyway, because
        coverage - how much of what an arm saw the benchmark can speak to - is
        itself a number the report has to publish.
        """

        arm_id = str(answer.get("arm_id"))
        answer_id = str(answer.get("answer_id"))
        evidence = self.package.evidence_for(question)
        served = {
            str(result.get("artifact_id"))
            for call in answer.get("search_calls") or []
            for result in call.get("results") or []
            if result.get("artifact_id")
        }
        for artifact_id in sorted(served):
            self.proof.record(
                question_id=question.question_id,
                arm_id=arm_id,
                answer_id=answer_id,
                target_id=artifact_id,
                event_type="served",
                directness="structured_metadata",
                independence="self_reported",
                specificity="document_level",
            )
        for claim in answer.get("claims") or []:
            for citation in claim.get("citations") or []:
                self.proof.record(
                    question_id=question.question_id,
                    arm_id=arm_id,
                    answer_id=answer_id,
                    target_id=str(citation),
                    event_type="cited",
                    directness="direct_text",
                    # The arm citing its own retrieval is self-reported. It is
                    # worth something and it is not worth what an independent
                    # judgement is worth, and the multiplier says so.
                    independence="self_reported",
                    specificity="document_level",
                )

        # The strong evidence: a deterministic matcher over a frozen expected
        # fact. This is the only proof in the ledger nobody in the loop chose.
        facts = self.package.facts_for(question)
        text = str(answer.get("answer_text") or "")
        citations = [
            c for claim in answer.get("claims") or [] for c in (claim.get("citations") or [])
        ]
        for fact in facts:
            if not fact.eligible:
                continue
            matched = fact.matcher.matches(
                text,
                abstained=bool(answer.get("abstained")),
                citations=citations,
                tolerance=self.config.metrics.answer_matching.numeric_relative_tolerance,
            )
            target = fact.source_ids[0] if fact.source_ids else fact.fact_id
            self.proof.record(
                question_id=question.question_id,
                arm_id=arm_id,
                answer_id=answer_id,
                target_id=target,
                target_type="source" if fact.source_ids else "fact",
                event_type="deterministic_validation_pass"
                if matched
                else "deterministic_validation_fail",
                directness=fact.confidence_basis,
                independence="independent_judge",
                specificity="exact_locator" if fact.claim_ids else "document_level",
                detail={"fact_id": fact.fact_id},
            )
        for negative in evidence.known_negative_source_ids:
            if negative in served:
                self.proof.record(
                    question_id=question.question_id,
                    arm_id=arm_id,
                    answer_id=answer_id,
                    target_id=negative,
                    target_type="source",
                    event_type="explicit_reject",
                    directness="structured_metadata",
                    independence="independent_judge",
                    specificity="source_level",
                    detail={
                        "reason": "benchmark marked this source a known negative for this question"
                    },
                )

    # -- treatments --------------------------------------------------------
    def _seed_and_tune(
        self, first_pass: Sequence[Mapping[str, Any]], questions: Sequence[Any]
    ) -> None:
        learned = [q.question_id for q in questions if "learned" in q.cohorts]
        forbidden = [
            q.question_id
            for q in questions
            if "temporal_holdout" in q.cohorts or "control" in q.cohorts
        ]
        if self.memory_seeder is not None and "P1" in self.config.arms:
            records = self.memory_seeder.seed(
                first_pass, learned_question_ids=learned, forbidden_question_ids=forbidden
            )
            self.result.limitations.append(
                f"P1 was measured with {len(records)} memory record(s) seeded from the learned cohort's "
                "first pass only"
            )
        if self.query_tuner is not None and "P2" in self.config.arms:
            strategy = self.query_tuner.tune(first_pass, learned_question_ids=learned)
            context = self.arm_contexts.get("P2")
            if context is not None:
                context.query_strategy = strategy
            self.tracer.emit("tuning.strategy", payload=strategy)
            self.result.limitations.append(
                f"P2 used {len(strategy.get('expansions', {}))} expansion rule(s) derived from the "
                "learned cohort's first-pass retrieval behaviour"
            )

    # -- scoring -----------------------------------------------------------
    def _score(self, questions: Sequence[Any], answers: Mapping[str, list[dict[str, Any]]]) -> None:
        k = self.config.metrics.metrics.k
        by_id = {question.question_id: question for question in questions}
        for arm_id, rows in sorted(answers.items()):
            for answer in rows:
                question = by_id.get(str(answer.get("question_id")))
                if question is None:
                    continue
                cohort = (question.cohorts or [None])[0]
                evidence = self.package.evidence_for(question)
                results: list[MetricResult] = []
                if arm_id in {"P0", "P1", "P2"}:
                    results.extend(
                        retrieval_metrics.compute(
                            question, evidence, answer, k=k, run_id=self.run_id, cohort=cohort
                        )
                    )
                results.extend(
                    answer_metrics.compute(
                        question,
                        self.package.facts_for(question),
                        evidence,
                        answer,
                        self.config.metrics.answer_matching,
                        run_id=self.run_id,
                        cohort=cohort,
                    )
                )
                for result in results:
                    row = result.as_dict()
                    row["answer_id"] = answer.get("answer_id")
                    row["repetition"] = answer.get("repetition")
                    row["question_type"] = question.type
                    row["difficulty"] = question.difficulty
                    self.result.per_query.append(row)

    def _aggregate(self, answers: Mapping[str, list[dict[str, Any]]]) -> None:
        by_arm: dict[str, dict[str, Any]] = defaultdict(dict)
        by_arm_cohort: dict[str, dict[str, Any]] = defaultdict(dict)
        buckets: dict[tuple[str, str, str | None], list[float]] = defaultdict(list)
        excluded: dict[tuple[str, str], int] = defaultdict(int)

        for row in self.result.per_query:
            key_metric = str(row["metric"])
            arm_id = str(row.get("arm_id"))
            cohort = row.get("cohort")
            if row.get("status") != "ok" or row.get("value") is None:
                excluded[(arm_id, key_metric)] += 1
                continue
            buckets[(arm_id, key_metric, None)].append(float(row["value"]))
            if cohort:
                buckets[(arm_id, key_metric, str(cohort))].append(float(row["value"]))

        minimum = self.config.proof.proof.minimum_evidence.per_metric_questions
        for (arm_id, metric, cohort), values in sorted(
            buckets.items(), key=lambda item: (item[0][0], item[0][1], item[0][2] or "")
        ):
            payload = {
                "metric": metric,
                "arm_id": arm_id,
                "cohort": cohort,
                "n": len(values),
                "mean": sum(values) / len(values),
                "excluded": excluded.get((arm_id, metric), 0),
                "status": "ok"
                if len(values) >= (minimum if cohort is None else 1)
                else "insufficient_evidence",
            }
            if payload["status"] != "ok":
                payload["mean"] = None
                payload["limitation"] = (
                    f"{len(values)} scored questions is below the configured minimum of {minimum}"
                )
            if cohort is None:
                by_arm[arm_id][metric] = payload
            else:
                by_arm_cohort[f"{arm_id}:{cohort}"][metric] = payload

        self.result.aggregates = {
            "by_arm": {arm: dict(metrics) for arm, metrics in sorted(by_arm.items())},
            "by_arm_cohort": {key: dict(metrics) for key, metrics in sorted(by_arm_cohort.items())},
            "answers_by_arm": {arm: len(rows) for arm, rows in sorted(answers.items())},
            "proof": self.proof.summarise().as_dict(),
        }

    # -- comparisons -------------------------------------------------------
    def _compare(self, questions: Sequence[Any]) -> None:
        question_map = {q.question_id: q.as_dict() for q in questions}
        metrics = list(self.config.metrics.metrics.primary)
        cohorts: list[str | None] = [None, "learned", "temporal_holdout", "control", "anchor"]

        for treatment, baseline, meaning in COMPARISONS:
            if treatment not in self.config.arms or baseline not in self.config.arms:
                continue
            for metric in metrics:
                for cohort in cohorts:
                    restrict = (
                        [q.question_id for q in questions if cohort in (q.cohorts or [])]
                        if cohort
                        else None
                    )
                    if restrict is not None and not restrict:
                        continue
                    pairing = pair_arms(
                        self.result.per_query,
                        metric=metric,
                        baseline_arm=baseline,
                        treatment_arm=treatment,
                        questions=question_map,
                        policy=self.config.replay.pairing_policy,
                        restrict_to=restrict,
                    )
                    if pairing.n == 0:
                        continue
                    result = classify(
                        pairing,
                        classification=self.config.metrics.classification,
                        statistics_config=self.config.metrics.statistics,
                        relative_lift_epsilon=self.config.metrics.metrics.relative_lift_epsilon,
                        cohort=cohort,
                    )
                    result.reasons.append(f"comparison meaning: {meaning}")
                    self.result.classifications.append(result)

        survivors = apply_multiple_comparison(
            self.result.classifications, statistics_config=self.config.metrics.statistics
        )
        if survivors:
            self.result.aggregates["multiple_comparison"] = {
                "correction": self.config.metrics.statistics.multiple_comparison_correction,
                "false_discovery_rate": self.config.metrics.statistics.false_discovery_rate,
                "survives": survivors,
            }

    # -- gates -------------------------------------------------------------
    def _evaluate_gates(
        self, questions: Sequence[Any], answers: Mapping[str, list[dict[str, Any]]]
    ) -> None:
        observations: dict[str, tuple[float | int | None, str]] = {}

        receipts, submitted = self.receipt_rate
        observations["ingest_receipt_rate"] = (
            (receipts / submitted) if submitted else None,
            f"{receipts}/{submitted} submitted sources carry a receipt",
        )
        observations["silent_loss"] = (
            self.reconcile.get("silent_loss"),
            f"reconcile reported {self.reconcile.get('silent_loss')} receipts claiming an artifact "
            "the region does not hold",
        )
        observations["benchmark_leakage"] = (
            len([f for f in self.leakage.findings if f.invalidating]) if self.leakage else None,
            "invalidating leakage findings against the frozen question set",
        )
        # Only *unexpected* drift is a gate failure. The memory section moves
        # because writing memory is what P1 is, and counting that as drift
        # would fail every run that measured the thing it was built for.
        drift = self.snapshot_verification.get("unexpected_drift")
        expected = self.snapshot_verification.get("expected_drift") or []
        observations["snapshot_drift"] = (
            len(drift) if drift is not None else None,
            f"sections that moved and should not have: {drift}; expected movement: {expected}",
        )
        observations["budget"] = (
            max(0.0, self.ledger.committed_usd - self.ledger.total_budget_usd),
            f"committed {self.ledger.committed_usd:.4f} against a budget of "
            f"{self.ledger.total_budget_usd:.4f}",
        )

        # A known positive the benchmark names that the region does not hold
        # is a benchmark-integrity violation, not a retrieval miss.
        if self.region_artifact_ids:
            missing = 0
            for question in questions:
                evidence = self.package.evidence_for(question)
                for artifact in evidence.acceptable_artifact_ids:
                    if artifact not in self.region_artifact_ids:
                        missing += 1
            observations["known_positive_exclusion"] = (
                missing,
                "expected artifacts named by the benchmark that the region does not hold",
            )
        else:
            observations["known_positive_exclusion"] = (
                None,
                "the region's holdings were not enumerated, so this could not be checked",
            )

        abstention = [
            row["value"]
            for row in self.result.per_query
            if row["metric"] == "abstention_accuracy"
            and row.get("arm_id") in {"P1", "P2"}
            and row.get("value") is not None
        ]
        observations["abstention"] = (
            (sum(abstention) / len(abstention)) if abstention else None,
            f"mean abstention accuracy over {len(abstention)} scored answers in the Pheasant arms",
        )

        control = self._control_regression()
        observations["control_regression"] = control
        observations["negative_exposure_increase"] = self._negative_exposure_delta()

        # ACL and stale-memory leakage cannot be checked in a region with no
        # ACL enforcement and no superseded records. Skipped, not passed: an
        # unchecked box and a failed one are equally disqualifying.
        observations["acl_leak"] = (
            None,
            "no principal-scoped ACL is configured for this run, so no leak could be observed",
        )
        superseded = getattr(self.memory_seeder, "written", []) if self.memory_seeder else []
        observations["stale_memory_leak"] = (
            None if not any(record.get("supersedes") for record in superseded) else 0.0,
            "no superseded memory record exists in this region, so this gate had nothing to test"
            if not any(record.get("supersedes") for record in superseded)
            else "no superseded record was returned by any arm",
        )
        observations["as_of_correctness"] = self._as_of_violations(questions, answers)

        self.result.gates = evaluate_gates(self.config.metrics.gates, observations)

    def _control_regression(self) -> tuple[float | None, str]:
        rows = [
            row
            for row in self.result.per_query
            if row["metric"] == "fact_f1"
            and row.get("cohort") == "control"
            and row.get("value") is not None
        ]
        by_question: dict[str, dict[str, float]] = defaultdict(dict)
        for row in rows:
            by_question[str(row["question_id"])][str(row["arm_id"])] = float(row["value"])
        pairs = [
            (values["P0"], values["P1"])
            for values in by_question.values()
            if "P0" in values and "P1" in values
        ]
        if not pairs:
            return (None, "no control-cohort question was scored in both P0 and P1")
        regressed = sum(1 for baseline, treatment in pairs if treatment < baseline)
        return (
            regressed / len(pairs),
            f"{regressed}/{len(pairs)} control-cohort questions scored worse with memory on",
        )

    def _negative_exposure_delta(self) -> tuple[float | None, str]:
        def mean(arm: str) -> float | None:
            values = [
                float(row["value"])
                for row in self.result.per_query
                if row["metric"] == "negative_exposure_at_k"
                and row.get("arm_id") == arm
                and row.get("value") is not None
            ]
            return sum(values) / len(values) if values else None

        base, treat = mean("P0"), mean("P1")
        if base is None or treat is None:
            return (None, "negative exposure was not scored in both P0 and P1")
        return (treat - base, f"negative exposure moved {treat - base:+.4f} from P0 to P1")

    def _as_of_violations(
        self, questions: Sequence[Any], answers: Mapping[str, list[dict[str, Any]]]
    ) -> tuple[int | None, str]:
        temporal = [q for q in questions if q.type == "temporal"]
        if not temporal:
            return (
                None,
                "the question set contains no temporal case, so as_of could not be checked",
            )
        by_id = {q.question_id: q for q in temporal}
        violations = 0
        for rows in answers.values():
            for answer in rows:
                question = by_id.get(str(answer.get("question_id")))
                if question is None:
                    continue
                cited = {
                    str(c)
                    for claim in answer.get("claims") or []
                    for c in (claim.get("citations") or [])
                }
                if cited & set(question.known_negative_source_ids):
                    violations += 1
        return (violations, f"{violations} answers to temporal questions cited post-as_of material")

    # -- the decision ------------------------------------------------------
    def _decide_non_inferiority(self) -> None:
        config = self.config.metrics.specialist_noninferiority
        decisions: dict[str, Any] = {}
        gates_pass = self.result.gates.verdict is Verdict.passed

        for arm in ("P1", "P2"):
            if arm not in self.config.arms or "S0" not in self.config.arms:
                continue
            per_metric: dict[str, Any] = {}
            verdict = True
            reasons: list[str] = []
            for metric, margin in sorted(config.margin.items()):
                overall = self.result.classification_for(metric, arm, "S0")
                holdout = self.result.classification_for(
                    metric, arm, "S0", cohort="temporal_holdout"
                )
                if overall is None:
                    per_metric[metric] = {
                        "result": None,
                        "reason": "no paired comparison was produced",
                    }
                    verdict = False
                    reasons.append(f"{metric}: no comparison")
                    continue
                answer, why = non_inferior(overall, margin=margin)
                holdout_answer, holdout_why = (
                    non_inferior(holdout, margin=margin)
                    if holdout is not None
                    else (None, "no holdout comparison")
                )
                per_metric[metric] = {
                    "overall": {"non_inferior": answer, "reason": why},
                    "holdout": {"non_inferior": holdout_answer, "reason": holdout_why},
                    "margin": margin,
                }
                if answer is not True:
                    verdict = False
                    reasons.append(f"{metric}: {why}")
                if config.require_holdout_cohort and holdout_answer is not True:
                    verdict = False
                    reasons.append(f"{metric} on the temporal holdout: {holdout_why}")

            if config.require_gates_pass and not gates_pass:
                verdict = False
                reasons.append(f"hard gates did not pass ({self.result.gates.verdict.value})")
            if (
                config.require_cost_within_budget
                and self.ledger.committed_usd > self.ledger.total_budget_usd
            ):
                verdict = False
                reasons.append("the run exceeded its cost budget")
            if config.require_no_protected_subgroup_regression:
                regressed = [
                    f"{c.metric}:{s.key}={s.value}"
                    for c in self.result.classifications
                    if c.treatment_arm == arm and c.baseline_arm == "S0"
                    for s in c.subgroups
                    if s.status == "regressed"
                ]
                if regressed:
                    verdict = False
                    reasons.append(f"protected subgroups regressed: {sorted(set(regressed))}")

            decisions[arm] = {
                "rivals_specialist": verdict if per_metric else None,
                "status": "pass"
                if verdict and per_metric
                else ("fail" if per_metric else "insufficient_evidence"),
                "metrics": per_metric,
                "reasons": reasons,
            }
        self.result.non_inferiority = decisions

    def _operational(self, answers: Mapping[str, list[dict[str, Any]]]) -> None:
        flat = [row for rows in answers.values() for row in rows]
        proven = sum(
            1 for event in self.proof.events if event.event_type == "deterministic_validation_pass"
        )
        results = operational_metrics.compute(
            run_id=self.run_id,
            answers=flat,
            cost_events=[event.as_payload() for event in self.ledger.events],
            proven_answers=proven,
            verified_sources=self.receipt_rate[0],
            questions=len(self.package.questions),
            budget=self.ledger.snapshot(),
        )
        self.result.aggregates["operational"] = {row.metric: row.as_dict() for row in results}

    # -- persistence -------------------------------------------------------
    def _persist(self) -> None:
        metrics_dir = self.paths.metrics
        metrics_dir.mkdir(parents=True, exist_ok=True)

        with (metrics_dir / "per-query.jsonl").open("w", encoding="utf-8") as handle:
            for row in self.result.per_query:
                handle.write(_json_line(row))

        write_json(metrics_dir / "aggregates.json", self.result.aggregates)
        write_json(
            metrics_dir / "classification.json",
            {
                "comparisons": [row.as_dict() for row in self.result.classifications],
                "non_inferiority": self.result.non_inferiority,
                "arm_order_seed": self.config.experiment.seed,
                "limitations": self.result.limitations,
            },
        )
        write_json(metrics_dir / "gates.json", self.result.gates.as_dict())

        with (metrics_dir / "paired-deltas.jsonl").open("w", encoding="utf-8") as handle:
            for row in self.result.classifications:
                handle.write(_json_line(row.as_dict()))

        header = (
            "metric,baseline_arm,treatment_arm,cohort,n,baseline_mean,treatment_mean,"
            "absolute_delta,relative_delta,practical_threshold,status\n"
        )
        with (metrics_dir / "paired-deltas.csv").open("w", encoding="utf-8") as handle:
            handle.write(header)
            for row in self.result.classifications:
                handle.write(
                    ",".join(
                        [
                            row.metric,
                            row.baseline_arm,
                            row.treatment_arm,
                            row.cohort or "all",
                            str(row.n),
                            _csv(row.baseline_mean),
                            _csv(row.treatment_mean),
                            _csv(row.absolute_delta),
                            _csv(row.relative_delta),
                            _csv(row.threshold),
                            row.status.value,
                        ]
                    )
                    + "\n"
                )

        for event in self.proof.events:
            self.tracer.append("proof-events.jsonl", event.as_dict())


def _json_line(row: Mapping[str, Any]) -> str:
    import json

    return json.dumps(row, sort_keys=True, ensure_ascii=False, default=str) + "\n"


def _csv(value: float | None) -> str:
    return "" if value is None else f"{value:.6g}"
