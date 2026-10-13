"""``P1`` - the same snapshot, with memory and steering on.

The paired difference ``P1 - P0`` is the memory-attributable effect, and it is
only that if two things hold:

* the **corpus is identical**. P1 runs against the snapshot P0 ran against;
  anything that changed the corpus between them would be attributed to memory.
* the memory was written from the **learned** cohort only. A record derived
  from a holdout question makes ``learned - holdout`` a restatement instead of
  a memorisation detector, and the leakage checker refuses that run.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from ..pheasant.capabilities import CapabilityMap
from ..pheasant.client import PheasantClient
from ..pheasant.ingestion import read_index_queue
from ..pheasant.retrieval import MemoryOptions
from ..settings import AnswerMatchingSection, PheasantFile
from ..textkit import content_terms, tokens
from . import memory_proof
from .memory_proof import Proof
from .pheasant_corpus import PheasantCorpusArm


class PheasantMemoryArm(PheasantCorpusArm):
    arm_id = "P1"
    memory_enabled = True

    def memory_options(self) -> MemoryOptions:
        return MemoryOptions(
            enabled=True,
            current_only=True,
            # Steering records change ranking; they are not passages. Asking
            # for them in the result list would hand an answerer a line of
            # rule syntax dressed as retrieved knowledge.
            include_rules=False,
            as_of=None,
        )


#: pheasant's memory source. A fleet publishes its sync like any other.
MEMORY_SOURCE = "agent-memory"


class MemorySeeder:
    """Writes the memory ``P1`` is measured with.

    Each learned question yields at most one **retrieval proof** - the
    question, the claims its answer made that the session's own retrieval
    validates, and the passages that validate them, quoted and linked (see
    :mod:`.memory_proof`) - plus the steering rules that proof supports. A
    question whose answer is validated by fewer than ``min_evidence`` distinct
    passages writes nothing at all, and the trace says why: an unproven answer
    is not something to remember, and a rule learned from one has no evidence
    behind it either.

    Every record names the question that produced it. That field is what the
    leakage checker reads to prove no holdout or control question created the
    treatment it is being tested against - so it is written even though
    nothing in the region requires it.

    On a role-split region a memory write answers ``sync: {status: queued}``
    (pheasant >= 0.13.2): the record is stored and is not yet searchable.
    ``P1`` starting then would be measured partly *without* the treatment it
    is named for, so :meth:`seed` waits for the queued tasks to leave the
    region's index queue and records how that went (``memory.indexed``).
    """

    def __init__(
        self,
        client: PheasantClient,
        capabilities: CapabilityMap,
        config: PheasantFile,
        *,
        tracer: Any = None,
        principal: str = "pheasant-swarm-lab",
        clock: Any = time,
        matching: AnswerMatchingSection | None = None,
        min_evidence: int = 1,
    ) -> None:
        if min_evidence < 1:
            raise ValueError("a memory needs at least one validated passage behind it")
        self.client = client
        #: The support rule `evidence_support_rate` scores with; a proof is
        #: held to the same one.
        self.matching = matching or AnswerMatchingSection()
        #: ``k``: distinct validated passages an answer needs to be remembered.
        self.min_evidence = min_evidence
        #: Why each learned question wrote nothing, keyed by question id.
        self.skipped: dict[str, str] = {}
        self.capabilities = capabilities
        self.config = config
        self.tracer = tracer
        self.principal = principal
        self._kb_field = str(config.argument_map.get("knowledge_base_field", "knowledge_base"))
        self.written: list[dict[str, Any]] = []
        self._clock = clock
        #: Task ids of memory syncs the region published rather than ran.
        self.queued_tasks: set[str] = set()
        #: How the wait for those tasks ended; ``None`` when nothing was queued.
        self.index_outcome: dict[str, Any] | None = None

    @property
    def available(self) -> bool:
        return self.capabilities.has("write_memory")

    def seed(
        self,
        first_pass: Sequence[Mapping[str, Any]],
        *,
        learned_question_ids: Iterable[str],
        forbidden_question_ids: Iterable[str] = (),
    ) -> list[dict[str, Any]]:
        """Derive records from the first pass over the learned cohort.

        Two rules, both enforced here rather than downstream:

        * only ``learned`` questions may produce a record;
        * a record is derived from what the *arm* did - the queries it ran and
          what came back - never from the expected answers, which this class
          is never given.
        """

        if not self.available:
            return []
        learned = set(learned_question_ids)
        forbidden = set(forbidden_question_ids)
        records: list[dict[str, Any]] = []

        proven = 0
        for entry in first_pass:
            question_id = str(entry.get("question_id") or "")
            if question_id not in learned or question_id in forbidden:
                continue
            candidates = self._candidates(entry)
            if not candidates:
                continue
            proven += 1
            for record in candidates:
                written = self._write(record, question_id)
                if written is not None:
                    records.append(written)
        self.written.extend(records)
        if self.tracer is not None:
            reasons: dict[str, int] = {}
            for reason in self.skipped.values():
                reasons[reason] = reasons.get(reason, 0) + 1
            self.tracer.emit(
                "memory.seeded",
                payload={
                    "records": len(records),
                    "learned_questions": len(learned),
                    "proven_questions": proven,
                    "min_evidence": self.min_evidence,
                    "kinds": {
                        kind: sum(1 for r in records if r["kind"] == kind)
                        for kind in ("fact", "alias", "preference")
                    },
                    "skipped": reasons,
                },
            )
        if self.queued_tasks:
            self.index_outcome = self.wait_until_indexed()
        return records

    def wait_until_indexed(self, wait_seconds: float | None = None) -> dict[str, Any]:
        """Wait for every queued memory sync to leave the index queue.

        A task is done when the region's complete listing no longer holds it.
        A ``dead`` task never will, so it ends the wait as ``failed``. Without
        the ``index_queue`` capability, or with a listing the backend cannot
        give, completion is unknowable: the outcome says so rather than
        guessing, and the engine turns it into a limitation.
        """

        pending = set(self.queued_tasks)
        deadline = self._clock.monotonic() + (
            self.config.timeout_seconds if wait_seconds is None else wait_seconds
        )
        backoff = max(0.05, self.config.retry_backoff_seconds)
        started = self._clock.monotonic()
        polls = 0
        outcome = "unknown"
        dead: list[str] = []
        while True:
            polls += 1
            tasks = read_index_queue(
                self.client,
                self.capabilities,
                {self._kb_field: self.config.knowledge_base},
                source=MEMORY_SOURCE,
            )
            if tasks is None:
                outcome = "unknown"
                break
            listed = {str(task.get("task_id")): task for task in tasks}
            dead = sorted(t for t in pending if (listed.get(t) or {}).get("state") == "dead")
            if dead:
                outcome = "failed"
                break
            pending = {t for t in pending if t in listed}
            if not pending:
                outcome = "indexed"
                break
            if self._clock.monotonic() >= deadline:
                outcome = "timed_out"
                break
            self._clock.sleep(min(backoff, self.config.retry_backoff_max_seconds))
            backoff *= 2
        result = {
            "outcome": outcome,
            "tasks": sorted(self.queued_tasks),
            "outstanding": sorted(pending) if outcome != "indexed" else [],
            "dead": dead,
            "polls": polls,
            "waited_seconds": round(self._clock.monotonic() - started, 3),
        }
        if self.tracer is not None:
            self.tracer.emit(
                "memory.indexed",
                status={"indexed": "succeeded", "failed": "failed"}.get(outcome, "partial"),
                payload=result,
            )
        return result

    @property
    def limitation(self) -> str | None:
        """What ``P1``'s numbers cannot claim, when the wait did not confirm."""

        if self.index_outcome is None or self.index_outcome["outcome"] == "indexed":
            return None
        reason = {
            "unknown": "the region offers no complete index-queue listing to confirm it",
            "timed_out": f"the wait ended after {self.index_outcome['waited_seconds']}s",
            "failed": "the region dead-lettered the task",
        }[self.index_outcome["outcome"]]
        return (
            f"P1's memory syncs were published to the region's index queue, and their indexing "
            f"was not confirmed before P1 ran ({reason}); P1 may have been measured partly "
            "without its memory"
        )

    def _candidates(self, entry: Mapping[str, Any]) -> list[dict[str, Any]]:
        """A proof, and the rules it supports - or nothing, with the reason.

        All three shapes come from the arm's own trace and nothing else, and
        all three need the proof: a rule learned from an answer nobody could
        validate is a rule with no evidence behind it.
        """

        question_id = str(entry.get("question_id") or "")
        if entry.get("abstained"):
            self.skipped[question_id] = "the arm abstained"
            return []
        if not entry.get("claims"):
            self.skipped[question_id] = "the answer made no claim"
            return []
        if not any(p.get("text") for p in entry.get("read_passages") or []) and not any(
            r.get("matched_text")
            for c in entry.get("search_calls") or []
            for r in c.get("results") or []
        ):
            # Nothing to validate against. Said as itself, so a run with
            # `privacy.store_response_text: false` is not read as one whose
            # answers were all unsupported.
            self.skipped[question_id] = "the passage text was not kept, so nothing can be validated"
            return []
        proof = memory_proof.build_proof(
            entry,
            self.matching,
            max_evidence=max(memory_proof.DEFAULT_MAX_EVIDENCE, self.min_evidence),
        )
        if proof.validated < self.min_evidence:
            self.skipped[question_id] = (
                f"fewer than {self.min_evidence} retrieved passage(s) validate the answer"
                if proof.claims
                else "no claim is validated by a passage the session read"
            )
            return []

        subject = memory_proof.subject(proof.question_text)
        out: list[dict[str, Any]] = [
            {
                "kind": "fact",
                "scope": "org",
                "subject": subject,
                "text": memory_proof.render_proof(proof, principal=self.principal),
                "tags": ["retrieval-proof"],
                "proof": proof,
            }
        ]

        # An alias: a word the refined search added, mapped from one the first
        # search used - only when the refined search found validating evidence
        # and the added word is in it. Otherwise the alias is a guess about
        # vocabulary that the session never showed to matter.
        queries = proof.queries
        refined = [item for item in proof.evidence if (item.query_round or 1) > 1]
        if len(queries) >= 2 and refined:
            first = [t for t in content_terms(queries[0]) if memory_proof.is_readable_term(t)]
            # The word the refinement replaced, when it replaced one: that is
            # the vocabulary shift the session made. A refinement that only
            # added words falls back to the first search's leading term.
            dropped = [t for t in first if t not in set(content_terms(queries[-1]))]
            first = dropped + [t for t in first if t not in dropped]
            read = entry_documents(entry)
            evidence_words = {
                token
                for item in refined
                for token in tokens(memory_proof.readable(read.get(item.artifact_id, "")))
            }
            later = [
                t for t in content_terms(queries[-1]) if t not in set(content_terms(queries[0]))
            ]
            fresh = [t for t in later if memory_proof.is_readable_term(t) and t in evidence_words]
            if first and fresh:
                out.append(
                    {
                        "kind": "alias",
                        "scope": "org",
                        "subject": subject,
                        "text": memory_proof.render_alias(
                            first[0],
                            fresh[0],
                            proof,
                            first_query=queries[0],
                            refined_query=queries[-1],
                            found=len(refined),
                        ),
                        "tags": ["learned-rule"],
                        "proof": proof,
                    }
                )

        # A preference, in the region's own rule grammar (`when: <terms> ->
        # prefer: <path>`), for the document that validates the most of the
        # answer. A record the region cannot parse is ignored there without a
        # word, so the path must survive pheasant's comma split intact.
        documents = entry_documents(entry)
        best = max(proof.evidence, key=lambda item: (len(item.claims), -item.number), default=None)
        if best is not None and best.relative_path and not set(",;\n") & set(best.relative_path):
            triggers = memory_proof.trigger_terms(
                proof.question_text, documents.get(best.artifact_id, "")
            )
            if triggers:
                out.append(
                    {
                        "kind": "preference",
                        "scope": "org",
                        "subject": subject,
                        "text": memory_proof.render_preference(triggers, best, proof),
                        "tags": ["learned-rule"],
                        "proof": proof,
                    }
                )
        return out

    def _write(self, record: Mapping[str, Any], question_id: str) -> dict[str, Any] | None:
        arguments = {
            self._kb_field: self.config.knowledge_base,
            "text": record["text"],
            "scope": record.get("scope", "org"),
            "kind": record.get("kind", "fact"),
            "subject": record.get("subject"),
            "principal": self.principal,
            "sync": True,
            "tags": ["pheasant-swarm-lab", f"origin:{question_id}", *record.get("tags", [])],
        }
        try:
            outcome = self.client.call(
                self.capabilities.tool("write_memory"), arguments, idempotent=False, stage="answer"
            )
        except Exception as exc:
            if self.tracer is not None:
                self.tracer.errors.record(
                    exc,
                    stage="answer",
                    component="arms.MemorySeeder",
                    operation="memory_write",
                    resolution="skipped",
                )
            return None
        payload = outcome.result.payload() if outcome.result else {}
        body = payload if isinstance(payload, Mapping) else {}
        sync = body.get("sync")
        if isinstance(sync, Mapping) and sync.get("status") == "queued" and sync.get("task_id"):
            self.queued_tasks.add(str(sync["task_id"]))
        # Pheasant nests the stored record under `record`.
        stored = body.get("record") if isinstance(body.get("record"), Mapping) else body
        written = {
            "record_id": str(stored.get("record_id") or body.get("record_id") or ""),
            "kind": record.get("kind"),
            "scope": record.get("scope"),
            "subject": record.get("subject"),
            "text": record["text"],
            "originating_question_id": question_id,
            "outcome": body.get("outcome"),
            "created": body.get("created"),
        }
        proof = record.get("proof")
        if isinstance(proof, Proof):
            # The structured half of what the text says: which lab artifacts
            # validated which claims, found by which search. The region holds
            # the readable half; this is what a metric or a reviewer joins on.
            written["proof"] = proof.as_dict()
        if self.tracer is not None:
            self.tracer.append("memory-records.jsonl", written)
        return written


def entry_documents(entry: Mapping[str, Any]) -> dict[str, str]:
    """What the session read, by artifact id - the text a preference is checked against."""

    return {
        str(passage.get("artifact_id") or ""): str(passage.get("text") or "")
        for passage in entry.get("read_passages") or []
        if passage.get("artifact_id")
    }
