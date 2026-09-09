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

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from ..pheasant.capabilities import CapabilityMap
from ..pheasant.client import PheasantClient
from ..pheasant.retrieval import MemoryOptions
from ..settings import PheasantFile
from ..textkit import content_terms, truncate
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


class MemorySeeder:
    """Writes the memory ``P1`` is measured with.

    Every record names the question that produced it. That field is what the
    leakage checker reads to prove no holdout or control question created the
    treatment it is being tested against - so it is written even though
    nothing in the region requires it.
    """

    def __init__(
        self,
        client: PheasantClient,
        capabilities: CapabilityMap,
        config: PheasantFile,
        *,
        tracer: Any = None,
        principal: str = "pheasant-swarm-lab",
    ) -> None:
        self.client = client
        self.capabilities = capabilities
        self.config = config
        self.tracer = tracer
        self.principal = principal
        self._kb_field = str(config.argument_map.get("knowledge_base_field", "knowledge_base"))
        self.written: list[dict[str, Any]] = []

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

        for entry in first_pass:
            question_id = str(entry.get("question_id") or "")
            if question_id not in learned or question_id in forbidden:
                continue
            for record in self._candidates(entry):
                written = self._write(record, question_id)
                if written is not None:
                    records.append(written)
        self.written.extend(records)
        if self.tracer is not None:
            self.tracer.emit(
                "memory.seeded",
                payload={"records": len(records), "learned_questions": len(learned)},
            )
        return records

    def _candidates(self, entry: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Three rule shapes, each derived from the arm's own trace."""

        out: list[dict[str, Any]] = []
        queries = [str(q) for q in entry.get("queries_used") or []]
        artifacts = [str(a) for a in entry.get("retrieved_artifact_ids") or []]
        question = str(entry.get("question_text") or "")

        # 1. A fact record: what this session actually established, in words.
        if entry.get("answer_text") and not entry.get("abstained"):
            out.append(
                {
                    "kind": "fact",
                    "scope": "org",
                    "subject": _subject(question),
                    "text": truncate(str(entry["answer_text"]), 600),
                }
            )
        # 2. An alias: a query term that retrieved nothing, mapped to one that
        #    did. This is the shape a region's own formation rules propose,
        #    and the one most likely to be a false positive, so it is only
        #    emitted when the two rounds actually differed.
        if len(queries) >= 2 and artifacts:
            first, later = content_terms(queries[0]), content_terms(queries[-1])
            fresh = [term for term in later if term not in first]
            if fresh and first:
                out.append(
                    {
                        "kind": "alias",
                        "scope": "org",
                        "subject": _subject(question),
                        "text": f"{first[0]} -> {fresh[0]}",
                    }
                )
        # 3. A preference: the source the session found useful.
        if artifacts:
            out.append(
                {
                    "kind": "preference",
                    "scope": "org",
                    "subject": _subject(question),
                    "text": f"prefer material like {artifacts[0]} for {_subject(question)}",
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
            "tags": ["pheasant-swarm-lab", f"origin:{question_id}"],
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
        written = {
            "record_id": str(body.get("record_id") or ""),
            "kind": record.get("kind"),
            "scope": record.get("scope"),
            "subject": record.get("subject"),
            "text": record["text"],
            "originating_question_id": question_id,
            "outcome": body.get("outcome"),
            "created": body.get("created"),
        }
        if self.tracer is not None:
            self.tracer.append("memory-records.jsonl", written)
        return written


def _subject(question: str) -> str:
    terms = content_terms(question)[:3]
    return " ".join(terms) or "general"
