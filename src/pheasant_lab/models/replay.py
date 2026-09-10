"""The deterministic offline provider.

This is **not** a stand-in for a model and nothing in this repository reads it
as one. It is a rule-based implementation of each role, so the whole pipeline
- planning, extraction, benchmark construction, answering, scoring - runs
offline, deterministically and for nothing, and so the tests exercise the real
code paths rather than recorded transcripts of them.

Two consequences are stated wherever its numbers appear:

* It has **no prior knowledge**. The prior-only control (``C0``) therefore
  abstains everywhere under this provider, which makes ``P0 - C0`` a floor on
  the corpus's contribution rather than an estimate of it.
* Its answering is extractive: it selects sentences from what it was given and
  cites them. That is the honest shape for a deterministic answerer, and it
  means answer quality here measures *retrieval*, which is what the lab is
  for.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..budget import estimate_tokens
from ..hashing import digest
from ..textkit import (
    DISAGREEMENT_MARKERS,
    containment,
    content_terms,
    salient_terms,
    sentences,
    tokens,
    truncate,
)
from .base import ModelProvider, ModelRequest, ModelResponse

MAX_CLAIMS = 6
MIN_TERM_HITS = 2


class ReplayProvider(ModelProvider):
    """Deterministic role handlers, dispatched on ``request.schema``."""

    name = "replay"

    def complete(self, request: ModelRequest) -> ModelResponse:
        handler = getattr(self, f"_schema_{request.schema}", None)
        if handler is None:
            raise ValueError(
                f"the replay provider has no handler for schema '{request.schema}'. "
                "Add one, or run this role against a hosted provider."
            )
        data = handler(request)
        text = digest(data)  # a stable stand-in for the response body
        return ModelResponse(
            data=data,
            text=text,
            provider=self.name,
            model=self.model,
            input_tokens=estimate_tokens(request.prompt_text),
            output_tokens=estimate_tokens(text)
            + sum(estimate_tokens(str(v)) for v in data.values()),
            deterministic=True,
        )

    # -- planner -----------------------------------------------------------
    def _schema_plan(self, request: ModelRequest) -> dict[str, Any]:
        topic = request.context.get("topic") or {}
        facets: Sequence[Mapping[str, Any]] = topic.get("facets") or []
        seeds: list[str] = list(topic.get("seed_terms") or [])
        budget = int(request.context.get("max_subtopics") or len(facets))

        subtopics: list[dict[str, Any]] = []
        for index, facet in enumerate(facets[:budget]):
            label = str(facet.get("label") or facet.get("id"))
            terminology = _terminology(label, seeds, index)
            subtopics.append(
                {
                    "subtopic_id": f"sub-{facet.get('id')}",
                    "facet_ids": [facet.get("id")],
                    "question": (
                        f"What does the literature establish about {label.lower()} "
                        f"in {topic.get('title', '')}?"
                    ),
                    "terminology": terminology,
                    "providers": list(request.context.get("providers") or []),
                    "date_from": (topic.get("date_range") or {}).get("from"),
                    "date_to": (topic.get("date_range") or {}).get("to"),
                    "expect_disagreement": "contest" in label.lower() or "disput" in label.lower(),
                    "rationale": f"facet {facet.get('id')} carries weight {facet.get('weight')} in this topic",
                }
            )
        return {"subtopics": subtopics}

    # -- researcher --------------------------------------------------------
    def _schema_research(self, request: ModelRequest) -> dict[str, Any]:
        """Extract atomic claims from the candidate records handed in.

        Every claim produced here is ``direct_text`` or
        ``structured_metadata``: this provider never infers, so it can never
        mislabel an inference as text - the one failure the researcher prompt
        calls the most damaging thing the role can do.
        """

        candidates: Sequence[Mapping[str, Any]] = request.context.get("candidates") or []
        question = str(request.context.get("question") or "")
        wanted = set(content_terms(question))
        claims: list[dict[str, Any]] = []
        contradictions: list[dict[str, Any]] = []
        by_polarity: dict[str, list[str]] = {}

        for candidate in candidates:
            source_id = str(candidate.get("source_id"))
            body = str(candidate.get("abstract") or candidate.get("text") or "")
            locator = "abstract" if candidate.get("abstract") else "full_text"
            for sentence in sentences(body):
                hits = wanted & set(tokens(sentence))
                if len(sentence) < 40 or len(hits) < 1:
                    continue
                claim_type = _claim_type(sentence)
                support = _support(sentence)
                claims.append(
                    {
                        "claim_text": truncate(sentence, 400),
                        "claim_type": claim_type,
                        "source_id": source_id,
                        "locator": locator,
                        "support": support,
                        "confidence_basis": "direct_text",
                    }
                )
                key = _polarity_key(sentence, wanted)
                if key:
                    by_polarity.setdefault(key, []).append(source_id)

        for key, sources in sorted(by_polarity.items()):
            if key.startswith("negative:") and by_polarity.get("positive:" + key.split(":", 1)[1]):
                contradictions.append(
                    {
                        "about": key.split(":", 1)[1],
                        "source_ids": sorted(
                            set(sources + by_polarity["positive:" + key.split(":", 1)[1]])
                        ),
                        "severity": "material",
                        "resolved": False,
                        "resolution_note": None,
                    }
                )

        return {
            "subtopic_id": request.context.get("subtopic_id"),
            "claims": claims,
            "contradictions": contradictions,
            "gaps": list(request.context.get("known_gaps") or []),
        }

    # -- auditor -----------------------------------------------------------
    def _schema_audit(self, request: ModelRequest) -> dict[str, Any]:
        """Return the auditor's own computation.

        The coverage numbers are computed deterministically upstream and
        handed in; this role exists so that a hosted provider can *narrate*
        them without any provider being able to change them.
        """

        computed = dict(request.context.get("computed") or {})
        computed.setdefault(
            "narrative", "deterministic audit; no model narration under the replay provider"
        )
        return computed

    # -- orchestrator ------------------------------------------------------
    def _schema_orchestrate(self, request: ModelRequest) -> dict[str, Any]:
        decision = dict(request.context.get("decision") or {})
        decision.setdefault("decision", "continue")
        decision.setdefault("reason", "deterministic stopping calculus")
        decision.setdefault("assignments", [])
        return decision

    # -- benchmark builder -------------------------------------------------
    def _schema_benchmark(self, request: ModelRequest) -> dict[str, Any]:
        return dict(request.context.get("questions") or {"questions": []})

    # -- answerers ---------------------------------------------------------
    def _schema_answer(self, request: ModelRequest) -> dict[str, Any]:
        question = str(request.context.get("question") or "")
        question_type = str(request.context.get("question_type") or "atomic_fact")
        passages: Sequence[Mapping[str, Any]] = request.context.get("passages") or []
        terms = set(content_terms(question))

        if not passages:
            return {
                "answer_text": "The available knowledge does not contain evidence for this question.",
                "claims": [],
                "abstained": True,
                "abstention_reason": "no passage was retrieved",
                "queries_used": list(request.context.get("queries_used") or []),
                "uncertainty": None,
            }

        scored: list[tuple[float, str, Mapping[str, Any]]] = []
        for passage in passages:
            body = str(passage.get("text") or "")
            for sentence in sentences(body):
                sentence_tokens = set(tokens(sentence))
                hits = terms & sentence_tokens
                if len(hits) < MIN_TERM_HITS:
                    continue
                score = len(hits) + 0.25 * containment(terms, sentence_tokens)
                scored.append((score, sentence, passage))
        scored.sort(key=lambda row: (-row[0], row[1]))

        if not scored:
            return {
                "answer_text": "The retrieved passages do not establish an answer to this question.",
                "claims": [],
                "abstained": True,
                "abstention_reason": "retrieved passages carry no sentence matching the question",
                "queries_used": list(request.context.get("queries_used") or []),
                "uncertainty": None,
            }

        claims: list[dict[str, Any]] = []
        seen_sentences: set[str] = set()
        seen_sources: set[str] = set()
        wants_two_sides = question_type in {"contradiction", "synthesis", "multi_source_synthesis"}
        for _score, sentence, passage in scored:
            key = sentence.strip().lower()
            if key in seen_sentences:
                continue
            source = str(passage.get("source_id") or passage.get("artifact_id") or "")
            if (
                wants_two_sides
                and len(claims) >= 1
                and source in seen_sources
                and len(seen_sources) < 2
            ):
                continue
            seen_sentences.add(key)
            seen_sources.add(source)
            claims.append(
                {
                    "text": truncate(sentence, 400),
                    "citations": [str(passage.get("artifact_id") or passage.get("id") or "")],
                }
            )
            if len(claims) >= MAX_CLAIMS:
                break

        prefix = ""
        if question_type == "contradiction" and len(seen_sources) > 1:
            prefix = "Sources disagree. "
        elif question_type == "contradiction":
            prefix = "The retrieved evidence is contested and one position dominates it. "
        answer_text = prefix + " ".join(claim["text"] for claim in claims)
        uncertainty = None
        if question_type == "contradiction" and not (
            set(tokens(answer_text)) & DISAGREEMENT_MARKERS
        ):
            uncertainty = "the retrieved passages did not state the disagreement in words"
        return {
            "answer_text": answer_text,
            "claims": claims,
            "abstained": False,
            "abstention_reason": None,
            "queries_used": list(request.context.get("queries_used") or []),
            "uncertainty": uncertainty,
        }

    # -- query planning for the retrieval arms ------------------------------
    def _schema_queries(self, request: ModelRequest) -> dict[str, Any]:
        """Produce the search queries an answering arm will issue.

        Round 1 is the question's own content terms. Later rounds re-query
        with vocabulary taken from what round 1 returned, which is the habit
        the answerer prompt asks for and the one that finds the older
        literature.
        """

        question = str(request.context.get("question") or "")
        seen: Sequence[Mapping[str, Any]] = request.context.get("results") or []
        round_ = int(request.context.get("round") or 1)
        base = content_terms(question)
        if round_ <= 1 or not seen:
            return {"queries": [" ".join(base[:8]) or question]}
        vocabulary: list[str] = []
        for result in seen[:5]:
            vocabulary.extend(
                salient_terms(str(result.get("text") or result.get("title") or ""), limit=4)
            )
        fresh = [term for term in dict.fromkeys(vocabulary) if term not in base][:4]
        if not fresh:
            return {"queries": []}
        return {"queries": [" ".join(base[:4] + fresh)]}


def _terminology(label: str, seeds: list[str], index: int) -> list[str]:
    terms = content_terms(label)
    if seeds:
        terms.append(seeds[index % len(seeds)])
    return sorted(dict.fromkeys(terms))


def _claim_type(sentence: str) -> str:
    lowered = sentence.lower()
    if any(word in lowered for word in ("we measured", "method", "assay", "protocol", "we used")):
        return "method"
    if any(
        word in lowered for word in ("limitation", "caveat", "did not", "failed to", "could not")
    ):
        return "limitation"
    if any(word in lowered for word in ("is defined", "refers to", "is a ")):
        return "definition"
    if any(
        word in lowered
        for word in ("we found", "results show", "increased", "decreased", "reduced", "%")
    ):
        return "result"
    return "observation"


def _support(sentence: str) -> str:
    lowered = sentence.lower()
    if any(
        word in lowered
        for word in ("in contrast", "however", "contrary", "did not replicate", "we found no")
    ):
        return "contradicts"
    if any(word in lowered for word in ("only when", "depends on", "under", "limited to")):
        return "qualifies"
    return "supports"


def _polarity_key(sentence: str, wanted: set[str]) -> str | None:
    hits = sorted(wanted & set(tokens(sentence)))
    if not hits:
        return None
    subject = hits[0]
    negative = any(
        word in sentence.lower() for word in ("no ", "not ", "failed", "did not", "absence of")
    )
    return ("negative:" if negative else "positive:") + subject
