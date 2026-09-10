"""Canonical identifiers.

Ids are immutable and content-addressed wherever the thing they name has
content. A correction creates a superseding record; it never rewrites a raw
event, because an id that can be rewritten is an id no downstream number can
be resolved through.

Two ids deliberately carry no clock:

``run_id``
    digests the experiment name, the resolved configuration and a start
    nonce. The nonce is what separates two runs of one configuration; the
    clock is not, because two runs a second apart must be two rows and two
    runs of an unchanged configuration in one second must not collapse into
    one.

``question_id``
    digests the normalised question text and the benchmark version, so the
    same question in two runs is the same question and a trend line exists.
"""

from __future__ import annotations

import os
import re
import unicodedata

from .hashing import digest, short

_WHITESPACE = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def normalise_text(text: str, *, strip_punctuation: bool = True) -> str:
    """Fold ``text`` to the spelling ids and matchers agree on."""

    folded = unicodedata.normalize("NFKD", text or "")
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    folded = folded.casefold()
    if strip_punctuation:
        folded = _PUNCT.sub(" ", folded)
    return _WHITESPACE.sub(" ", folded).strip()


def new_nonce(size: int = 16) -> str:
    """A per-attempt nonce.

    Deliberately random rather than ``host:pid``: two threads in one process
    share a pid, which makes an in-process race indistinguishable from a
    re-entrant acquire.
    """

    return os.urandom(size).hex()


def run_id(experiment_name: str, config_digest: str, nonce: str) -> str:
    return "run-" + short(digest({"e": experiment_name, "c": config_digest, "n": nonce}), 16)


def topic_id(raw: str) -> str:
    return raw if raw.startswith("topic-") else "topic-" + short(digest(normalise_text(raw)))


def subtopic_id(topic: str, question: str) -> str:
    return "sub-" + short(digest({"t": topic, "q": normalise_text(question)}))


def agent_id(role: str, run: str, ordinal: int) -> str:
    return f"agent-{role}-{ordinal:03d}-" + short(digest({"r": run, "role": role, "i": ordinal}), 8)


def source_id(stable_identifier: str | None, canonical_url: str | None, content_digest: str) -> str:
    """Stable identifier when there is one, otherwise URL plus content."""

    if stable_identifier:
        return "source-" + short(
            digest({"sid": normalise_text(stable_identifier, strip_punctuation=False)}), 16
        )
    return "source-" + short(digest({"u": canonical_url or "", "d": content_digest}), 16)


def claim_id(claim_text: str, source: str, locator: str, extraction_profile: str) -> str:
    return "claim-" + short(
        digest(
            {"t": normalise_text(claim_text), "s": source, "l": locator, "p": extraction_profile}
        ),
        16,
    )


def question_id(question_text: str, benchmark_version: str) -> str:
    return "q-" + short(digest({"t": normalise_text(question_text), "v": benchmark_version}), 16)


def fact_id(claim: str, question: str) -> str:
    return "fact-" + short(digest({"c": claim, "q": question}), 16)


def event_id(run: str, sequence: int, payload_digest: str) -> str:
    return "event-" + short(digest({"r": run, "s": sequence, "p": payload_digest}), 16)


def span_id() -> str:
    """A 64-bit span id, W3C-shaped."""

    return os.urandom(8).hex()


def trace_id() -> str:
    """A 128-bit trace id, W3C-shaped."""

    return os.urandom(16).hex()


def error_id(run: str, event: str | None, attempt: int, payload_digest: str) -> str:
    return "error-" + short(digest({"r": run, "e": event, "a": attempt, "p": payload_digest}), 16)


def metric_result_id(metric: str, version: str, scope: dict, operand_digest: str) -> str:
    return "metric-" + short(
        digest({"m": metric, "v": version, "s": scope, "o": operand_digest}), 16
    )


def ingest_request_id(source: str, content_digest: str) -> str:
    return "ingest-" + short(digest({"s": source, "d": content_digest}), 16)


def idempotency_key(namespace: str, source: str, content_digest: str) -> str:
    """The key a retry carries so the region folds rather than duplicates."""

    return f"{namespace}:{source}:{short(content_digest, 32)}"


def search_request_id(run: str, arm: str, question: str, query: str, round_: int) -> str:
    return "search-" + short(
        digest({"r": run, "a": arm, "q": question, "t": normalise_text(query), "n": round_}), 16
    )


def answer_id(run: str, arm: str, question: str, repetition: int) -> str:
    return "answer-" + short(digest({"r": run, "a": arm, "q": question, "i": repetition}), 16)


def candidate_id(target: str, trigger: str, evidence_digest: str) -> str:
    return "refine-" + short(digest({"t": target, "g": trigger, "e": evidence_digest}), 16)


def contradiction_id(subtopic: str, about: str) -> str:
    return "contra-" + short(digest({"s": subtopic, "a": normalise_text(about)}), 16)


def proof_id(question: str, arm: str, target: str, event_type: str) -> str:
    return "proof-" + short(digest({"q": question, "a": arm, "t": target, "e": event_type}), 16)


def benchmark_version(ledger_digest: str, composition: dict[str, int]) -> str:
    """The benchmark's own version, which every question id folds in.

    Content-addressed over the evidence ledger and the question mix, and
    **not** over the run. A version that carried the run id would give the
    same question a different id in every run, and there would be no trend
    line to draw: the whole point of folding the version into ``question_id``
    is that one question asked of two runs is one question.
    """

    return "bench-" + short(
        digest({"l": ledger_digest, "c": dict(sorted(composition.items()))}), 12
    )
