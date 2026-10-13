"""What a seeded memory record says, in words a person can read.

A memory the lab writes is the record of one learned question: the question,
the claims the arm made that its own retrieval *validates*, and the passages
that validate them - each named by its title, linked by the region's own path,
found by a stated search at a stated rank, and quoted. A person reading it in
the region can check every line of it; the region can link it to the files it
cites.

"Validated" is the rule ``evidence_support_rate`` already scores with
(:func:`~pheasant_lab.evaluation.answer_metrics.supporting_citations`): a
claim cites a passage this session read, and enough of the claim's words are
in that passage. One rule for both, because a memory that called a claim
proven by a looser test than the report would be a proof the report does not
stand behind. It is token containment, not entailment, and the record says so.

The body is shaped for what pheasant does with it, not only for the reader:

* each cited document is a Markdown link to its ``relative_path``, which
  pheasant's cross-source pass resolves to the real file, so the record gains
  an ``about`` edge to it (the "reference" rung, confidence 1.0) and the graph
  arm can walk from a document to the memory that cites it and back;
* no ``#`` headings - a heading in a memory record is a graph node the bridge
  ignores, so labels are bold text instead;
* it stays under one chunk (pheasant's ``fixed`` chunker cuts at 4,000
  characters), so a search hit on the record carries the whole proof rather
  than half of it.

A steering record (``alias``, ``preference``) keeps its rule on the **first
line**, exactly where pheasant's parser reads it, and explains itself on the
lines after. Nothing in the explanation may parse as a rule: pheasant reads a
preference with a regex over the whole text, so the explanation is scrubbed of
the rule grammar's own markers.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..textkit import containment, content_terms, sentences, tokens, truncate

#: The shape of the text a proof record carries, recorded with it in the
#: lab's trace so a reader of an old run knows which layout it is looking at.
PROOF_FORMAT = "retrieval-proof/1"

#: pheasant links a memory to at most three targets per rung
#: (``memory.bridge.DEFAULT_MAX_TARGETS``). Listing more evidence than that
#: by default would show the reader links the graph does not have.
DEFAULT_MAX_EVIDENCE = 3
MAX_CLAIMS = 4
CLAIM_CHARS = 260
EXCERPT_CHARS = 240
TITLE_CHARS = 140
SUBJECT_CHARS = 160
#: Under pheasant's fixed chunk ceiling (4,000) with room to spare.
MAX_RECORD_CHARS = 3_800
#: Below this a claim is not an assertion (`answer_metrics.eligible_claims`).
MIN_CLAIM_CHARS = 20

_FRONT_MATTER = re.compile(r"\A\s*---\r?\n.*?\r?\n---[ \t]*(?:\r?\n|\Z)", re.S)
#: A front-matter block that a preview cut off before its closing fence.
_FRONT_MATTER_OPEN = re.compile(r"\A\s*---\r?\n(?:[A-Za-z_][\w-]*:[^\n]*(?:\n|\Z))+", re.S)
_HEADING_MARK = re.compile(r"^\s{0,3}#{1,6}\s+", re.M)
_FIELD_LINE = re.compile(r"^\s*([A-Za-z_][\w-]*)\s*:\s*(.*)$")
#: An opaque id: a run of hex long enough to be a digest, or a token that is
#: mostly digits. Neither is a word anybody searches for.
_OPAQUE = re.compile(r"^(?:[0-9a-f]{8,}|.*\d.*\d.*\d.*)$")
#: The markers pheasant's steering parser keys on. An explanation containing
#: one could be read as a second rule - or as the first one.
_RULE_MARKERS = re.compile(r"(->|→|=>|\b(?:when|never|prefer)\s*:)", re.I)


# -- the evidence ------------------------------------------------------------


@dataclass
class Evidence:
    """One passage that validates at least one claim."""

    number: int
    artifact_id: str
    title: str
    relative_path: str | None
    region_source: str | None
    rank: int | None
    score: float | None
    query: str | None
    query_round: int | None
    retrieval_arm: str | None
    excerpt: str
    claims: list[int] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "artifact_id": self.artifact_id,
            "title": self.title,
            "relative_path": self.relative_path,
            "region_source": self.region_source,
            "rank": self.rank,
            "score": self.score,
            "query": self.query,
            "query_round": self.query_round,
            "retrieval_arm": self.retrieval_arm,
            "excerpt": self.excerpt,
            "supports_claims": list(self.claims),
        }


@dataclass
class Proof:
    """One learned question's answer, reduced to what its retrieval proves."""

    question_id: str
    question_text: str
    claims: list[dict[str, Any]]
    evidence: list[Evidence]
    claims_total: int
    rejected: list[dict[str, str]]
    passages_read: int
    queries: list[str]
    answer_id: str | None = None
    snapshot_id: str | None = None
    support_threshold: float = 0.5

    @property
    def validated(self) -> int:
        """Distinct passages that validate the answer - the ``k`` a record needs."""

        return len(self.evidence)

    def as_dict(self) -> dict[str, Any]:
        return {
            "format": PROOF_FORMAT,
            "question_id": self.question_id,
            "question_text": self.question_text,
            "answer_id": self.answer_id,
            "snapshot_id": self.snapshot_id,
            "validated_claims": [dict(c) for c in self.claims],
            "rejected_claims": [dict(r) for r in self.rejected],
            "claims_total": self.claims_total,
            "evidence": [e.as_dict() for e in self.evidence],
            "validated_passages": self.validated,
            "passages_read": self.passages_read,
            "queries": list(self.queries),
            "support_threshold": self.support_threshold,
        }


def build_proof(
    entry: Mapping[str, Any],
    matching: Any,
    *,
    max_evidence: int = DEFAULT_MAX_EVIDENCE,
) -> Proof:
    """Validate one first-pass answer against what its session read.

    ``entry`` is the engine's first-pass row: the question, the arm's claims
    with their citations, ``read_passages`` and ``search_calls``. Nothing
    else - in particular never the benchmark's expected answers, which the
    seeder is never given. A claim is kept only when
    :func:`supporting_citations` finds a passage that carries it; the rest are
    counted, with the reason, and left out of the record.
    """

    # Imported here: `evaluation` imports the engine, which imports `arms`,
    # so a module-level import would be a cycle. The rule itself lives with
    # the metric it was written for.
    from ..evaluation.answer_metrics import retrieved_passages, supporting_citations

    passages = retrieved_passages(entry)
    hits = _first_hits(entry.get("search_calls") or [])
    evidence: dict[str, Evidence] = {}
    claims: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    raw_claims = list(entry.get("claims") or [])

    for claim in raw_claims:
        text = readable(str(claim.get("text") or ""))
        citations = [str(c) for c in claim.get("citations") or [] if c]
        reason = _ineligible(text)
        if reason is None and not citations:
            reason = "cites nothing"
        if reason is None and not any(c in passages for c in citations):
            reason = "cites only passages this session did not read"
        supporting = (
            supporting_citations(text, citations, passages, matching) if reason is None else []
        )
        if reason is None and not supporting:
            reason = "the cited passage does not carry the claim's words"
        numbers: list[int] = []
        for artifact_id in supporting:
            item = evidence.get(artifact_id)
            if item is None:
                if len(evidence) >= max_evidence:
                    continue
                item = _evidence(len(evidence) + 1, artifact_id, passages, hits)
                evidence[artifact_id] = item
            numbers.append(item.number)
        if reason is None and not numbers:
            reason = f"its passages fall past the {max_evidence} the record lists"
        if reason is not None:
            rejected.append(
                {"text": truncate(text or str(claim.get("text") or ""), 160), "reason": reason}
            )
            continue
        if len(claims) >= MAX_CLAIMS:
            rejected.append(
                {
                    "text": truncate(text, 160),
                    "reason": f"past the first {MAX_CLAIMS} validated claims",
                }
            )
            continue
        index = len(claims) + 1
        claims.append({"text": truncate(text, CLAIM_CHARS), "evidence": numbers})
        for number in numbers:
            for item in evidence.values():
                if item.number == number and index not in item.claims:
                    item.claims.append(index)

    # Evidence no kept claim points at (a claim dropped by the cap) is not
    # evidence for this record.
    kept = [item for item in evidence.values() if item.claims]
    renumber = {item.number: position for position, item in enumerate(kept, start=1)}
    for item in kept:
        item.number = renumber[item.number]
    for claim in claims:
        claim["evidence"] = [renumber[n] for n in claim["evidence"] if n in renumber]
    for item in kept:
        item.excerpt = _excerpt(
            passages.get(item.artifact_id, ""),
            " ".join(claims[i - 1]["text"] for i in item.claims),
        )

    return Proof(
        question_id=str(entry.get("question_id") or ""),
        question_text=str(entry.get("question_text") or ""),
        claims=claims,
        evidence=kept,
        claims_total=len(raw_claims),
        rejected=rejected,
        passages_read=len(passages),
        queries=[str(q) for q in entry.get("queries_used") or []],
        answer_id=entry.get("answer_id"),
        snapshot_id=entry.get("snapshot_id"),
        support_threshold=float(getattr(matching, "support_overlap_threshold", 0.5)),
    )


def _ineligible(text: str) -> str | None:
    if len(text) < MIN_CLAIM_CHARS:
        return "too short to be an assertion"
    if text.endswith(("…", "...")):
        # Whoever wrote the claim cut it off; half a sentence is not a
        # statement a proof can stand behind, however well its words match.
        return "cut off mid-sentence"
    folded = text.lower()
    if len(folded) < 80 and any(
        marker in folded
        for marker in ("sources disagree", "the retrieved evidence", "the available knowledge")
    ):
        return "frames the answer rather than asserting anything"
    return None


def _first_hits(calls: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Each artifact's first appearance: the search that found it, and where."""

    out: dict[str, dict[str, Any]] = {}
    for call in calls:
        for result in call.get("results") or []:
            artifact_id = str(result.get("artifact_id") or "")
            if artifact_id and artifact_id not in out:
                out[artifact_id] = {
                    **dict(result),
                    "query": call.get("query"),
                    "round": call.get("round"),
                }
    return out


def _evidence(
    number: int,
    artifact_id: str,
    passages: Mapping[str, str],
    hits: Mapping[str, Mapping[str, Any]],
) -> Evidence:
    hit = hits.get(artifact_id) or {}
    body = passages.get(artifact_id, "")
    path = _relative_path(hit, artifact_id)
    return Evidence(
        number=number,
        artifact_id=artifact_id,
        title=_title(hit, body, path, artifact_id),
        relative_path=path,
        region_source=hit.get("region_source"),
        rank=hit.get("rank"),
        score=hit.get("score"),
        query=hit.get("query"),
        query_round=hit.get("round"),
        retrieval_arm=hit.get("retrieval_arm"),
        excerpt="",
    )


def _relative_path(hit: Mapping[str, Any], artifact_id: str) -> str | None:
    """The region path a Markdown link resolves against, or ``None``.

    The hit's own ``relative_path`` first. Then pheasant's artifact id, whose
    grammar (``file:{source}:{relpath}:branch={b}``) is a documented contract
    of the region's. Then ``locator`` - only when it looks like a file, since
    under taxonomy it is a section heading.
    """

    if hit.get("relative_path"):
        return str(hit["relative_path"])
    match = re.match(r"^file:[^:]+:(?P<path>.+):branch=[^:]*$", artifact_id)
    if match:
        return match.group("path")
    locator = str(hit.get("locator") or "")
    if "/" in locator or re.search(r"\.[A-Za-z0-9]{1,5}$", locator):
        return locator
    return None


def _title(hit: Mapping[str, Any], body: str, path: str | None, artifact_id: str) -> str:
    title = str(hit.get("title") or "").strip()
    if not title or title == path:
        title = front_matter(body).get("title", "") or _first_heading(body)
    if not title:
        title = (path or artifact_id).rsplit("/", 1)[-1]
    return truncate(" ".join(title.split()), TITLE_CHARS)


def _excerpt(passage: str, claim_text: str) -> str:
    """The passage's sentence that carries the claim best, quoted verbatim."""

    wanted = set(tokens(claim_text))
    best, best_score = "", 0.0
    # Line by line before sentence by sentence: a heading has no full stop,
    # and joined to the paragraph under it would read as one run-on sentence.
    lines = _FRONT_MATTER.sub("", passage or "", count=1).splitlines()
    units = [s for line in lines if line.strip() for s in sentences(readable(line))]
    for sentence in units:
        score = containment(wanted, tokens(sentence)) if wanted else 0.0
        if score > best_score:
            best, best_score = sentence, score
    return truncate(" ".join(best.split()), EXCERPT_CHARS)


# -- the text a person reads ---------------------------------------------------


def readable(text: str) -> str:
    """``text`` without the front matter or heading marks a document opens with.

    This lab's documents start with a YAML block, and a passage or a claim
    copied from one carries it - which is how a memory record ended up
    reading ``lab_source_id: source-…``. A block a preview truncated before
    its closing fence is stripped too.
    """

    body = _FRONT_MATTER.sub("", text or "", count=1)
    if body == (text or ""):
        body = _FRONT_MATTER_OPEN.sub("", body, count=1)
    body = _HEADING_MARK.sub("", body)
    return " ".join(body.split())


def front_matter(text: str) -> dict[str, str]:
    match = _FRONT_MATTER.match(text or "")
    out: dict[str, str] = {}
    if match:
        for line in match.group(0).splitlines():
            field_match = _FIELD_LINE.match(line)
            if field_match and field_match.group(2).strip():
                out.setdefault(field_match.group(1), field_match.group(2).strip())
    return out


def _first_heading(text: str) -> str:
    for line in _FRONT_MATTER.sub("", text or "", count=1).splitlines():
        if _HEADING_MARK.match(line):
            return _HEADING_MARK.sub("", line).strip()
    return ""


def subject(question: str) -> str:
    """The record's subject: the question itself, on one line.

    pheasant refuses a line break in any frontmatter field, and the subject
    is what its memory views show beside a record - so it should be the thing
    a person would recognise, not three stemmed keywords.
    """

    return truncate(" ".join((question or "").split()), SUBJECT_CHARS) or "general"


def is_readable_term(term: str) -> bool:
    """A word a person would search for: not a digest, not an id."""

    return bool(term) and term.isascii() and not _OPAQUE.match(term.lower())


def _inline(text: str) -> str:
    """Prose safe to place beside a rule: no rule markers, no link syntax."""

    text = _RULE_MARKERS.sub(_defuse, text or "")
    return " ".join(text.replace("[", "(").replace("]", ")").split())


def _defuse(match: re.Match[str]) -> str:
    marker = match.group(0)
    return "to" if marker in {"->", "→", "=>"} else marker.rstrip(": \t")


def _link(item: Evidence) -> str:
    title = _inline(item.title)
    path = item.relative_path
    # pheasant reads `[label](target)` with `[^)]+` for the target, so a path
    # with a parenthesis or a space would be cut short; name it instead.
    if path and not re.search(r"[\s()<>]", path):
        return f"[{title}]({path})"
    return f"{title} ({_inline(path)})" if path else title


def render_proof(proof: Proof, *, principal: str) -> str:
    """The record's text. Trimmed claim by claim until it fits one chunk."""

    claims = list(proof.claims)
    while True:
        text = _render(proof, claims, principal)
        if len(text) <= MAX_RECORD_CHARS or len(claims) <= 1:
            return text
        claims = claims[:-1]


def _render(proof: Proof, claims: Sequence[Mapping[str, Any]], principal: str) -> str:
    used = sorted({n for claim in claims for n in claim["evidence"]})
    evidence = [item for item in proof.evidence if item.number in used]
    lines = [
        f"**Question:** {_inline(proof.question_text)}",
        "",
        f"**Answer**, validated by {len(evidence)} of the {proof.passages_read} {passages_word(proof.passages_read)} "
        "this session retrieved:",
        "",
    ]
    for claim in claims:
        marks = "".join(f"[{n}]" for n in claim["evidence"])
        lines.append(f"- {_inline(claim['text'])} {marks}")
    lines += ["", "**Evidence**", ""]
    for item in evidence:
        lines.append(f"[{item.number}] {_link(item)}")
        found = []
        if item.rank is not None:
            found.append(f"rank {item.rank}")
        if item.query:
            found.append(f'for the search "{_inline(str(item.query))}"')
        if item.retrieval_arm:
            found.append(f"({item.retrieval_arm} arm)")
        if found:
            lines.append(f"    Found at {' '.join(found)}.")
        if item.excerpt:
            lines.append(f"    > {_inline(item.excerpt)}")
        lines.append("")
    left_out = proof.claims_total - len(claims)
    checked = (
        f"**How this was checked:** each claim above cites a passage this session read, and at "
        f"least {proof.support_threshold:.0%} of the claim's words appear in that passage (word "
        f"overlap, not entailment). {len(claims)} of {proof.claims_total} claims passed"
        + (
            f"; the other {left_out} {'is' if left_out == 1 else 'are'} left out."
            if left_out
            else "."
        )
    )
    lines.append(checked)
    origin = f"Written by {principal} from learned question {proof.question_id}"
    if proof.answer_id:
        origin += f", answer {proof.answer_id}"
    if proof.snapshot_id:
        origin += f", snapshot {proof.snapshot_id}"
    lines.append(origin + ".")
    return "\n".join(lines).strip() + "\n"


def passages_word(count: int) -> str:
    return "passage" if count == 1 else "passages"


def render_alias(
    term: str, expansion: str, proof: Proof, *, first_query: str, refined_query: str, found: int
) -> str:
    """An alias rule, then why the lab believes it."""

    return (
        f"{term} -> {expansion}\n\n"
        f'Searches that use "{_inline(term)}" also look for "{_inline(expansion)}". '
        f"Learned from the question: {_inline(proof.question_text)} "
        f'Its first search, "{_inline(first_query)}", was refined to "{_inline(refined_query)}", '
        f"and the refined search found {found} of the {proof.validated} "
        f"{passages_word(proof.validated)} that validate the answer recorded for that question.\n"
    )


def render_preference(triggers: Sequence[str], item: Evidence, proof: Proof) -> str:
    """A preference rule, then which validated document it prefers and why."""

    words = ", ".join(triggers)
    named = _inline(item.title)
    return (
        f"when: {words} -> prefer: {item.relative_path}\n\n"
        f'Searches using any of the words {_inline(words)} rank "{named}" higher. '
        f"Learned from the question: {_inline(proof.question_text)} "
        f"That document is evidence [{item.number}] in the validated answer recorded for this "
        f"question: it supports {len(item.claims)} of its {len(proof.claims)} validated "
        f"{'claim' if len(proof.claims) == 1 else 'claims'}.\n"
    )


def trigger_terms(question: str, document: str, limit: int = 3) -> list[str]:
    """Question words the preferred document also uses, readable ones only.

    pheasant fires a preference when *any* trigger is in the query, so a
    trigger the document never mentions would steer unrelated searches
    towards it. Requiring the document to share the word keeps the rule about
    what the document is about.
    """

    present = set(tokens(readable(document)))
    return [term for term in content_terms(question) if is_readable_term(term) and term in present][
        :limit
    ]
