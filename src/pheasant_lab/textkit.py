"""Deterministic text handling shared by extraction, answering and scoring.

One home, because a matcher that normalises differently from the extractor
that produced its operand is a matcher that reports misses nobody made. Every
function here is pure and its output is a function of its input alone.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")
_TOKEN = re.compile(r"[a-z0-9][a-z0-9\-']*")
_NUMBER = re.compile(r"-?\d+(?:[.,]\d+)*(?:[eE][-+]?\d+)?")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WHITESPACE = re.compile(r"\s+")

# Framing words a query expansion should drop, and that never carry a claim.
STOPWORDS = frozenset(
    [
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "also",
        "am",
        "an",
        "and",
        "any",
        "are",
        "aren't",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "cannot",
        "could",
        "couldn't",
        "did",
        "didn't",
        "do",
        "does",
        "doesn't",
        "doing",
        "don't",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "hadn't",
        "has",
        "hasn't",
        "have",
        "haven't",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "however",
        "i",
        "if",
        "in",
        "into",
        "is",
        "isn't",
        "it",
        "its",
        "itself",
        "let's",
        "me",
        "more",
        "most",
        "mustn't",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "ought",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "same",
        "shan't",
        "she",
        "should",
        "shouldn't",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "wasn't",
        "we",
        "were",
        "weren't",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "with",
        "won't",
        "would",
        "wouldn't",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "according",
        "report",
        "reports",
        "reported",
        "study",
        "studies",
        "paper",
        "papers",
        "according-to",
        "based",
    ]
)

NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "twenty": "20",
    "thirty": "30",
    "forty": "40",
    "fifty": "50",
    "hundred": "100",
    "thousand": "1000",
}

# Words that mark an answer as reporting a disagreement rather than picking a
# side. Used by the contradiction-handling metric, which is why the list is
# here and not inlined at the metric.
DISAGREEMENT_MARKERS = frozenset(
    {
        "disagree",
        "disagreement",
        "contradict",
        "contradicts",
        "contradictory",
        "conflicting",
        "conflict",
        "dispute",
        "disputed",
        "contested",
        "inconsistent",
        "however",
        "whereas",
        "but",
        "although",
        "unresolved",
        "debate",
        "debated",
    }
)

ABSTENTION_MARKERS = frozenset(
    {
        "no evidence",
        "not contain",
        "does not contain",
        "cannot answer",
        "no information",
        "not available",
        "insufficient evidence",
        "no results",
        "nothing retrieved",
        "not established",
        "unable to answer",
        "no supporting",
        "not covered",
    }
)


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalise(
    text: str,
    *,
    casefold: bool = True,
    accents: bool = True,
    punctuation: bool = True,
    numbers: bool = True,
) -> str:
    """The one folded spelling matchers compare against."""

    out = text or ""
    if accents:
        out = strip_accents(out)
    if casefold:
        out = out.casefold()
    if numbers:
        out = " ".join(NUMBER_WORDS.get(word, word) for word in out.split())
    if punctuation:
        out = _PUNCT.sub(" ", out)
    return _WHITESPACE.sub(" ", out).strip()


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(normalise(text))


def content_terms(text: str, *, minimum_length: int = 3) -> list[str]:
    """Query terms with framing words dropped, order preserved, deduplicated."""

    seen: set[str] = set()
    out: list[str] = []
    for token in tokens(text):
        if token in STOPWORDS or len(token) < minimum_length:
            continue
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def sentences(text: str) -> list[str]:
    """Split into sentences, conservatively.

    Abbreviations and decimals defeat a naive split, so the boundary requires
    whitespace followed by something that can start a sentence, and a fragment
    shorter than four characters is folded back into its predecessor.
    """

    raw = [part.strip() for part in _SENTENCE_END.split((text or "").strip()) if part.strip()]
    merged: list[str] = []
    for part in raw:
        if merged and len(part) < 4:
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)
    return merged


def numbers(text: str) -> list[float]:
    found: list[float] = []
    for match in _NUMBER.finditer(text or ""):
        try:
            found.append(float(match.group(0).replace(",", "")))
        except ValueError:
            continue
    return found


def jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    a, b = set(left), set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def containment(needle: Iterable[str], haystack: Iterable[str]) -> float:
    """Fraction of ``needle``'s tokens present in ``haystack``.

    Asymmetric on purpose: "is this claim's content present in this passage"
    is not the same question as "are these two texts similar", and using a
    symmetric measure for it penalises a long passage for being long.
    """

    a, b = set(needle), set(haystack)
    if not a:
        return 0.0
    return len(a & b) / len(a)


def salient_terms(text: str, *, limit: int = 8) -> list[str]:
    """The terms a question would be built from: content words plus numbers."""

    terms = content_terms(text)
    ordered = sorted(terms, key=lambda t: (-len(t), t))[:limit]
    return sorted(ordered)


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"
