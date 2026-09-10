"""``P2`` - tuned-search replay.

First-pass question, answer and error evidence may tune the **search
strategy**; the answers are then regenerated fresh. What it measures is query
adaptation, not a better corpus and not a better model.

The line this module holds: tuning is derived from what the *arm* observed -
a query that returned nothing, a result set that came from one source, a round
that added nothing - and never from the expected answers or the known-positive
sets, which this class is never given. Tuning on the answer key would produce
an improvement that means nothing and confirms itself.

``P2`` is reported in two cohorts. **Learned replay** is the questions whose
first-pass evidence created the strategy; **holdout** is the questions that
did not. Improvement on learned replay is not generalization, and the report
never presents it as such.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from ..pheasant.retrieval import MemoryOptions
from ..textkit import content_terms, tokens
from .pheasant_corpus import PheasantCorpusArm

MIN_SUPPORT = 2


class TunedReplayArm(PheasantCorpusArm):
    arm_id = "P2"
    memory_enabled = True

    def memory_options(self) -> MemoryOptions:
        strategy = dict(self.context.query_strategy or {})
        return MemoryOptions(
            enabled=bool(strategy.get("memory_enabled", True)),
            current_only=True,
            include_rules=False,
        )

    def _queries(self, question: Any, responses: list[Any], round_number: int) -> list[str]:
        queries = super()._queries(question, responses, round_number)
        strategy = dict(self.context.query_strategy or {})
        expansions: Mapping[str, list[str]] = strategy.get("expansions") or {}
        drop: set[str] = set(strategy.get("drop_terms") or [])

        tuned: list[str] = []
        for query in queries:
            terms = [term for term in content_terms(query) if term not in drop]
            for term in list(terms):
                for addition in expansions.get(term, []):
                    if addition not in terms:
                        terms.append(addition)
            tuned.append(" ".join(terms) or query)
        return tuned


class QueryTuner:
    """Derives a query strategy from a first pass.

    Every rule carries its support count, and a rule below ``MIN_SUPPORT`` is
    not emitted: a "strategy" learned from one query is a coincidence with a
    name.
    """

    def __init__(self, *, minimum_support: int = MIN_SUPPORT) -> None:
        self.minimum_support = minimum_support

    def tune(
        self,
        first_pass: Sequence[Mapping[str, Any]],
        *,
        learned_question_ids: Iterable[str],
    ) -> dict[str, Any]:
        learned = set(learned_question_ids)
        expansions: dict[str, Counter] = {}
        empty_terms: Counter = Counter()
        supported_by: dict[str, set[str]] = {}

        for entry in first_pass:
            question_id = str(entry.get("question_id") or "")
            if question_id not in learned:
                continue
            calls = list(entry.get("search_calls") or [])
            productive = [call for call in calls if call.get("results")]
            barren = [call for call in calls if not call.get("results")]

            for call in barren:
                for term in content_terms(str(call.get("query") or "")):
                    empty_terms[term] += 1

            # A later round that produced results after an earlier one did not
            # is the evidence for an expansion: the vocabulary that worked,
            # keyed by the vocabulary that did not.
            if barren and productive:
                failed_terms = content_terms(str(barren[0].get("query") or ""))
                working = _result_vocabulary(productive)
                for term in failed_terms[:3]:
                    counter = expansions.setdefault(term, Counter())
                    for addition in working[:3]:
                        if addition != term:
                            counter[addition] += 1
                            supported_by.setdefault(f"{term}->{addition}", set()).add(question_id)

        rules = {
            term: [
                addition
                for addition, count in counter.most_common(3)
                if count >= self.minimum_support
            ]
            for term, counter in expansions.items()
        }
        rules = {term: additions for term, additions in rules.items() if additions}
        drop = [term for term, count in empty_terms.items() if count >= self.minimum_support * 2]

        return {
            "expansions": rules,
            "drop_terms": sorted(drop),
            "memory_enabled": True,
            "support": {
                key: sorted(question_ids)
                for key, question_ids in sorted(supported_by.items())
                if len(question_ids) >= self.minimum_support
            },
            "derived_from": sorted(learned),
            "limitation": (
                "derived from first-pass retrieval behaviour only: which queries returned nothing "
                "and which vocabulary the productive rounds used. No expected answer, known-positive "
                "set or other arm's response was read."
            ),
        }


def _result_vocabulary(calls: Sequence[Mapping[str, Any]], *, limit: int = 12) -> list[str]:
    counter: Counter = Counter()
    for call in calls:
        for result in call.get("results") or []:
            for token in tokens(str(result.get("title") or "")):
                if len(token) > 4:
                    counter[token] += 1
    return [term for term, _count in counter.most_common(limit)]
