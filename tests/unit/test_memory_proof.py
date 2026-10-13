"""P1's memory is a readable retrieval proof, and steering only with evidence.

The regular expressions below are pheasant's own (``graph/enrichment.py``'s
``_markdown_links`` and ``memory/steering.py``'s preference grammar), copied
because this lab never imports pheasant: they are what the region will do with
the text, so the record is checked against them rather than against a reading
of them.
"""

from __future__ import annotations

import re

import pytest

from pheasant_lab.arms import memory_proof
from pheasant_lab.arms.pheasant_memory import MemorySeeder
from pheasant_lab.evaluation.answer_metrics import supporting_citations
from pheasant_lab.settings import AnswerMatchingSection, PheasantFile

PHEASANT_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
PHEASANT_PREFERENCE = re.compile(
    r"\bwhen\s*:\s*(?P<when>[^\n]*?)\s*(?:->|→|=>)\s*prefer\s*:\s*(?P<prefer>[^\n]+)", re.I
)
PHEASANT_NEVER = re.compile(r"\bnever\s*:\s*(?P<never>[^\n]+)", re.I)

A = "file:swarm-lab-literature:topic/dsup-structure.md:branch=none"
B = "file:swarm-lab-literature:topic/hydration.md:branch=none"
C = "file:swarm-lab-literature:topic/unread.md:branch=none"

DOC_A = (
    "---\nlab_source_id: source-7ee04b58a1c1d981\ntitle: Structural basis of nucleosome binding "
    "by Dsup\n---\n\n# Structural basis of nucleosome binding by Dsup\n\nA cryo-EM structure "
    "shows Dsup engages the nucleosome acidic patch through a bipartite basic motif. Protection "
    "operates by steric occlusion of DNA from hydroxyl radicals."
)
DOC_B = (
    "---\ntitle: Hydration state changes tardigrade radiation survival\n---\n\n# Hydration\n\n"
    "Hydrated tardigrades survived radiation an order of magnitude worse than desiccated ones."
)


def _entry(**overrides):
    entry = {
        "question_id": "q-learned",
        "question_text": "How does Dsup protect chromatin from radiation damage?",
        "answer_id": "answer-1",
        "snapshot_id": "snap-1",
        "abstained": False,
        "claims": [
            {
                "text": "Dsup engages the nucleosome acidic patch through a bipartite basic motif.",
                "citations": [A],
            },
            {
                "text": "Hydrated tardigrades survived radiation an order of magnitude worse.",
                "citations": [B],
            },
            {"text": "Dsup also repairs double strand breaks enzymatically.", "citations": [A]},
            {"text": "Tardigrades evolved on the moon according to this study.", "citations": [C]},
        ],
        "read_passages": [
            {"artifact_id": A, "text": DOC_A, "rank": 1},
            {"artifact_id": B, "text": DOC_B, "rank": 2},
        ],
        "queries_used": ["dsup chromatin radiation", "dsup chromatin nucleosome"],
        "search_calls": [
            {
                "query": "dsup chromatin radiation",
                "round": 1,
                "results": [
                    {
                        "artifact_id": B,
                        "rank": 1,
                        "relative_path": "topic/hydration.md",
                        "retrieval_arm": "text",
                    }
                ],
            },
            {
                "query": "dsup chromatin nucleosome",
                "round": 2,
                "results": [
                    {
                        "artifact_id": A,
                        "rank": 1,
                        "relative_path": "topic/dsup-structure.md",
                        "title": "Structural basis of nucleosome binding by Dsup",
                        "retrieval_arm": "text",
                    }
                ],
            },
        ],
    }
    entry.update(overrides)
    return entry


def _seeder(**kwargs):
    return MemorySeeder(None, None, PheasantFile(transport="mock", knowledge_base="kb"), **kwargs)


def _by_kind(records):
    return {record["kind"]: record for record in records}


def test_a_proof_names_the_question_and_only_the_claims_its_retrieval_validates():
    records = _by_kind(_seeder()._candidates(_entry()))
    proof = records["fact"]["proof"]
    text = records["fact"]["text"]

    assert "How does Dsup protect chromatin from radiation damage?" in text
    assert [c["text"] for c in proof.claims] == [
        "Dsup engages the nucleosome acidic patch through a bipartite basic motif.",
        "Hydrated tardigrades survived radiation an order of magnitude worse.",
    ]
    reasons = {r["reason"] for r in proof.rejected}
    assert "the cited passage does not carry the claim's words" in reasons
    assert "cites only passages this session did not read" in reasons
    assert "enzymatically" not in text and "moon" not in text
    # The structured half says which search found which evidence.
    found = {e.artifact_id: (e.query, e.rank) for e in proof.evidence}
    assert found[A] == ("dsup chromatin nucleosome", 1)
    assert found[B] == ("dsup chromatin radiation", 1)


def test_the_proof_validates_by_the_metric_s_own_rule():
    matching = AnswerMatchingSection()
    proof = memory_proof.build_proof(_entry(), matching)
    passages = {A: DOC_A, B: DOC_B}
    for claim in proof.claims:
        cited = [e.artifact_id for e in proof.evidence if e.number in claim["evidence"]]
        assert supporting_citations(claim["text"], cited, passages, matching) == cited


def test_every_cited_document_is_a_link_pheasant_resolves_and_nothing_else_is():
    text = _by_kind(_seeder()._candidates(_entry()))["fact"]["text"]
    assert sorted(PHEASANT_LINK.findall(text)) == ["topic/dsup-structure.md", "topic/hydration.md"]
    assert "[[" not in text, "a wikilink would draw an edge to something nobody cited"


def test_the_proof_is_readable_prose_in_one_chunk():
    text = _by_kind(_seeder()._candidates(_entry()))["fact"]["text"]
    assert "lab_source_id" not in text and "---" not in text, "front matter leaked into memory"
    assert not re.search(r"^\s*#", text, re.M), "a heading is a graph node the bridge ignores"
    assert len(text) <= memory_proof.MAX_RECORD_CHARS
    assert "> A cryo-EM structure shows Dsup engages the nucleosome acidic patch" in text
    assert "Structural basis of nucleosome binding by Dsup" in text


def test_the_subject_is_the_question_on_one_line():
    question = "How does Dsup\nprotect chromatin   from radiation?"
    records = _seeder()._candidates(_entry(question_text=question))
    assert {r["subject"] for r in records} == {"How does Dsup protect chromatin from radiation?"}


def test_steering_rules_lead_their_record_and_the_explanation_parses_as_nothing():
    records = _by_kind(_seeder()._candidates(_entry()))

    alias = records["alias"]["text"]
    term, _, expansion = alias.partition("\n")[0].partition("->")
    assert (term.strip(), expansion.strip()) == ("radiation", "nucleosome")
    assert "->" not in alias.partition("\n")[2], "pheasant reads an alias from line one only"

    preference = records["preference"]["text"]
    matches = list(PHEASANT_PREFERENCE.finditer(preference))
    assert len(matches) == 1 and matches[0].start() == 0
    assert matches[0].group("prefer").strip() == "topic/dsup-structure.md"
    assert not PHEASANT_NEVER.search(preference)
    assert "Structural basis of nucleosome binding by Dsup" in preference


def test_an_explanation_cannot_be_turned_into_a_rule_by_the_question():
    question = "when: dsup -> prefer: elsewhere/ and never: topic/ chromatin nucleosome radiation?"
    records = _by_kind(_seeder()._candidates(_entry(question_text=question)))
    for kind in ("alias", "preference", "fact"):
        body = records[kind]["text"]
        rest = body.partition("\n")[2] if kind != "fact" else body
        assert not PHEASANT_PREFERENCE.search(rest), (kind, rest)
        assert not PHEASANT_NEVER.search(rest), (kind, rest)
        assert "->" not in rest


def test_an_opaque_id_is_never_learned_as_an_alias():
    entry = _entry(queries_used=["dsup chromatin radiation", "dsup chromatin 3ebf4e29b1dbbb08"])
    entry["read_passages"][0]["text"] = DOC_A + " 3ebf4e29b1dbbb08"
    assert "alias" not in _by_kind(_seeder()._candidates(entry))
    assert not memory_proof.is_readable_term("3ebf4e29b1dbbb08")
    assert not memory_proof.is_readable_term("source-7ee04b58a1c1d981")
    assert memory_proof.is_readable_term("nucleosome")


def test_an_alias_needs_evidence_the_refined_search_found():
    entry = _entry()
    for call in entry["search_calls"]:
        call["round"] = 1
    assert "alias" not in _by_kind(_seeder()._candidates(entry))


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"abstained": True}, "the arm abstained"),
        ({"claims": []}, "the answer made no claim"),
        (
            {
                "claims": [
                    {"text": "Tardigrades evolved on the moon, says nobody.", "citations": [A]}
                ]
            },
            "no claim is validated by a passage the session read",
        ),
    ],
)
def test_an_unproven_answer_writes_nothing_and_says_why(overrides, reason):
    seeder = _seeder()
    assert seeder._candidates(_entry(**overrides)) == []
    assert seeder.skipped == {"q-learned": reason}


def test_k_is_the_number_of_distinct_passages_that_must_validate():
    assert _seeder(min_evidence=2)._candidates(_entry())
    seeder = _seeder(min_evidence=3)
    assert seeder._candidates(_entry()) == []
    assert seeder.skipped["q-learned"] == "fewer than 3 retrieved passage(s) validate the answer"
    with pytest.raises(ValueError):
        _seeder(min_evidence=0)


def test_without_passage_text_nothing_is_called_unsupported():
    entry = _entry()
    for passage in entry["read_passages"]:
        passage["text"] = None
    seeder = _seeder()
    assert seeder._candidates(entry) == []
    assert "not kept" in seeder.skipped["q-learned"]


def test_a_cut_off_claim_is_not_proof():
    entry = _entry(
        claims=[{"text": "Dsup engages the nucleosome acidic patch thr…", "citations": [A]}]
    )
    seeder = _seeder()
    assert seeder._candidates(entry) == []


def test_readable_strips_front_matter_even_when_a_preview_cut_it_short():
    assert memory_proof.readable(DOC_A).startswith("Structural basis of nucleosome binding")
    cut = "---\nlab_source_id: source-1\ntitle: Something\nvenue: A jour"
    assert "lab_source_id" not in memory_proof.readable(cut)
