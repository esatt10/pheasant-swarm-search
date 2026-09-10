"""Canonicalisation and digest stability.

An id that moves when nothing it names has changed makes every consumer
downstream believe the region changed shape. These tests exist to make that a
red build rather than a report nobody can reproduce.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from pheasant_lab import ids
from pheasant_lab.hashing import canonical_json, digest, digest_text, short


def test_canonical_json_is_key_order_independent():
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})


def test_digest_is_stable_across_equivalent_spellings():
    assert digest({"n": 1.0}) == digest({"n": 1})
    assert digest([1, 2]) != digest([2, 1])


def test_datetimes_normalise_to_utc():
    aware = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    naive = datetime(2026, 1, 1, 12, 0)
    assert digest(aware) == digest(naive)
    assert "2026-01-01" in canonical_json(date(2026, 1, 1))


def test_non_finite_floats_are_refused():
    with pytest.raises(ValueError, match="non-finite"):
        canonical_json({"x": float("nan")})


def test_unrepresentable_types_are_refused_rather_than_stringified():
    class Opaque:
        pass

    with pytest.raises(TypeError, match="no canonical spelling"):
        canonical_json({"x": Opaque()})


def test_question_id_is_a_function_of_text_and_version_only():
    first = ids.question_id("What is Dsup?", "bench-1")
    second = ids.question_id("what is   dsup?", "bench-1")
    assert first == second, "normalisation must fold whitespace and case"
    assert first != ids.question_id("What is Dsup?", "bench-2")


def test_run_id_separates_two_runs_of_one_configuration():
    first = ids.run_id("exp", "sha256:cfg", ids.new_nonce())
    second = ids.run_id("exp", "sha256:cfg", ids.new_nonce())
    assert first != second


def test_source_id_prefers_the_stable_identifier():
    with_doi = ids.source_id("10.1234/abc", "https://example.org/a", digest_text("body"))
    same_doi_other_url = ids.source_id("10.1234/abc", "https://example.org/b", digest_text("other"))
    assert with_doi == same_doi_other_url


def test_source_id_falls_back_to_url_plus_content():
    a = ids.source_id(None, "https://example.org/a", digest_text("body"))
    b = ids.source_id(None, "https://example.org/a", digest_text("edited"))
    assert a != b


def test_idempotency_key_is_content_addressed():
    key = ids.idempotency_key("ns", "source-1", digest_text("body"))
    assert key == ids.idempotency_key("ns", "source-1", digest_text("body"))
    assert key != ids.idempotency_key("ns", "source-1", digest_text("body "))


def test_short_strips_the_algorithm_prefix():
    assert not short(digest("x")).startswith("sha256")
