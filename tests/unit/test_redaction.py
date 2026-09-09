"""Secret handling."""

from __future__ import annotations

from pheasant_lab.redaction import Redactor, hash_principal


def test_a_registered_secret_is_replaced_by_a_stable_token():
    redactor = Redactor()
    redactor.register("OPENAI_API_KEY", "sk-abcdefghijklmnop")
    first = redactor.text("bearer sk-abcdefghijklmnop here")
    second = redactor.text("and again sk-abcdefghijklmnop")
    assert "sk-abcdefghijklmnop" not in first
    token = first.split("bearer ")[1].split(" here")[0]
    assert token in second, "the token must be stable so a reader can correlate two calls"


def test_forbidden_headers_are_dropped_rather_than_masked():
    redactor = Redactor()
    headers = redactor.headers({"authorization": "Bearer abc", "accept": "application/json"})
    assert "authorization" not in headers
    assert headers["accept"] == "application/json"


def test_inline_key_shapes_are_caught_without_registration():
    redactor = Redactor()
    assert "sk-" not in redactor.text("leaked sk-0123456789abcdefghij in a log line")


def test_nested_payloads_are_scrubbed():
    redactor = Redactor()
    redactor.register("TOKEN", "supersecretvalue")
    payload = redactor.payload({"a": ["supersecretvalue"], "headers": {"authorization": "x"}})
    assert "supersecretvalue" not in str(payload["a"])
    assert payload["headers"]["authorization"] == "[redacted:header]"


def test_short_values_are_not_registered():
    redactor = Redactor()
    redactor.register("SHORT", "abc")
    assert redactor.known == 0, "registering 'abc' would redact the letter sequence everywhere"


def test_disabled_redactor_is_a_pass_through():
    redactor = Redactor(enabled=False)
    redactor.register("TOKEN", "supersecretvalue")
    assert redactor.text("supersecretvalue") == "supersecretvalue"


def test_principal_hashing_is_stable_within_a_salt():
    assert hash_principal("alice", "salt") == hash_principal("alice", "salt")
    assert hash_principal("alice", "salt") != hash_principal("alice", "other")
    assert hash_principal(None, "salt") is None
