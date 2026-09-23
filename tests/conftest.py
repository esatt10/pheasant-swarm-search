"""Shared fixtures.

The suite is **offline by design**: no test reaches the network, and the two
that would (the live-region smoke test and the provider contract tests) are
marked and deselected. That is not a convenience - a suite that sometimes
needs a network is a suite whose failures nobody can attribute.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_LITERATURE = REPO_ROOT / "tests" / "fixtures" / "literature"


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(autouse=True)
def _offline_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point every default at something local, and remove real credentials.

    A test that picks up an operator's real API key from the environment is a
    test that can bill them.
    """

    monkeypatch.setenv("PHEASANT_LAB_FIXTURES", str(FIXTURE_LITERATURE))
    monkeypatch.setenv("PHEASANT_LAB_PROMPTS", str(REPO_ROOT / "prompts"))
    for name in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "PHEASANT_MCP_TOKEN",
        "NCBI_API_KEY",
        "BRAVE_SEARCH_API_KEY",
        "TAVILY_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def config(tmp_path: Path):
    """The demo configuration, resolved against a temporary output root."""

    from pheasant_lab.settings import load_config

    return load_config(
        REPO_ROOT / "configs" / "demo.yaml",
        overrides={"experiment.output_root": str(tmp_path / "runs")},
        env_file=None,
        environ={},
        project_root=REPO_ROOT,
    )


@pytest.fixture
def mock_server():
    from pheasant_lab.pheasant.mock import MockPheasantServer

    return MockPheasantServer(knowledge_base="pheasant-lab")


@pytest.fixture
def connected_client(config, mock_server):
    """A client that has completed the handshake against the mock region."""

    from pheasant_lab.pheasant.capabilities import resolve
    from pheasant_lab.pheasant.client import PheasantClient

    client = PheasantClient.in_process(config.pheasant, mock_server)
    session = client.connect()
    return client, resolve(config.pheasant, session.tools)


@pytest.fixture
def tracer(tmp_path: Path):
    from pheasant_lab.lifecycle import RunPaths
    from pheasant_lab.redaction import Redactor
    from pheasant_lab.tracing.events import Tracer

    paths = RunPaths(tmp_path / "run-test").ensure()
    tracer = Tracer(
        paths, run_id="run-test", config_digest="sha256:test", redactor=Redactor(enabled=True)
    )
    yield tracer
    tracer.close()


@pytest.fixture(scope="session")
def demo_run(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Path, str]]:
    """One complete offline run, shared by the integration and replay tests.

    Session-scoped because it is the expensive fixture in this suite and every
    test that reads it is read-only. It runs the *CLI*, not the internals, so
    the tests exercise the path a person uses.
    """

    root = tmp_path_factory.mktemp("demo")
    environment = {
        **os.environ,
        "PHEASANT_LAB_FIXTURES": str(FIXTURE_LITERATURE),
        "PHEASANT_LAB_PROMPTS": str(REPO_ROOT / "prompts"),
    }
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pheasant_lab.cli",
            "demo",
            "--config",
            str(REPO_ROOT / "configs" / "demo.yaml"),
            "--output-root",
            str(root / "runs"),
            "--project-root",
            str(REPO_ROOT),
        ],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
    )
    runs = sorted((root / "runs").glob("run-*"))
    if not runs:
        pytest.fail(
            f"demo produced no run\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    yield runs[-1], completed.stdout
    shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
def question():
    """One question, with a matcher whose terms are not in its wording."""

    from pheasant_lab.benchmark.question_types import ExpectedFact, FactMatcher, Question

    q = Question(
        question_id="q-test",
        topic_id="topic-test",
        text="What does the literature establish about chromatin shielding?",
        type="atomic_fact",
        required_fact_ids=["fact-test"],
        acceptable_evidence_source_ids=["source-a"],
        known_negative_source_ids=["source-z"],
        cohorts=["anchor"],
    )
    fact = ExpectedFact(
        fact_id="fact-test",
        text="Dsup reduced double-strand breaks by 42 percent",
        matcher=FactMatcher(kind="all_of", groups=[["dsup"], ["hydroxyl", "radical"]]),
        source_ids=["source-a"],
        claim_ids=["claim-a"],
    )
    return q, fact
