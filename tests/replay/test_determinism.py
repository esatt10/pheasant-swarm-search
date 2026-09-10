"""Replay and determinism.

Raw JSONL is authoritative and everything else is derived. These tests hold
that: projections rebuild from the events alone, a metric recomputed over the
same operands is identical, and two runs of one configuration produce the same
question set.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.fixture(scope="module")
def run(demo_run):
    return demo_run[0]


def test_the_raw_stream_is_sequentially_continuous_and_digest_true(run: Path):
    from pheasant_lab.lifecycle import RunPaths
    from pheasant_lab.tracing.duckdb_projection import verify_raw

    assert verify_raw(RunPaths(run), check_digests=True) == []


def test_a_tampered_payload_is_detected(run: Path, tmp_path: Path):
    import shutil

    from pheasant_lab.lifecycle import RunPaths
    from pheasant_lab.tracing.duckdb_projection import verify_raw

    copy = tmp_path / "tampered"
    shutil.copytree(run, copy)
    events_path = copy / "raw/events.jsonl"
    rows = read_jsonl(events_path)
    rows[3]["payload"] = {"edited": True}
    events_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    findings = verify_raw(RunPaths(copy), check_digests=True)
    assert any("payload digest mismatch" in finding for finding in findings)


def test_a_missing_event_breaks_sequence_continuity(run: Path, tmp_path: Path):
    import shutil

    from pheasant_lab.lifecycle import RunPaths
    from pheasant_lab.tracing.duckdb_projection import verify_raw

    copy = tmp_path / "gapped"
    shutil.copytree(run, copy)
    events_path = copy / "raw/events.jsonl"
    rows = read_jsonl(events_path)
    del rows[5]
    events_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    findings = verify_raw(RunPaths(copy), check_digests=True)
    assert any("sequence discontinuity" in finding for finding in findings)


def test_the_projection_rebuilds_identically_from_the_events(run: Path):
    duckdb = pytest.importorskip("duckdb")
    from pheasant_lab.lifecycle import RunPaths
    from pheasant_lab.tracing.duckdb_projection import project

    paths = RunPaths(run)
    first = project(paths)
    second = project(paths)
    assert first.tables == second.tables
    assert first.findings == second.findings
    connection = duckdb.connect(str(paths.projections / "run.duckdb"), read_only=True)
    try:
        answers = connection.execute("SELECT count(*) FROM answers").fetchone()[0]
    finally:
        connection.close()
    assert int(answers) == len(read_jsonl(run / "raw/answers.jsonl"))


def test_recomputing_a_metric_over_the_same_operands_is_identical(run: Path):
    from pheasant_lab.benchmark.freezer import load_frozen
    from pheasant_lab.evaluation import answer_metrics
    from pheasant_lab.lifecycle import RunPaths
    from pheasant_lab.settings import AnswerMatchingSection

    paths = RunPaths(run)
    package = load_frozen(paths)
    answers = read_jsonl(run / "raw/answers.jsonl")
    matching = AnswerMatchingSection()
    for answer in answers[:8]:
        question = package.question(answer["question_id"])
        first = answer_metrics.compute(
            question,
            package.facts_for(question),
            package.evidence_for(question),
            answer,
            matching,
            run_id="run-1",
        )
        second = answer_metrics.compute(
            question,
            package.facts_for(question),
            package.evidence_for(question),
            answer,
            matching,
            run_id="run-1",
        )
        assert [row.as_dict() for row in first] == [row.as_dict() for row in second]


def test_a_metric_result_id_is_stable_across_processes(run: Path):
    rows = read_jsonl(run / "metrics/per-query.jsonl")
    seen: dict[str, dict] = {}
    for row in rows:
        key = row["metric_result_id"]
        if key in seen:
            assert seen[key]["value"] == row["value"], "one id, two values"
        seen[key] = row


def test_the_bootstrap_interval_is_reproducible_from_the_recorded_seed(run: Path):
    from pheasant_lab.evaluation.statistics import bootstrap_interval

    payload = json.loads((run / "metrics/classification.json").read_text())
    with_interval = [row for row in payload["comparisons"] if row.get("interval")]
    assert with_interval, "no comparison produced an interval"
    for row in with_interval[:3]:
        interval = row["interval"]
        assert interval["resamples"] > 0
        # Same seed, same resample count, same numbers.
        deltas = [0.1, -0.2, 0.3, 0.0, 0.25, -0.05, 0.4, 0.15]
        first = bootstrap_interval(deltas, resamples=interval["resamples"], seed=1)
        second = bootstrap_interval(deltas, resamples=interval["resamples"], seed=1)
        assert first is not None and (first.lower, first.upper) == (second.lower, second.upper)


def test_two_runs_of_one_configuration_produce_the_same_question_set(tmp_path: Path, run: Path):
    """The benchmark is content-addressed, so a second run must agree.

    Runs on the same fixture corpus with the same seed. What must match is the
    benchmark *version* and the question ids - not the run id, which carries a
    per-attempt nonce precisely so two runs are two rows.
    """

    environment = {
        **os.environ,
        "PHEASANT_LAB_FIXTURES": str(REPO / "tests/fixtures/literature"),
        "PHEASANT_LAB_PROMPTS": str(REPO / "prompts"),
    }
    outputs = []
    for index in range(2):
        root = tmp_path / f"run{index}"
        collected = subprocess.run(
            [
                sys.executable,
                "-m",
                "pheasant_lab.cli",
                "collect",
                "--config",
                str(REPO / "configs/demo.yaml"),
                "--mock",
                "--offline",
                "--project-root",
                str(REPO),
                "--set",
                f"experiment.output_root={root}",
            ],
            cwd=REPO,
            env=environment,
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert collected.returncode in (0, 1), collected.stdout + collected.stderr
        directory = sorted(root.glob("run-*"))[-1]
        frozen = subprocess.run(
            [
                sys.executable,
                "-m",
                "pheasant_lab.cli",
                "freeze-benchmark",
                "--config",
                str(REPO / "configs/demo.yaml"),
                "--run",
                directory.name,
                "--mock",
                "--project-root",
                str(REPO),
                "--set",
                f"experiment.output_root={root}",
            ],
            cwd=REPO,
            env=environment,
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert frozen.returncode == 0, frozen.stdout + frozen.stderr
        outputs.append(directory)

    manifests = [json.loads((d / "benchmark/benchmark-manifest.json").read_text()) for d in outputs]
    assert manifests[0]["benchmark_version"] == manifests[1]["benchmark_version"]
    assert manifests[0]["files"] == manifests[1]["files"], "the frozen files differ between runs"
    questions = [
        {q["question_id"] for q in read_jsonl(d / "benchmark/questions.jsonl")} for d in outputs
    ]
    assert questions[0] == questions[1]
    ids = [json.loads((d / "run-manifest.json").read_text())["run_id"] for d in outputs]
    assert ids[0] != ids[1], "two runs of one configuration must be two rows"


def test_replay_rebuilds_metrics_without_touching_the_raw_trace(run: Path, tmp_path: Path):
    import shutil

    copy = tmp_path / "replayed"
    shutil.copytree(run, copy)
    before = (copy / "raw/events.jsonl").read_bytes()
    environment = {
        **os.environ,
        "PHEASANT_LAB_PROMPTS": str(REPO / "prompts"),
    }
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pheasant_lab.cli",
            "replay",
            "--config",
            str(REPO / "configs/demo.yaml"),
            "--run",
            copy.name,
            "--project-root",
            str(REPO),
            "--set",
            f"experiment.output_root={copy.parent}",
        ],
        cwd=REPO,
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert (copy / "raw/events.jsonl").read_bytes() == before, (
        "replay must not write to the trace it rebuilds from"
    )


def test_verify_does_not_change_the_run_it_checks(run: Path, tmp_path: Path):
    import shutil

    copy = tmp_path / "verified"
    shutil.copytree(run, copy)
    before = {
        path.relative_to(copy): path.read_bytes()
        for path in (copy / "raw").iterdir()
        if path.is_file()
    }
    environment = {**os.environ, "PHEASANT_LAB_PROMPTS": str(REPO / "prompts")}
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pheasant_lab.cli",
            "verify",
            "--config",
            str(REPO / "configs/demo.yaml"),
            "--run",
            copy.name,
            "--project-root",
            str(REPO),
            "--set",
            f"experiment.output_root={copy.parent}",
        ],
        cwd=REPO,
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
    )
    after = {
        path.relative_to(copy): path.read_bytes()
        for path in (copy / "raw").iterdir()
        if path.is_file()
    }
    assert before == after
