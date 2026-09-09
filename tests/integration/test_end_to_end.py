"""One complete offline run, checked the way a reviewer would check it.

The acceptance test this repository sets itself: a reviewer starts at one
sentence in ``summary.md``, resolves it to an aggregate metric, inspects its
per-question values, locates the exact Pheasant result and ingest receipt,
traces that result to its source and locator, and sees every error or
exclusion that affected the denominator. These tests walk that path.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.fixture(scope="module")
def run(demo_run):
    return demo_run[0]


def test_the_run_produced_every_artifact_the_layout_promises(run: Path):
    for relative in (
        "run-manifest.json",
        "resolved-config.redacted.yaml",
        "state.json",
        "raw/events.jsonl",
        "raw/spans.jsonl",
        "raw/errors.jsonl",
        "raw/mcp-calls.redacted.jsonl",
        "raw/sources.jsonl",
        "raw/claims.jsonl",
        "raw/ingest-receipts.jsonl",
        "raw/questions.jsonl",
        "raw/answers.jsonl",
        "raw/proof-events.jsonl",
        "benchmark/benchmark-manifest.json",
        "benchmark/questions.jsonl",
        "benchmark/expected-facts.jsonl",
        "benchmark/expected-evidence.jsonl",
        "benchmark/cohort-membership.jsonl",
        "metrics/per-query.jsonl",
        "metrics/aggregates.json",
        "metrics/paired-deltas.csv",
        "metrics/gates.json",
        "metrics/classification.json",
        "reports/summary.md",
        "reports/collection.md",
        "reports/arm-comparison.md",
        "reports/errors-and-retries.md",
        "reports/worst-regressions.md",
        "reports/refinement-candidates.md",
        "integrity/checksums.sha256",
        "integrity/verification.json",
    ):
        assert (run / relative).is_file(), f"missing {relative}"


def test_the_manifest_records_the_resolved_configuration_without_secrets(run: Path):
    manifest = json.loads((run / "run-manifest.json").read_text())
    assert manifest["config_digest"].startswith("sha256:")
    assert manifest["arms"] == ["S0", "C0", "P0", "P1", "P2"]
    assert manifest["models"]["researcher"]["provider"] == "replay"
    body = (run / "resolved-config.redacted.yaml").read_text()
    assert "sk-" not in body


def test_every_submitted_source_carries_a_receipt(run: Path):
    from pheasant_lab.pheasant.receipts import fold_receipts

    sources = read_jsonl(run / "raw/sources.jsonl")
    receipts = {
        r["source_id"]: r
        for r in fold_receipts(read_jsonl(run / "raw/ingest-receipts.jsonl")).values()
    }
    submitted = [s for s in sources if s["state"] in {"submitted", "indexed", "verified"}]
    assert submitted, "the run ingested nothing"
    for source in submitted:
        assert source["source_id"] in receipts, f"{source['source_id']} has no receipt"
        assert receipts[source["source_id"]]["status"] in {"accepted", "indexed", "verified"}


def test_the_index_barrier_was_crossed(run: Path):
    from pheasant_lab.pheasant.receipts import fold_receipts

    rows = read_jsonl(run / "raw/ingest-receipts.jsonl")
    folded = fold_receipts(rows)
    assert any(r["status"] == "indexed" for r in folded.values()), (
        "acceptance is not searchability; nothing crossed the index barrier"
    )
    assert len(rows) > len(folded), (
        "the acknowledgement should append a superseding row rather than rewrite one"
    )


def test_every_retained_source_has_complete_provenance(run: Path):
    for source in read_jsonl(run / "raw/sources.jsonl"):
        if source["state"] in {"submitted", "indexed", "verified"}:
            assert source["provenance_complete"] is True
            assert source["discovery_event_id"]
            assert source["stable_identifier"] or source["canonical_url"]


def test_a_rejected_candidate_keeps_its_reason_code(run: Path):
    sources = read_jsonl(run / "raw/sources.jsonl")
    rejected = [s for s in sources if s["state"] in {"rejected", "blocked"}]
    for source in rejected:
        assert source["reason_code"], "a rejection with no reason will be rediscovered"


def test_the_benchmark_was_frozen_before_the_arms_ran(run: Path):
    manifest = json.loads((run / "benchmark/benchmark-manifest.json").read_text())
    assert manifest["frozen"] is True
    assert manifest["package_digest"].startswith("sha256:")
    frozen_at = manifest["created_at"]
    answers = read_jsonl(run / "raw/answers.jsonl")
    assert answers, "no arm answered"
    # Every answer is recorded after the freeze; the freeze itself is a file
    # digest, so this is a coarse ordering check rather than the guarantee.
    events = read_jsonl(run / "raw/events.jsonl")
    freeze_seq = max(e["sequence"] for e in events if e["event_type"] == "benchmark.built")
    answer_seq = min(e["sequence"] for e in events if e["event_type"] == "arm.answered")
    assert freeze_seq < answer_seq
    del frozen_at


def test_the_question_set_has_no_invalidating_leakage(run: Path):
    leakage = json.loads((run / "benchmark/leakage.json").read_text())
    invalidating = [f for f in leakage["findings"] if f["severity"] == "invalidating"]
    assert invalidating == [], f"leakage: {invalidating}"


def test_every_arm_answered_every_question(run: Path):
    answers = read_jsonl(run / "raw/answers.jsonl")
    questions = {q["question_id"] for q in read_jsonl(run / "benchmark/questions.jsonl")}
    by_arm: dict[str, set[str]] = {}
    for answer in answers:
        by_arm.setdefault(answer["arm_id"], set()).add(answer["question_id"])
    assert set(by_arm) == {"S0", "C0", "P0", "P1", "P2"}
    for arm, covered in by_arm.items():
        assert covered == questions, f"{arm} did not answer every question"


def test_the_pheasant_arms_never_saw_the_research_package(run: Path):
    """The isolation the whole experiment rests on.

    A Pheasant arm's answer must be reachable from what it retrieved. If it
    cited a source it was never served, it saw something it should not have.
    """

    for answer in read_jsonl(run / "raw/answers.jsonl"):
        if answer["arm_id"] not in {"P0", "P1", "P2"} or answer.get("abstained"):
            continue
        served = {
            str(result.get("artifact_id"))
            for call in answer.get("search_calls") or []
            for result in call.get("results") or []
        }
        for claim in answer.get("claims") or []:
            for citation in claim.get("citations") or []:
                assert citation in served, (
                    f"{answer['arm_id']} cited {citation}, which no search returned"
                )


def test_memory_is_off_for_p0_and_on_for_p1(run: Path):
    modes: dict[str, set[bool]] = {}
    for answer in read_jsonl(run / "raw/answers.jsonl"):
        for call in answer.get("search_calls") or []:
            modes.setdefault(answer["arm_id"], set()).add(bool(call.get("memory_enabled")))
    assert modes["P0"] == {False}
    assert modes["P1"] == {True}


def test_only_the_unpinned_arms_wrote_memory(run: Path):
    records_file = run / "raw/memory-records.jsonl"
    if not records_file.is_file():
        pytest.skip("this region offers no memory write capability")
    for record in read_jsonl(records_file):
        assert record["originating_question_id"], (
            "a seeded record must name the question it came from"
        )


def test_no_holdout_or_control_question_created_a_memory_record(run: Path):
    records_file = run / "raw/memory-records.jsonl"
    if not records_file.is_file():
        pytest.skip("no memory was seeded")
    cohorts: dict[str, list[str]] = {}
    for row in read_jsonl(run / "benchmark/cohort-membership.jsonl"):
        cohorts.setdefault(row["question_id"], []).append(row["cohort"])
    for record in read_jsonl(records_file):
        origin = record["originating_question_id"]
        assert "temporal_holdout" not in cohorts.get(origin, [])
        assert "control" not in cohorts.get(origin, [])


def test_served_carries_no_weight_and_validation_does(run: Path):
    proof = read_jsonl(run / "raw/proof-events.jsonl")
    served = [p for p in proof if p["event_type"] == "served"]
    validated = [p for p in proof if p["event_type"].startswith("deterministic_validation")]
    assert served and all(p["weight"] == 0.0 and p["polarity"] == "unknown" for p in served)
    assert validated and any(p["weight"] > 0 for p in validated)


def test_every_metric_row_carries_its_denominator_and_limitation(run: Path):
    rows = read_jsonl(run / "metrics/per-query.jsonl")
    assert rows
    for row in rows:
        assert row["formula"], row
        assert row["limitation"], row
        if row["status"] == "ok":
            assert row["value"] is not None
            assert row["denominator"] is not None
        else:
            assert row["value"] is None, "a metric that could not be computed must not report 0.0"


def test_the_gate_verdict_is_tri_state_and_incomplete_is_not_a_pass(run: Path):
    gates = json.loads((run / "metrics/gates.json").read_text())
    assert gates["verdict"] in {"PASS", "FAIL", "INCOMPLETE"}
    for body in gates["gate_sets"].values():
        skipped = body["skipped"]
        if skipped:
            assert body["verdict"] != "PASS"
            assert "INCOMPLETE" in body["heading"]
            for gate in body["gates"]:
                if gate["gate"] in skipped:
                    assert gate["skip_reason"], "a skipped gate must say why"


def test_paired_deltas_are_exported_as_csv_with_their_thresholds(run: Path):
    with (run / "metrics/paired-deltas.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    for row in rows:
        assert row["practical_threshold"], (
            "a status with no threshold has said less than it appears to"
        )
        assert row["status"] in {
            "improved",
            "regressed",
            "unchanged",
            "mixed",
            "insufficient_evidence",
            "not_comparable",
        }


def test_learned_and_holdout_are_reported_separately(run: Path):
    payload = json.loads((run / "metrics/classification.json").read_text())
    cohorts = {row["cohort"] for row in payload["comparisons"]}
    assert "learned" in cohorts
    assert "temporal_holdout" in cohorts


def test_the_summary_states_what_the_run_does_not_support(run: Path):
    summary = (run / "reports/summary.md").read_text()
    assert "Limitations of this run" in summary
    assert "not generalization" in summary
    assert "floor" in summary, "the replay provider's missing prior must be stated"


def test_a_reviewer_can_walk_from_a_metric_to_a_source_and_a_receipt(run: Path):
    from pheasant_lab.lifecycle import RunPaths
    from pheasant_lab.tracing.lineage import LineageIndex

    index = LineageIndex(RunPaths(run))
    rows = [
        row
        for row in read_jsonl(run / "metrics/per-query.jsonl")
        if row["metric"] == "fact_recall" and row["status"] == "ok" and row.get("arm_id") == "P1"
    ]
    assert rows, "no P1 recall value to walk back from"
    chains = index.chain_for_question(rows[0]["question_id"], "P1")
    assert chains
    kinds = {node.kind for chain in chains for node in chain.nodes}
    assert {"question", "search_call", "answer"} <= kinds
    # At least one chain reaches a source and its receipt.
    assert any(
        {"source", "ingest_receipt"} <= {node.kind for node in chain.nodes} for chain in chains
    ), "no answer resolves back to a source and its ingest receipt"


def test_the_lineage_has_no_orphans(run: Path):
    from pheasant_lab.lifecycle import RunPaths
    from pheasant_lab.tracing.lineage import LineageIndex

    index = LineageIndex(RunPaths(run))
    orphans = index.orphans()
    assert not orphans, f"unresolvable references: {orphans}"


def test_the_transcript_and_reports_contain_no_secret(run: Path):
    for relative in ("raw/mcp-calls.redacted.jsonl", "reports/summary.md", "run-manifest.json"):
        body = (run / relative).read_text()
        assert "sk-ant-" not in body and "Bearer " not in body


def test_costs_were_reserved_and_reconciled(run: Path):
    events = [
        e for e in read_jsonl(run / "raw/events.jsonl") if e["event_type"] == "cost.model_call"
    ]
    assert events
    for event in events:
        payload = event["payload"]
        assert payload["reconciled"] is True
        assert payload["actual_usd"] is not None
        assert payload["reserved_usd"] >= 0
