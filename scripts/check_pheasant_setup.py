#!/usr/bin/env python3
"""Exercise a real local Pheasant with fixtures and replay models, without API keys.

Start it first: docker compose -f deploy/stub/compose.yaml up -d --wait
Then run: uv run python scripts/check_pheasant_setup.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import httpx

from pheasant_lab.pheasant.receipts import fold_receipts
from pheasant_lab.settings import load_config

ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> int:
    config = load_config(ROOT / "configs/stub.yaml", env_file=None, environ={}, project_root=ROOT)
    if any(model.provider != "replay" for model in config.models.values()):
        raise RuntimeError("Setup check requires replay for every model role.")
    if config.collection.providers != ["fixtures"]:
        raise RuntimeError("Setup check requires fixture literature.")
    if config.pheasant.transport != "streamable_http":
        raise RuntimeError("Setup check requires the real HTTP service, not the mock.")

    output = ROOT / "runs" / f"setup-check-{uuid.uuid4().hex[:12]}"
    output.mkdir(parents=True)
    print(f"Setup evidence and logs: {output}", flush=True)
    endpoint = config.pheasant.url.removesuffix("/").removesuffix("/mcp")
    health = {}
    with httpx.Client(timeout=30) as client:
        for path in ("health", "ready", "jobs"):
            response = client.get(f"{endpoint}/{path}")
            response.raise_for_status()
            health[path] = response.json()
    (output / "service.json").write_text(json.dumps(health, indent=2), encoding="utf-8")
    print("Pheasant health, readiness, and jobs: OK", flush=True)

    environment = {**os.environ, "PYTHONUTF8": "1"}
    # Keep the documented fixture and prompt set, even in an operator's shell.
    environment["PHEASANT_LAB_FIXTURES"] = str(ROOT / "tests/fixtures/literature")
    environment["PHEASANT_LAB_PROMPTS"] = str(ROOT / "prompts")
    common = [
        "--config",
        "configs/stub.yaml",
        "--env-file",
        str(output / "unused.env"),
        "--set",
        f"experiment.output_root={output / 'runs'}",
    ]

    def step(command: str, *extra: str, accepted: tuple[int, ...] = (0,)) -> None:
        log = output / f"{command}.log"
        print(f"Running {command} ...", flush=True)
        with log.open("w", encoding="utf-8") as stream:
            result = subprocess.run(
                [sys.executable, "-m", "pheasant_lab.cli", command, *common, *extra],
                cwd=ROOT,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=600,
            )
        if result.returncode not in accepted:
            raise RuntimeError(f"{command} exited {result.returncode}; read {log}")

    step("doctor")
    step("plan")
    before = set((output / "runs").glob("run-*"))
    step("collect")
    created = set((output / "runs").glob("run-*")) - before
    if len(created) != 1:
        raise RuntimeError(f"Expected one collection run, found {len(created)}.")
    run = created.pop()
    receipts = list(fold_receipts(rows(run / "raw/ingest-receipts.jsonl")).values())
    if not receipts or any(r["status"] not in {"indexed", "verified"} for r in receipts):
        raise RuntimeError("Every submitted fixture must cross the indexing barrier.")
    if any(r["digest_matches"] is not True for r in receipts):
        raise RuntimeError("Pheasant must confirm each submitted document's content digest.")
    print(f"Stored, indexed, and digest-checked {len(receipts)} documents.", flush=True)
    for command in ("audit", "freeze-benchmark", "evaluate", "replay", "report"):
        step(command, "--run", run.name)
    step("verify", "--run", run.name, accepted=(0, 1))

    verification = read_json(run / "integrity/verification.json")
    structural = [f for f in verification["findings"] if not f.startswith("hard gates:")]
    if structural:
        raise RuntimeError(f"Run integrity failed: {structural}")
    gates = read_json(run / "metrics/gates.json")
    if gates["gate_sets"]["core"]["verdict"] != "PASS":
        raise RuntimeError("Core ingestion, leakage, and snapshot checks must pass.")
    answers = rows(run / "raw/answers.jsonl")
    questions = rows(run / "benchmark/questions.jsonl")
    if len(answers) != len(questions) * len(config.arms):
        raise RuntimeError("Each arm must answer every fixture question.")
    if any(a.get("error") or a["status"] == "failed" for a in answers):
        raise RuntimeError("An answering arm encountered an execution error.")
    source_ids = {r["source_id"] for r in receipts}
    for arm in ("P0", "P1", "P2"):
        hits = [
            hit
            for answer in answers
            if answer["arm_id"] == arm
            for call in answer["search_calls"]
            for hit in call["results"]
        ]
        if not any(h.get("matched_text") and h.get("source_id") in source_ids for h in hits):
            raise RuntimeError(f"{arm} did not retrieve source-linked fixture evidence.")
    state = read_json(run / "state.json")
    if state["budget"]["committed_usd"] != 0:
        raise RuntimeError("A replay setup check must record zero model cost.")
    result = {
        "setup": "PASS",
        "run": run.name,
        "indexed_documents": len(receipts),
        "answers": len(answers),
        "model_cost_usd": 0,
        "experiment_gates": gates["verdict"],
        "limitation": "Synthetic fixture literature and replay models test setup, not model quality.",
        "report": str(run / "reports/summary.md"),
    }
    (output / "setup-check.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, httpx.HTTPError, subprocess.TimeoutExpired) as exc:
        print(f"Setup check failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
