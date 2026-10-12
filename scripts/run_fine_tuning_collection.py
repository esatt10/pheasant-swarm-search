"""Collect each fine-tuning research topic into one Pheasant source, in order.

This is a job launcher, not a second research runtime. Each topic is a normal
`pheasant-lab collect` child process with its own run directory and budget.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from pheasant_lab.settings import load_config, load_dotenv

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/experiment.fine-tuning.yaml"
JOB = ROOT / "runs" / ".fine-tuning-collection"
STATUS = JOB / "status.json"
RUN_RE = re.compile(r"run: (run-[a-f0-9]+)")


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def save(status: dict) -> None:
    JOB.mkdir(parents=True, exist_ok=True)
    temporary = STATUS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    temporary.replace(STATUS)


def main() -> int:
    env = os.environ.copy()
    secrets = load_dotenv(ROOT / ".env")
    for key in ("OPENAI_API_KEY", "PHEASANT_API_TOKEN"):
        if secrets.get(key):
            env[key] = secrets[key]
        if not env.get(key):
            raise RuntimeError(f"{key} is required for the live collection")

    # Some constrained shells install a deliberately dead proxy. This job is
    # launched with direct network access; preserve any real proxy setting.
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        if env.get(key) == "http://127.0.0.1:9":
            env.pop(key)

    config = load_config(CONFIG, project_root=ROOT, env_file=".env")
    if config.pheasant.source_name != "swarm-lab-fine-tuning":
        raise RuntimeError("the collection config points at the wrong source")

    topics = [topic.id for topic in config.topics]
    status: dict = {"started_at": stamp(), "source": config.pheasant.source_name, "topics": {}}
    save(status)

    for topic_id in topics:
        log_path = JOB / f"{topic_id}.log"
        record = {"status": "running", "started_at": stamp(), "log": str(log_path)}
        status["topics"][topic_id] = record
        save(status)
        command = [
            sys.executable,
            "-m",
            "pheasant_lab.cli",
            "collect",
            "--config",
            CONFIG,
            "--topic",
            topic_id,
        ]
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            record["pid"] = process.pid
            save(status)
            code = process.wait()

        output = log_path.read_text(encoding="utf-8", errors="replace")
        run_ids = RUN_RE.findall(output)
        if run_ids:
            record["run_id"] = run_ids[-1]
        record["exit_code"] = code
        record["finished_at"] = stamp()
        run_id = record.get("run_id")
        if run_id:
            state_path = ROOT / "runs" / run_id / "state.json"
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
                decision = (state.get("stages") or {}).get("collect") or {}
            except (OSError, ValueError):
                decision = {}
            record["decision"] = decision.get("decision")
            completed = decision.get("status") == "completed"
        else:
            completed = False
        record["status"] = "completed" if completed else "stopped"
        save(status)
        if not completed:
            return code or 1

    status["finished_at"] = stamp()
    save(status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
