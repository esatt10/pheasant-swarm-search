"""Run directories, the run manifest, and the resumable checkpoint.

``state.json`` is a checkpoint, not history. The event stream reconstructs
state; the checkpoint exists so a resumed run does not redo work it already
paid for. Where the two disagree, the events win - which is why ``replay``
rebuilds every projection and every metric from the events alone.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import ids
from .hashing import digest, digest_file
from .redaction import Redactor
from .settings import LabConfig

MANIFEST_NAME = "run-manifest.json"
STATE_NAME = "state.json"
RESOLVED_CONFIG_NAME = "resolved-config.redacted.yaml"

RAW_FILES = (
    "events.jsonl",
    "spans.jsonl",
    "errors.jsonl",
    "mcp-calls.redacted.jsonl",
    "sources.jsonl",
    "claims.jsonl",
    "ingest-receipts.jsonl",
    "questions.jsonl",
    "answers.jsonl",
    "question-memories.jsonl",
    "proof-events.jsonl",
)

BENCHMARK_FILES = (
    "questions.jsonl",
    "expected-facts.jsonl",
    "expected-evidence.jsonl",
    "exclusions.jsonl",
    "abstention-cases.jsonl",
    "cohort-membership.jsonl",
    "benchmark-manifest.json",
)


def utcnow() -> datetime:
    return datetime.now(UTC)


def isonow() -> str:
    return utcnow().isoformat(timespec="microseconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class RunPaths:
    """Every path a run writes to. Created once, on entry."""

    root: Path

    @property
    def manifest(self) -> Path:
        return self.root / MANIFEST_NAME

    @property
    def state(self) -> Path:
        return self.root / STATE_NAME

    @property
    def resolved_config(self) -> Path:
        return self.root / RESOLVED_CONFIG_NAME

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def source_metadata(self) -> Path:
        return self.artifacts / "source-metadata"

    @property
    def permitted_content(self) -> Path:
        return self.artifacts / "permitted-content"

    @property
    def benchmark(self) -> Path:
        return self.root / "benchmark"

    @property
    def projections(self) -> Path:
        return self.root / "projections"

    @property
    def metrics(self) -> Path:
        return self.root / "metrics"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def integrity(self) -> Path:
        return self.root / "integrity"

    def raw_file(self, name: str) -> Path:
        return self.raw / name

    def ensure(self) -> RunPaths:
        for directory in (
            self.root,
            self.raw,
            self.artifacts,
            self.source_metadata,
            self.permitted_content,
            self.benchmark,
            self.projections,
            self.metrics,
            self.reports,
            self.integrity,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return self


def run_paths(output_root: str | Path, run_id: str) -> RunPaths:
    return RunPaths(Path(output_root) / run_id)


def environment_fingerprint() -> dict[str, Any]:
    """What about this laptop could change a result.

    Deliberately not a full package list: a fingerprint nobody can reproduce
    is a fingerprint nobody compares.
    """

    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(terse=True),
        "machine": platform.machine(),
    }


def build_manifest(
    *,
    config: LabConfig,
    run_id: str,
    nonce: str,
    config_digest: str,
    redactor: Redactor,
    command: str,
    package_version: str,
) -> dict[str, Any]:
    """The document every later comparison is judged against."""

    return {
        "schema_version": 1,
        "run_id": run_id,
        "experiment_name": config.experiment.name,
        "created_at": isonow(),
        "nonce": nonce,
        "config_digest": config_digest,
        "package_version": package_version,
        "command": command,
        "seed": config.experiment.seed,
        "arms": list(config.arms),
        "topics": [topic.id for topic in config.topics],
        "cost_budget_usd": config.experiment.cost_budget_usd,
        "runtime_budget_minutes": config.experiment.runtime_budget_minutes,
        "models": {
            role: {
                "provider": spec.provider,
                "model": spec.model,
                "reasoning_effort": spec.reasoning_effort,
                "temperature": spec.temperature,
                "max_output_tokens": spec.max_output_tokens,
                "tool_call_limit": spec.tool_call_limit,
            }
            for role, spec in sorted(config.models.items())
        },
        "pheasant": {
            "transport": config.pheasant.transport,
            "knowledge_base": config.pheasant.knowledge_base,
            "source_name": config.pheasant.source_name,
            "protocol_version": config.pheasant.protocol_version,
            "capabilities": {
                name: spec.tool for name, spec in sorted(config.pheasant.capabilities.items())
            },
        },
        "replay": config.replay.model_dump(mode="json"),
        "privacy": config.privacy.model_dump(mode="json"),
        "source_files": config.source_files,
        "overrides": config.overrides,
        "environment": environment_fingerprint(),
        "resolved_config": config.redacted(redactor),
        "status": "started",
    }


class RunState:
    """The resumable checkpoint.

    Every mutation writes the whole file through a temp name unique to this
    writer. A fixed ``.partial`` is a collision waiting for a second process,
    and a failed write that leaves its temp behind turns one orphan into one
    per attempt.
    """

    def __init__(self, path: Path, data: dict[str, Any] | None = None) -> None:
        self.path = path
        self.data: dict[str, Any] = data if data is not None else {}

    @classmethod
    def load(cls, path: Path) -> RunState:
        if path.is_file():
            return cls(path, json.loads(path.read_text(encoding="utf-8")))
        return cls(path, {})

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value
        self.save()

    def update(self, **values: Any) -> None:
        self.data.update(values)
        self.save()

    def mark_stage(self, stage: str, status: str, **detail: Any) -> None:
        stages = self.data.setdefault("stages", {})
        stages[stage] = {"status": status, "at": isonow(), **detail}
        self.save()

    def stage_status(self, stage: str) -> str | None:
        entry = self.data.get("stages", {}).get(stage)
        return entry.get("status") if entry else None

    def completed(self, stage: str) -> bool:
        return self.stage_status(stage) == "completed"

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(f".{os.getpid()}.{ids.new_nonce(4)}.tmp")
        try:
            temp.write_text(
                json.dumps(self.data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            _replace_with_retry(temp, self.path)
        finally:
            if temp.exists():
                temp.unlink(missing_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.{ids.new_nonce(4)}.tmp")
    try:
        temp.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
        )
        _replace_with_retry(temp, path)
    finally:
        if temp.exists():
            temp.unlink(missing_ok=True)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _replace_with_retry(source: Path, destination: Path, *, attempts: int = 10) -> None:
    """Finish an atomic write despite a short Windows sync/indexing lock."""

    delay = 0.05
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 1.0)


def write_checksums(paths: RunPaths) -> dict[str, str]:
    """Digest every file a run produced, into ``integrity/checksums.sha256``.

    Excludes the checksum file itself and the DuckDB projection: the
    projection is derived and rebuilt, so digesting it would make a rebuilt
    run look tampered with.
    """

    checksums: dict[str, str] = {}
    for file in sorted(paths.root.rglob("*")):
        if not file.is_file():
            continue
        relative = file.relative_to(paths.root).as_posix()
        if relative.startswith("integrity/") or relative.startswith("projections/"):
            continue
        checksums[relative] = digest_file(file)
    paths.integrity.mkdir(parents=True, exist_ok=True)
    lines = [f"{value.removeprefix('sha256:')}  {name}" for name, value in checksums.items()]
    (paths.integrity / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return checksums


def read_checksums(paths: RunPaths) -> dict[str, str]:
    file = paths.integrity / "checksums.sha256"
    if not file.is_file():
        return {}
    out: dict[str, str] = {}
    for line in file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value, _, name = line.partition("  ")
        out[name] = "sha256:" + value
    return out


def manifest_digest(paths: RunPaths) -> str:
    """The digest a completed run is archived under."""

    return digest(read_json(paths.manifest))
