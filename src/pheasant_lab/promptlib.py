"""Locating the role prompts.

Prompts live in ``prompts/`` at the repository root, not inside the package,
because they are the part of this lab a reader is most likely to want to read
and edit, and burying them in ``site-packages`` makes that harder. The lookup
therefore checks, in order: an explicit override, the repository root relative
to this file, and the current working directory.

A prompt that cannot be found is an error rather than an empty string. A role
running with no instructions still produces output, and that output looks like
a result.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

ENV_VAR = "PHEASANT_LAB_PROMPTS"

ROLE_FILES = {
    "orchestrator": "orchestrator.md",
    "planner": "planner.md",
    "researcher": "researcher.md",
    "auditor": "coverage-auditor.md",
    "benchmark_builder": "benchmark-builder.md",
    "specialist": "specialist-answerer.md",
    "test_agent": "pheasant-answerer.md",
    "control": "pheasant-answerer.md",
}


class PromptNotFound(FileNotFoundError):
    pass


def prompt_dir() -> Path:
    override = os.environ.get(ENV_VAR)
    if override:
        return Path(override)
    candidates = [
        Path(__file__).resolve().parents[2] / "prompts",
        Path.cwd() / "prompts",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


@lru_cache(maxsize=32)
def load(role: str) -> str:
    """Return the system prompt for ``role``."""

    filename = ROLE_FILES.get(role)
    if filename is None:
        raise PromptNotFound(f"no prompt is registered for role '{role}'")
    path = prompt_dir() / filename
    if not path.is_file():
        raise PromptNotFound(
            f"prompt for role '{role}' not found at {path}. "
            f"Set {ENV_VAR} when running outside a checkout."
        )
    return path.read_text(encoding="utf-8")


def digest_all() -> dict[str, str]:
    """Digest every prompt, for the run manifest.

    A prompt edit changes what a role does, so two runs whose prompts differ
    are not comparable until the difference is enumerated - the same rule the
    resolved configuration digest exists for.
    """

    from .hashing import digest_text

    out: dict[str, str] = {}
    for role, filename in sorted(ROLE_FILES.items()):
        path = prompt_dir() / filename
        if path.is_file():
            out[role] = digest_text(path.read_text(encoding="utf-8"))
    return out
