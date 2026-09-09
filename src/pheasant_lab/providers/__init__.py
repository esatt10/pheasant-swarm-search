"""Scientific literature providers.

Each provider answers one question - "what does this corpus have for this
query" - and normalises its answer into :class:`SourceCandidate`. Provider
order is configured, not hard-coded: which index a field lives in is a
property of the field, not of this repository.

The ``fixtures`` provider reads a local pack and makes a full offline run
possible. It is what the tests and ``pheasant-lab demo`` use, and a run
against it says so in its manifest.
"""

from .base import (
    LiteratureProvider,
    ProviderError,
    ProviderRegistry,
    SourceCandidate,
    build_provider,
    known_providers,
)

__all__ = [
    "LiteratureProvider",
    "ProviderError",
    "ProviderRegistry",
    "SourceCandidate",
    "build_provider",
    "known_providers",
]
