"""Provider protocol + failure isolation."""

from __future__ import annotations

import logging
from typing import List

try:
    from typing import Protocol
except ImportError:  # pragma: no cover
    from typing_extensions import Protocol

from korani.models import PaperCandidate

logger = logging.getLogger(__name__)


class PaperProvider(Protocol):
    name: str

    def search(self, query: str, limit: int = 10) -> List[PaperCandidate]:
        """Return candidate papers for a query (metadata only)."""
        ...


def safe_search(provider: PaperProvider, query: str, limit: int = 10) -> List[PaperCandidate]:
    """One provider failing (rate limit, outage) must not sink the search."""
    try:
        return provider.search(query, limit=limit)
    except Exception as exc:
        logger.warning("Provider %s failed for query %r: %s", provider.name, query, exc)
        return []
