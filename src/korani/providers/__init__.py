"""Literature metadata providers (KoCoScientist provider-layer pattern).

Each provider implements the ``PaperProvider`` protocol; adding a new source
later is a new module here, nothing else changes.
"""

from korani.providers.base import PaperProvider, safe_search
from korani.providers.openalex import OpenAlexProvider
from korani.providers.semanticscholar import SemanticScholarProvider

__all__ = [
    "PaperProvider",
    "safe_search",
    "OpenAlexProvider",
    "SemanticScholarProvider",
]
