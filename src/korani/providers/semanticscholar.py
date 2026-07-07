"""Semantic Scholar Graph API paper search — free tier, no API key required.

Unauthenticated requests share a global rate pool, so 429s are routine under
load: ``search`` retries with exponential backoff (honoring ``Retry-After``)
before giving up. Failures that survive the retries are tolerated upstream by
``safe_search``.
"""

from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Optional

import httpx

from korani.models import PaperCandidate
from korani.providers.openalex import normalize_doi

S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

_FIELDS = ",".join(
    ["title", "abstract", "year", "authors", "externalIds", "citationCount", "openAccessPdf", "url"]
)

# 429 = shared-pool rate limit; 5xx = transient server trouble. Anything else
# (403, 404, ...) will not get better on retry.
_RETRY_STATUS = (429, 500, 502, 503, 504)


class SemanticScholarProvider:
    name = "semanticscholar"

    def __init__(
        self,
        client: Optional[httpx.Client] = None,
        timeout: float = 20.0,
        verify_ssl: bool = True,
        max_retries: int = 3,
        backoff_base: float = 1.5,
        max_delay: float = 30.0,
    ):
        self._client = client or httpx.Client(timeout=timeout, verify=verify_ssl)
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.max_delay = max_delay

    def search(self, query: str, limit: int = 10) -> List[PaperCandidate]:
        response = None
        for attempt in range(self.max_retries + 1):
            if attempt:
                time.sleep(self._retry_delay(response, attempt))
            response = self._client.get(
                S2_SEARCH_URL,
                params={"query": query, "limit": limit, "fields": _FIELDS},
            )
            if response.status_code not in _RETRY_STATUS:
                break
        response.raise_for_status()
        papers = response.json().get("data", []) or []
        return [self._parse(p) for p in papers if p.get("title")]

    def _retry_delay(self, response: Optional[httpx.Response], attempt: int) -> float:
        """Server's Retry-After if given, else exponential backoff + jitter."""
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    return min(float(retry_after), self.max_delay)
                except ValueError:
                    pass  # HTTP-date form — fall through to backoff
        base = self.backoff_base * (2 ** (attempt - 1))
        return min(base + random.uniform(0, base), self.max_delay)

    def _parse(self, paper: Dict[str, Any]) -> PaperCandidate:
        authors = [a["name"] for a in (paper.get("authors") or [])[:10] if a.get("name")]
        external_ids = paper.get("externalIds") or {}
        oa_pdf = paper.get("openAccessPdf") or {}
        return PaperCandidate(
            title=paper["title"],
            abstract=paper.get("abstract"),
            year=paper.get("year"),
            authors=authors,
            doi=normalize_doi(external_ids.get("DOI")),
            url=paper.get("url"),
            pdf_url=oa_pdf.get("url"),
            citation_count=paper.get("citationCount"),
            sources=[self.name],
        )
