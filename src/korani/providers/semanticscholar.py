"""Semantic Scholar Graph API paper search — free tier, no API key required.

Unauthenticated requests share a global rate pool; provider failures are
tolerated upstream by ``safe_search``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx

from korani.models import PaperCandidate
from korani.providers.openalex import normalize_doi

S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

_FIELDS = ",".join(
    ["title", "abstract", "year", "authors", "externalIds", "citationCount", "openAccessPdf", "url"]
)


class SemanticScholarProvider:
    name = "semanticscholar"

    def __init__(
        self,
        client: Optional[httpx.Client] = None,
        timeout: float = 20.0,
        verify_ssl: bool = True,
    ):
        self._client = client or httpx.Client(timeout=timeout, verify=verify_ssl)

    def search(self, query: str, limit: int = 10) -> List[PaperCandidate]:
        response = self._client.get(
            S2_SEARCH_URL,
            params={"query": query, "limit": limit, "fields": _FIELDS},
        )
        response.raise_for_status()
        papers = response.json().get("data", []) or []
        return [self._parse(p) for p in papers if p.get("title")]

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
