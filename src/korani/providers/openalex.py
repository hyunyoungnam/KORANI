"""OpenAlex works search — free, no API key.

Set ``search.mailto`` in config.yaml (or pass mailto=) to join OpenAlex's
polite pool: https://docs.openalex.org/how-to-use-the-api/rate-limits-and-authentication
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx

from korani.models import PaperCandidate

OPENALEX_WORKS_URL = "https://api.openalex.org/works"

_SELECT_FIELDS = ",".join(
    [
        "id",
        "display_name",
        "publication_year",
        "authorships",
        "abstract_inverted_index",
        "doi",
        "cited_by_count",
        "primary_location",
        "open_access",
        "best_oa_location",
    ]
)


def reconstruct_abstract(inverted_index: Optional[Dict[str, List[int]]]) -> Optional[str]:
    """OpenAlex ships abstracts as {word: [positions]}; rebuild the text."""
    if not inverted_index:
        return None
    positions: List[tuple] = []
    for word, indices in inverted_index.items():
        for idx in indices:
            positions.append((idx, word))
    if not positions:
        return None
    positions.sort()
    return " ".join(word for _, word in positions)


def normalize_doi(doi: Optional[str]) -> Optional[str]:
    if not doi:
        return None
    return doi.lower().replace("https://doi.org/", "").strip() or None


class OpenAlexProvider:
    name = "openalex"

    def __init__(
        self,
        mailto: str = "",
        client: Optional[httpx.Client] = None,
        timeout: float = 20.0,
        verify_ssl: bool = True,
    ):
        self.mailto = mailto
        self._client = client or httpx.Client(timeout=timeout, verify=verify_ssl)

    def search(self, query: str, limit: int = 10) -> List[PaperCandidate]:
        params: Dict[str, Any] = {
            "search": query,
            "per-page": limit,
            "select": _SELECT_FIELDS,
        }
        if self.mailto:
            params["mailto"] = self.mailto
        response = self._client.get(OPENALEX_WORKS_URL, params=params)
        response.raise_for_status()
        results = response.json().get("results", [])
        return [self._parse(work) for work in results if work.get("display_name")]

    def _parse(self, work: Dict[str, Any]) -> PaperCandidate:
        authors = []
        for authorship in (work.get("authorships") or [])[:10]:
            name = (authorship.get("author") or {}).get("display_name")
            if name:
                authors.append(name)

        best_oa = work.get("best_oa_location") or {}
        open_access = work.get("open_access") or {}
        primary = work.get("primary_location") or {}

        return PaperCandidate(
            title=work["display_name"],
            abstract=reconstruct_abstract(work.get("abstract_inverted_index")),
            year=work.get("publication_year"),
            authors=authors,
            doi=normalize_doi(work.get("doi")),
            url=primary.get("landing_page_url") or work.get("id"),
            pdf_url=best_oa.get("pdf_url") or open_access.get("oa_url"),
            citation_count=work.get("cited_by_count"),
            sources=[self.name],
        )
