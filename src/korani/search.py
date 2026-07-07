"""Stage B search coordination — deterministic, no LLM.

Fan queries out to all providers, deduplicate (DOI first, normalized title as
fallback), pre-rank with a cheap heuristic, and cap the candidate set that
goes to the (LLM) Paper Triage agent. Mirrors KoCoScientist stages 1/1.5/2 in
miniature.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional

from korani.models import PaperCandidate
from korani.providers.base import PaperProvider, safe_search


def _title_key(title: str) -> str:
    return re.sub(r"[^a-z0-9]", "", title.lower())


def _dedup_key(candidate: PaperCandidate) -> str:
    if candidate.doi:
        return "doi:" + candidate.doi
    return "title:" + _title_key(candidate.title)


def _merge(existing: PaperCandidate, new: PaperCandidate) -> PaperCandidate:
    """Keep the richer record; union the provider list."""
    merged = existing.model_copy()
    merged.sources = sorted(set(existing.sources) | set(new.sources))
    if not merged.abstract and new.abstract:
        merged.abstract = new.abstract
    if not merged.pdf_url and new.pdf_url:
        merged.pdf_url = new.pdf_url
    if not merged.url and new.url:
        merged.url = new.url
    if not merged.doi and new.doi:
        merged.doi = new.doi
    if (new.citation_count or 0) > (merged.citation_count or 0):
        merged.citation_count = new.citation_count
    if merged.year is None and new.year is not None:
        merged.year = new.year
    return merged


def prerank_score(candidate: PaperCandidate) -> float:
    """Cheap pre-triage ranking: citations (log), recency, OA pdf, abstract.

    Only decides WHICH candidates the LLM triage sees, not the final order —
    reproducibility judgment belongs to the triage agent.
    """
    score = math.log10((candidate.citation_count or 0) + 1)
    if candidate.year:
        score += max(0.0, min(candidate.year - 2010, 15) * 0.05)
    if candidate.pdf_url:
        score += 1.0
    if candidate.abstract:
        score += 0.5
    return score


class SearchCoordinator:
    def __init__(
        self,
        providers: List[PaperProvider],
        per_query_limit: int = 10,
        max_candidates: int = 12,
    ):
        self.providers = providers
        self.per_query_limit = per_query_limit
        self.max_candidates = max_candidates

    def search(self, queries: List[str]) -> List[PaperCandidate]:
        pool: Dict[str, PaperCandidate] = {}
        for query in queries:
            for provider in self.providers:
                for candidate in safe_search(provider, query, limit=self.per_query_limit):
                    key = _dedup_key(candidate)
                    if key in pool:
                        pool[key] = _merge(pool[key], candidate)
                    else:
                        pool[key] = candidate

        ranked = sorted(pool.values(), key=prerank_score, reverse=True)
        return ranked[: self.max_candidates]
