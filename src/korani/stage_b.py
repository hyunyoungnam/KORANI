"""Stage B assembly: Search Planner → multi-provider search → Paper Triage.

The user picks from the resulting shortlist — stage B never auto-selects.
"""

from __future__ import annotations

from typing import Dict, List

from korani.agents.paper_triage import PaperTriage
from korani.agents.search_planner import SearchPlanner
from korani.llm import LLMClient, OpenAICompatClient
from korani.models import Shortlist, TaskSpec
from korani.providers import OpenAlexProvider, SemanticScholarProvider
from korani.search import SearchCoordinator


def _build_providers(search_config: Dict) -> List:
    verify_ssl = search_config.get("verify_ssl", True)
    registry = {
        "openalex": lambda: OpenAlexProvider(
            mailto=search_config.get("mailto", ""), verify_ssl=verify_ssl
        ),
        "semanticscholar": lambda: SemanticScholarProvider(verify_ssl=verify_ssl),
    }
    providers = []
    for name in search_config.get("providers", []):
        factory = registry.get(name)
        if factory is not None:
            providers.append(factory())
    return providers


def run_stage_b(spec: TaskSpec, config: Dict, client: LLMClient = None) -> Shortlist:
    """Run search-and-triage for a Mode B TaskSpec and return the shortlist.

    ``client`` is injectable for tests; defaults to the configured endpoint.
    """
    if client is None:
        client = OpenAICompatClient(
            base_url=config["llm"]["base_url"],
            api_key=config["llm"].get("api_key", "not-needed"),
        )
    search_config = config["search"]

    planner = SearchPlanner(client=client, model=config["models"]["search_planner"])
    queries = planner.plan(spec)

    coordinator = SearchCoordinator(
        providers=_build_providers(search_config),
        per_query_limit=search_config["per_query_limit"],
        max_candidates=search_config["max_candidates"],
    )
    candidates = coordinator.search(queries)

    triage = PaperTriage(client=client, model=config["models"]["paper_triage"])
    return triage.triage(
        spec,
        candidates,
        queries_used=queries,
        shortlist_size=search_config["shortlist_size"],
    )
