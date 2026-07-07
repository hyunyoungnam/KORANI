"""Search Planner agent — stage B (Mode B only).

Expands the Interpreter's initial queries into a diverse English query set
for the multi-provider search. Small-model role (see config.yaml registry).
"""

from __future__ import annotations

from typing import List

from korani.jsonutil import extract_json_object
from korani.llm import LLMClient
from korani.models import TaskSpec

SYSTEM_PROMPT = """\
You are the Search Planner agent of KORANI, a co-scientist system that helps \
researchers find a research paper whose simulation can be REPRODUCED with \
open-source solvers (PyBaMM for batteries, DEVSIM for semiconductor TCAD).

Given a research task, produce diverse English literature-search queries. \
Favor queries likely to surface papers with complete model descriptions \
(equations, parameters) and validation data. Include at least one query \
naming a concrete modeling approach for the domain (e.g. "P2D model", \
"drift-diffusion") and at most one query naming the solver itself.

Respond with ONE JSON object and NOTHING else:
{"queries": ["...", "..."]}   (3-5 queries)
"""


class SearchPlanner:
    def __init__(self, client: LLMClient, model: str, temperature: float = 0.4, max_tokens: int = 512):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def plan(self, spec: TaskSpec, max_queries: int = 6) -> List[str]:
        """Return interpreter queries + LLM expansions, deduped, capped.

        Falls back to the interpreter's queries if the LLM output is
        unusable — stage B must not die on a bad planner response.
        """
        llm_queries: List[str] = []
        try:
            raw = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Research task: {spec.task_en}\n"
                            f"Domain: {spec.domain}\n"
                            f"Existing queries: {spec.search_queries_en}"
                        ),
                    },
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            llm_queries = _parse_queries(raw)
        except Exception:
            llm_queries = []

        merged: List[str] = []
        for query in list(spec.search_queries_en) + llm_queries:
            normalized = query.strip()
            if normalized and normalized.lower() not in {q.lower() for q in merged}:
                merged.append(normalized)
        return merged[:max_queries] if merged else [spec.task_en]


def _parse_queries(text: str) -> List[str]:
    try:
        data = extract_json_object(text)
    except ValueError:
        return []
    queries = data.get("queries", [])
    return [q for q in queries if isinstance(q, str) and q.strip()]
