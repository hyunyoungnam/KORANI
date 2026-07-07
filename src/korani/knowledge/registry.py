"""Registry for solver-specific knowledge modules."""

from __future__ import annotations

from korani.knowledge.base import KnowledgeModule
from korani.knowledge.devsim import DevsimKnowledgeModule
from korani.knowledge.pybamm import PybammKnowledgeModule

_MODULES = {
    "devsim": DevsimKnowledgeModule(),
    "pybamm": PybammKnowledgeModule(),
}


def get_knowledge_module(solver: str) -> KnowledgeModule:
    try:
        return _MODULES[solver]
    except KeyError as exc:
        raise ValueError(f"No knowledge module registered for solver: {solver}") from exc
