"""Small interface for solver-specific generation templates."""

from __future__ import annotations

from dataclasses import dataclass

from korani.models import EvaluationContract, SimulationSpec


@dataclass(frozen=True)
class TemplateContext:
    """Prompt material that constrains solver code generation."""

    name: str
    instructions: str
    template: str


class KnowledgeModule:
    """Base class for solver-scoped knowledge modules."""

    solver: str = ""

    def build_template_context(
        self, spec: SimulationSpec, contract: EvaluationContract
    ) -> TemplateContext:
        raise NotImplementedError
