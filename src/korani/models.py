"""Typed artifacts passed between KORANI agents."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

try:
    from typing import Literal
except ImportError:  # pragma: no cover
    from typing_extensions import Literal


class AmbiguousTerm(BaseModel):
    """A Korean technical term the Interpreter refuses to silently translate.

    Non-negotiable (CLAUDE.md): ambiguous Korean terms are flagged for human
    review, never guessed.
    """

    term_ko: str
    candidates_en: List[str] = Field(default_factory=list)
    reason: str = ""


class TaskSpec(BaseModel):
    """Output of stage A (Understand): the user's Korean request, structured.

    ``mode`` is decided programmatically from whether a paper was attached —
    never by the LLM:
      - "A": paper attached → skip search, go straight to Extract.
      - "B": vague idea → search-and-triage funnel.
    """

    mode: Literal["A", "B"]
    domain: Literal["battery", "semiconductor", "unknown"] = "unknown"
    task_en: str
    search_queries_en: List[str] = Field(default_factory=list)
    ambiguous_terms: List[AmbiguousTerm] = Field(default_factory=list)
    clarifying_questions_ko: List[str] = Field(default_factory=list)
    reply_ko: str = ""
    paper_path: Optional[str] = None


class PaperCandidate(BaseModel):
    """One paper found during stage B search (metadata only, no fulltext)."""

    title: str
    abstract: Optional[str] = None
    year: Optional[int] = None
    authors: List[str] = Field(default_factory=list)
    doi: Optional[str] = None
    url: Optional[str] = None
    pdf_url: Optional[str] = None
    citation_count: Optional[int] = None
    sources: List[str] = Field(default_factory=list)  # provider names that returned it


class TriageAssessment(BaseModel):
    """Paper Triage verdict for one candidate.

    ``score`` is REPRODUCIBILITY (0-10) — can this paper's simulation be
    rebuilt and verified? — not topical relevance.
    """

    candidate: PaperCandidate
    score: float = Field(ge=0, le=10)
    solver_fit: Literal["pybamm", "devsim", "none"] = "none"
    rationale: str = ""


class Shortlist(BaseModel):
    """Stage B output: ranked candidates for the user to pick from."""

    task_en: str
    queries_used: List[str] = Field(default_factory=list)
    entries: List[TriageAssessment] = Field(default_factory=list)


class ParameterEntry(BaseModel):
    """One physical/model parameter extracted from the paper.

    ``value`` is kept as a string exactly as written (with units) — unit
    conversion is the Engineer's job, and silent conversion here would hide
    extraction errors from the human reviewer.
    """

    name: str
    symbol: Optional[str] = None
    value: Optional[str] = None
    units: Optional[str] = None
    source: Literal["paper", "assumed", "missing"] = "paper"
    notes: str = ""


class TargetResult(BaseModel):
    """A concrete reported result the reproduction will be verified against."""

    description: str
    location: str = ""  # e.g. "Figure 3", "Table 2"
    quantity: str = ""
    value: Optional[str] = None  # if tabulated in text


class AmbiguousField(BaseModel):
    """An underspecified aspect of the paper.

    Feeds two mechanisms: the human review of the spec, and
    branch-on-ambiguity in stage E (candidate implementations that resolve
    the ambiguity differently).
    """

    field: str
    issue: str
    candidates: List[str] = Field(default_factory=list)


class SimulationSpec(BaseModel):
    """Stage C output — KORANI's core artifact.

    A typed, human-reviewable intermediate representation of the paper's
    simulation: everything the Evaluator (stage D) and Engineer (stage E)
    need, and nothing that requires re-reading the paper.
    """

    title: str
    domain: Literal["battery", "semiconductor", "unknown"] = "unknown"
    solver: Literal["pybamm", "devsim", "none"] = "none"
    model_summary: str = ""  # e.g. "DFN (P2D) with SEI growth side reaction"
    governing_equations: List[str] = Field(default_factory=list)
    geometry: str = ""
    materials: List[str] = Field(default_factory=list)
    parameters: List[ParameterEntry] = Field(default_factory=list)
    operating_conditions: List[str] = Field(default_factory=list)
    numerical_settings: List[str] = Field(default_factory=list)
    target_results: List[TargetResult] = Field(default_factory=list)
    ambiguities: List[AmbiguousField] = Field(default_factory=list)
    work_id: Optional[str] = None  # set on persistence
