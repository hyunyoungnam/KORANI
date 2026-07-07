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


class EvaluationCheck(BaseModel):
    """One verification check derived from a TargetResult (stage D).

    ``key`` names the value the stage E runner must write into the results
    file. ``kind`` separates machine-checkable numeric targets from
    curve/qualitative ones, which the stage F Result Analyst judges instead
    of evaluate.py. ``expected_text`` stays verbatim with units (like
    ParameterEntry.value) so the human reviewer sees exactly what the paper
    reported; ``expected_value`` is the parsed number evaluate.py compares
    against.
    """

    key: str
    description: str
    location: str = ""  # e.g. "Figure 3", "Table 2"
    quantity: str = ""
    expected_text: Optional[str] = None
    expected_value: Optional[float] = None
    units: Optional[str] = None
    rel_tol: float = Field(default=0.05, ge=0.0, le=1.0)
    kind: Literal["numeric", "curve", "qualitative"] = "numeric"
    notes: str = ""


class EvaluationContract(BaseModel):
    """Stage D output: how the reproduction will be verified.

    Drafted by the Evaluator, rendered to evaluate.py, and locked only after
    explicit human approval (non-negotiable checkpoint) — stage E must
    refuse to run while ``status`` is "draft".
    """

    work_id: Optional[str] = None
    paper_title: str = ""
    solver: Literal["pybamm", "devsim", "none"] = "none"
    checks: List[EvaluationCheck] = Field(default_factory=list)
    results_file: str = "results.json"  # contract with the stage E runner
    status: Literal["draft", "approved"] = "draft"


class VariantPlan(BaseModel):
    """How one stage E variant resolves the spec's ambiguities.

    Branch-on-ambiguity (CLAUDE.md): ambiguities that survive the user
    clarification attempt fan out into a few variants that resolve them
    differently — best-of-N single generation, NOT a tree.
    """

    name: str
    resolutions: List[str] = Field(default_factory=list)  # "field: chosen resolution"


class VariantOutcome(BaseModel):
    """What happened to one stage E variant, reported honestly."""

    name: str
    status: Literal[
        "success", "solver_error", "timeout", "engineer_error", "budget_exhausted"
    ]
    attempts: int = 0  # solver executions consumed by this variant
    resolutions: List[str] = Field(default_factory=list)
    code_path: Optional[str] = None
    results_path: Optional[str] = None
    error_tail: str = ""
    # evaluate.py outcome (numeric checks only; None = never ran)
    eval_exit: Optional[int] = None
    eval_passed: int = 0
    eval_failed: int = 0
    eval_deferred: int = 0  # curve/qualitative → stage F Result Analyst


class StageEReport(BaseModel):
    """Stage E output: every variant's outcome plus budget accounting."""

    work_id: Optional[str] = None
    solver: Literal["pybamm", "devsim", "none"] = "none"
    variants: List[VariantOutcome] = Field(default_factory=list)
    solver_runs_used: int = 0
    solver_runs_budget: int = 0
    best_variant: Optional[str] = None  # most numeric checks passed
    session_dir: Optional[str] = None  # runs/{work_id}/{session}; stage F adds rungs here


class CurveAssessment(BaseModel):
    """Result Analyst verdict for one curve/qualitative check (stage F)."""

    key: str
    verdict: Literal["match", "mismatch", "uncertain"] = "uncertain"
    comment: str = ""


class AnalysisReport(BaseModel):
    """Result Analyst output (stage F) — ⚠ risk stage (vision over plots).

    Persisted as a file referenced from the DB (Analysis Base pattern).
    ``verdict`` covers the whole reproduction; the honesty non-negotiable
    applies: a match is only claimed when the evidence shows one, and
    "uncertain" is an allowed answer.
    """

    variant: str = ""
    verdict: Literal["match", "partial", "mismatch", "uncertain"] = "uncertain"
    diagnosis: str = ""  # what deviates and the likely physical/numerical cause
    suggested_fixes: List[str] = Field(default_factory=list)  # feeds rung 1
    curves: List[CurveAssessment] = Field(default_factory=list)
    used_vision: bool = False


class RevisionPlan(BaseModel):
    """Rung 2 output: implementation-level changes for a fresh Engineer
    attempt (one Proposer↔Critic round, up to 2 plans — never a tree)."""

    name: str
    changes: List[str] = Field(default_factory=list)
    rationale: str = ""


class StageFReport(BaseModel):
    """Stage F output: final verdict plus the honest escalation history."""

    work_id: Optional[str] = None
    verdict: Literal["match", "mismatch", "no_result"] = "no_result"
    final_variant: Optional[str] = None
    rungs_used: int = 0  # 0 = matched without escalation
    history: List[str] = Field(default_factory=list)  # one line per ladder event
    analysis: Optional[AnalysisReport] = None  # latest analyst report
    solver_runs_used: int = 0
    solver_runs_budget: int = 0
