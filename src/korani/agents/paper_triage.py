"""Paper Triage agent — stage B (Mode B only).

Ranks search candidates by REPRODUCIBILITY, not topical relevance: can this
paper's simulation be rebuilt with PyBaMM/DEVSIM and verified against its
reported results? One batched LLM call scores all candidates; the user picks
from the shortlist (the agent never auto-selects).
"""

from __future__ import annotations

from typing import List

from korani.jsonutil import extract_json_array
from korani.llm import LLMClient
from korani.models import PaperCandidate, Shortlist, TaskSpec, TriageAssessment

SYSTEM_PROMPT = """\
You are the Paper Triage agent of KORANI. A researcher wants to REPRODUCE a \
paper's simulation in Python using open-source solvers:
- PyBaMM: battery models (SPM/SPMe/DFN a.k.a. P2D, thermal, degradation).
- DEVSIM: semiconductor TCAD (drift-diffusion class device simulation; NOT \
quantum transport, NOT Monte Carlo, NOT full process simulation).

For EACH candidate paper, judge REPRODUCIBILITY — not how interesting it is:
1. Completeness: are governing equations, parameters, geometry, and boundary/\
operating conditions likely fully specified in the paper?
2. Solver fit: can the model realistically be built in PyBaMM or DEVSIM?
3. Verifiability: does the paper report concrete results (figures/tables) to \
validate a reproduction against?
4. Access: an open-access PDF is a plus.

Respond with ONE JSON array and NOTHING else, scoring EVERY candidate:
[{"index": 1, "score": 0-10, "solver_fit": "pybamm"|"devsim"|"none",
  "rationale": "<one concise sentence>"}, ...]
"""

_ABSTRACT_CHARS = 700


class TriageError(RuntimeError):
    def __init__(self, message: str, raw_output: str = ""):
        super().__init__(message)
        self.raw_output = raw_output


def _candidate_block(index: int, candidate: PaperCandidate) -> str:
    abstract = (candidate.abstract or "(no abstract)")[:_ABSTRACT_CHARS]
    return (
        f"[{index}] {candidate.title}\n"
        f"    year: {candidate.year} | citations: {candidate.citation_count} "
        f"| open-access pdf: {'yes' if candidate.pdf_url else 'no'}\n"
        f"    abstract: {abstract}"
    )


class PaperTriage:
    def __init__(self, client: LLMClient, model: str, temperature: float = 0.2, max_tokens: int = 2048):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def triage(
        self,
        spec: TaskSpec,
        candidates: List[PaperCandidate],
        queries_used: List[str],
        shortlist_size: int = 5,
    ) -> Shortlist:
        if not candidates:
            return Shortlist(task_en=spec.task_en, queries_used=queries_used, entries=[])

        listing = "\n".join(_candidate_block(i + 1, c) for i, c in enumerate(candidates))
        raw = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Research task: {spec.task_en}\n"
                        f"Domain: {spec.domain}\n\n"
                        f"Candidates:\n{listing}"
                    ),
                },
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        assessments = self._parse(raw, candidates)
        assessments.sort(key=lambda a: a.score, reverse=True)
        return Shortlist(
            task_en=spec.task_en,
            queries_used=queries_used,
            entries=assessments[:shortlist_size],
        )

    def _parse(self, raw: str, candidates: List[PaperCandidate]) -> List[TriageAssessment]:
        items = _extract_json_array(raw)
        assessments: List[TriageAssessment] = []
        seen = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                index = int(item.get("index", 0))
            except (TypeError, ValueError):
                continue
            # Ignore hallucinated or duplicate indices instead of failing.
            if not (1 <= index <= len(candidates)) or index in seen:
                continue
            seen.add(index)
            try:
                score = max(0.0, min(10.0, float(item.get("score", 0))))
            except (TypeError, ValueError):
                score = 0.0
            solver_fit = item.get("solver_fit")
            if solver_fit not in ("pybamm", "devsim", "none"):
                solver_fit = "none"
            assessments.append(
                TriageAssessment(
                    candidate=candidates[index - 1],
                    score=score,
                    solver_fit=solver_fit,
                    rationale=str(item.get("rationale", "")),
                )
            )
        if not assessments:
            raise TriageError("Triage output contained no usable assessments.", raw)
        return assessments


def _extract_json_array(text: str) -> list:
    try:
        return extract_json_array(text)
    except ValueError as exc:
        raise TriageError(str(exc), text) from exc
