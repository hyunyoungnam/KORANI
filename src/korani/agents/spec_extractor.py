"""Spec Extractor agent — stage C.

Turns parsed paper text into a SimulationSpec, KORANI's core artifact.
This is a flagged RISK STAGE (CLAUDE.md): a wrong parameter here poisons
everything downstream, so the prompt's honesty rules mirror the project
non-negotiables — missing values are marked missing, never invented, and
every uncertainty lands in ``ambiguities`` for human review and stage E
branch-on-ambiguity.
"""

from __future__ import annotations

from korani.jsonutil import extract_json_object
from korani.llm import LLMClient
from korani.models import SimulationSpec, TaskSpec

SYSTEM_PROMPT = """\
You are the Spec Extractor agent of KORANI. Read the research paper text and \
produce a SimulationSpec JSON so an engineer agent can REPRODUCE the paper's \
simulation with an open-source solver:
- PyBaMM: battery models (SPM/SPMe/DFN a.k.a. P2D, thermal, degradation).
- DEVSIM: semiconductor TCAD (drift-diffusion class device simulation).

HONESTY RULES (critical - violating them poisons the whole pipeline):
1. NEVER invent a parameter value. If it is not stated in the text, set its \
"value" to null, "source" to "missing", and add an entry to "ambiguities".
2. Copy values EXACTLY as written, with units, into strings (e.g. \
"1.2e-14 m^2 s^-1"). Do not convert units.
3. "target_results" must cite concrete reported results (figure/table \
numbers) that a reproduction can be verified against.
4. Anything underspecified (mesh, tolerances, unclear boundary conditions, \
parameter values only shown in plots) goes into "ambiguities" with candidate \
resolutions if you can think of any.

Respond with ONE JSON object and NOTHING else:
{
 "title": "<paper title>",
 "domain": "battery" | "semiconductor" | "unknown",
 "solver": "pybamm" | "devsim" | "none",
 "model_summary": "<one/two sentences, e.g. 'DFN (P2D) with SEI growth'>",
 "governing_equations": ["<name/description of each equation or submodel>"],
 "geometry": "<domain geometry / cell format / device structure>",
 "materials": ["<electrode/electrolyte or semiconductor materials>"],
 "parameters": [{"name": ..., "symbol": ..., "value": ..., "units": ...,
                 "source": "paper"|"assumed"|"missing", "notes": ...}],
 "operating_conditions": ["<cycling protocol / bias sweep / temperature ...>"],
 "numerical_settings": ["<mesh, solver tolerances, timestep - if given>"],
 "target_results": [{"description": ..., "location": "Figure 3",
                     "quantity": ..., "value": ...}],
 "ambiguities": [{"field": ..., "issue": ..., "candidates": [...]}]
}
"""


class SpecExtractionError(RuntimeError):
    def __init__(self, message: str, raw_output: str = ""):
        super().__init__(message)
        self.raw_output = raw_output


class SpecExtractor:
    def __init__(self, client: LLMClient, model: str, temperature: float = 0.1, max_tokens: int = 4096):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def extract(self, task: TaskSpec, paper_text: str) -> SimulationSpec:
        raw = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Research task: {task.task_en}\n"
                        f"Expected domain: {task.domain}\n\n"
                        f"Paper text:\n{paper_text}"
                    ),
                },
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        try:
            data = extract_json_object(raw)
        except ValueError as exc:
            raise SpecExtractionError(str(exc), raw) from exc

        # Normalize enums defensively; drop unknown keys via pydantic.
        if data.get("domain") not in ("battery", "semiconductor", "unknown"):
            data["domain"] = "unknown"
        if data.get("solver") not in ("pybamm", "devsim", "none"):
            data["solver"] = "none"
        data.pop("work_id", None)

        try:
            return SimulationSpec(**data)
        except Exception as exc:
            raise SpecExtractionError(
                f"Model JSON does not fit SimulationSpec: {exc}", raw
            ) from exc
