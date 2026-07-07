"""Engineer agent — stage E.

Writes the solver script from the SimulationSpec, held to the approved
evaluation contract's interface: numeric check keys → results.json, curve
check keys → CSV files. The Engineer is deliberately NOT shown the paper's
expected values — it must compute results, never hardcode them, so the
stage D/F verification stays meaningful.
"""

from __future__ import annotations

from korani.codeutil import extract_python_code
from korani.llm import LLMClient
from korani.models import EvaluationContract, SimulationSpec, VariantPlan

SYSTEM_PROMPT = """\
You are the Engineer agent of KORANI. Write ONE complete, runnable Python \
script that reproduces the paper's simulation described by the \
SimulationSpec, using the open-source solver named by the user \
(PyBaMM for battery models, DEVSIM for semiconductor TCAD).

REQUIREMENTS (the verification pipeline depends on them):
1. Self-contained script, no command-line arguments, runs top-to-bottom \
with `python script.py`.
2. At the END the script MUST write `results.json` in the current working \
directory: a JSON object mapping EXACTLY the numeric check keys listed by \
the user to floats COMPUTED FROM THE SIMULATION, in the stated units. \
NEVER hardcode a literature/expected value as a result — an honest wrong \
number beats a dishonest right one.
3. For each curve check key, save the simulated curve to `<key>.csv` \
(two columns with a header) for later plot comparison.
4. Take parameter values from the spec verbatim; when the solver needs \
different units, show the conversion in a comment. Mark every assumed or \
ambiguity-resolved value with an `# ASSUMPTION:` comment.
5. Never call plt.show() or block on input; saving figures as PNG is fine.
6. No network access and no package installation inside the script.
7. Print brief progress lines so failures are diagnosable.

Respond with the script in ONE ```python code fence and nothing else.
"""


class EngineerError(RuntimeError):
    def __init__(self, message: str, raw_output: str = ""):
        super().__init__(message)
        self.raw_output = raw_output


def _check_lines(contract: EvaluationContract) -> str:
    numeric, curves = [], []
    for c in contract.checks:
        units = " in %s" % c.units if c.units else ""
        if c.kind == "numeric" and c.expected_value is not None:
            numeric.append('- "%s": %s (%s%s)' % (c.key, c.description, c.quantity or "value", units))
        else:
            curves.append('- "%s": %s (%s)' % (c.key, c.description, c.location or "curve"))
    parts = []
    if numeric:
        parts.append("Numeric check keys for results.json:\n" + "\n".join(numeric))
    if curves:
        parts.append("Curve check keys (save <key>.csv each):\n" + "\n".join(curves))
    return "\n\n".join(parts)


class Engineer:
    def __init__(self, client: LLMClient, model: str, temperature: float = 0.2, max_tokens: int = 6144):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(
        self, spec: SimulationSpec, contract: EvaluationContract, plan: VariantPlan
    ) -> str:
        resolutions = ""
        if plan.resolutions:
            # Covers both stage E ambiguity resolutions and stage F rung-2
            # revision directives — the Engineer treats them the same way.
            resolutions = (
                "\n\nFollow these implementation directives EXACTLY "
                "(ambiguity resolutions / revision guidance; document each "
                "choice in the code with # ASSUMPTION:):\n"
                + "\n".join("- %s" % r for r in plan.resolutions)
            )
        raw = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Solver: %s\n\nSimulationSpec:\n%s\n\n%s%s"
                        % (
                            spec.solver,
                            # target_results withheld: they hold the paper's
                            # reported numbers — the Engineer must not see
                            # what it could hardcode.
                            spec.model_dump_json(
                                exclude={"work_id", "target_results"}, indent=2
                            ),
                            _check_lines(contract),
                            resolutions,
                        )
                    ),
                },
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        try:
            return extract_python_code(raw)
        except ValueError as exc:
            raise EngineerError(str(exc), raw) from exc
