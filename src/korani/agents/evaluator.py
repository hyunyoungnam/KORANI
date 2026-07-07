"""Evaluator agent — stage D.

Drafts the evaluation contract from the SimulationSpec: each reported result
becomes a check with a verbatim expected value, a tolerance, and a kind
(numeric = machine-checkable by the generated evaluate.py; curve/qualitative
= deferred to the stage F Result Analyst). The LLM supplies only the
judgment — expected values, tolerances, kinds — never script code; stage D
renders evaluate.py deterministically from the checks. The draft MUST be
approved by a human before stage E may run (non-negotiable checkpoint).
"""

from __future__ import annotations

import re
from typing import List

from korani.jsonutil import extract_json_array
from korani.llm import LLMClient
from korani.models import EvaluationCheck, EvaluationContract, SimulationSpec

SYSTEM_PROMPT = """\
You are the Evaluator agent of KORANI. A research paper's simulation will be \
reproduced in Python; you define HOW the reproduction is verified against \
the paper's reported results.

You get a SimulationSpec JSON extracted from the paper. Turn its \
"target_results" (plus any other concrete reported values in the spec) into \
verification checks.

RULES:
1. NEVER invent an expected value. "expected_value" must be a number the \
spec actually reports. If a target exists only as a plotted curve, set \
"kind" to "curve" and "expected_value" to null — a vision analyst compares \
the plots later.
2. "key" is a short snake_case identifier; the reproduction code will write \
its computed value under this key in a results.json file. Keys must be unique.
3. Choose "rel_tol" (relative tolerance) honestly: 0.02-0.05 for values \
tabulated in the paper, 0.10-0.20 for values read off a figure.
4. "expected_text" is the value EXACTLY as the spec states it, with units. \
"expected_value" is that number parsed (same units as expected_text).

Respond with ONE JSON array and NOTHING else:
[{"key": "<snake_case>", "description": ..., "location": "Figure 3",
  "quantity": ..., "expected_text": "3.2 Ah" or null,
  "expected_value": 3.2 or null, "units": "Ah" or null,
  "rel_tol": 0.05, "kind": "numeric"|"curve"|"qualitative",
  "notes": ...}]
"""


class EvaluatorError(RuntimeError):
    def __init__(self, message: str, raw_output: str = ""):
        super().__init__(message)
        self.raw_output = raw_output


class Evaluator:
    def __init__(self, client: LLMClient, model: str, temperature: float = 0.1, max_tokens: int = 2048):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def draft(self, spec: SimulationSpec) -> EvaluationContract:
        raw = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "SimulationSpec:\n"
                    + spec.model_dump_json(exclude={"work_id"}, indent=2),
                },
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        checks = self._parse(raw)
        return EvaluationContract(
            paper_title=spec.title, solver=spec.solver, checks=checks
        )

    def _parse(self, raw: str) -> List[EvaluationCheck]:
        try:
            items = extract_json_array(raw)
        except ValueError as exc:
            raise EvaluatorError(str(exc), raw) from exc

        checks: List[EvaluationCheck] = []
        seen_keys = set()
        for item in items:
            if not isinstance(item, dict) or not item.get("description"):
                continue
            key = _snake_case(str(item.get("key") or item["description"]))
            if not key:
                continue
            base, n = key, 2
            while key in seen_keys:  # LLM reused a key — disambiguate, don't drop
                key = "%s_%d" % (base, n)
                n += 1
            seen_keys.add(key)

            try:
                expected_value = (
                    None if item.get("expected_value") is None
                    else float(item["expected_value"])
                )
            except (TypeError, ValueError):
                expected_value = None

            kind = item.get("kind")
            if kind not in ("numeric", "curve", "qualitative"):
                kind = "numeric"
            if kind == "numeric" and expected_value is None:
                # Not machine-checkable without a number — hand to the analyst.
                location = str(item.get("location") or "")
                kind = "curve" if "fig" in location.lower() else "qualitative"

            try:
                rel_tol = float(item.get("rel_tol", 0.05))
            except (TypeError, ValueError):
                rel_tol = 0.05
            rel_tol = max(0.005, min(0.5, rel_tol))

            checks.append(
                EvaluationCheck(
                    key=key,
                    description=str(item["description"]),
                    location=str(item.get("location") or ""),
                    quantity=str(item.get("quantity") or ""),
                    expected_text=(
                        None if item.get("expected_text") is None
                        else str(item["expected_text"])
                    ),
                    expected_value=expected_value,
                    units=None if item.get("units") is None else str(item["units"]),
                    rel_tol=rel_tol,
                    kind=kind,
                    notes=str(item.get("notes") or ""),
                )
            )
        if not checks:
            raise EvaluatorError("Evaluator output contained no usable checks.", raw)
        return checks


def _snake_case(text: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return key[:60]
