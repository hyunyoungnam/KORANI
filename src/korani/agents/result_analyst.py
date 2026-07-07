"""Result Analyst agent — stage F. ⚠ RISK STAGE (vision over plots).

Compares the reproduction's outputs against the paper's reported results:
numeric evaluate.py lines, downsampled curve data as text, and — when a
vision-capable model is configured — PNG images (simulated curve plots and
renders of the paper pages that contain the cited figures). If the endpoint
rejects the multimodal request, the analyst degrades to text-only and says
so via ``used_vision``.

Unlike the Engineer, the analyst DOES see the paper's expected values —
judging agreement is its whole job.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from korani.jsonutil import extract_json_object
from korani.llm import LLMClient, LLMError
from korani.models import (
    AnalysisReport,
    CurveAssessment,
    EvaluationContract,
    SimulationSpec,
)

SYSTEM_PROMPT = """\
You are the Result Analyst agent of KORANI. A research paper's simulation \
was reproduced in Python; judge how well the reproduction's outputs agree \
with the paper's reported results.

You receive: the paper's target results (expected values, figure/table \
citations), the numeric verification output, simulated curve data as CSV \
text, and possibly images (simulated curve plots and renders of the paper \
pages containing the cited figures).

HONESTY RULES (non-negotiable):
1. Report a "match" ONLY when the evidence in front of you shows one. If \
you cannot see enough to judge a curve, say "uncertain" — never guess.
2. A mismatch is reported as a mismatch, with the deviation described \
concretely (direction, rough magnitude, region of the curve).
3. "diagnosis" must state the most likely PHYSICAL or NUMERICAL cause \
(wrong submodel, unit error, missing side reaction, mesh/tolerance, \
mis-read parameter), not a restatement of the numbers.
4. "suggested_fixes" are implementation-level changes a debugger could \
apply. NEVER suggest hardcoding or tuning values to force agreement.

Respond with ONE JSON object and NOTHING else:
{"verdict": "match" | "partial" | "mismatch" | "uncertain",
 "diagnosis": "<likely cause of any deviation; empty if match>",
 "suggested_fixes": ["<change 1>", ...],
 "curves": [{"key": "<curve check key>",
             "verdict": "match"|"mismatch"|"uncertain",
             "comment": "<one sentence>"}]}
"""


class AnalystError(RuntimeError):
    def __init__(self, message: str, raw_output: str = ""):
        super().__init__(message)
        self.raw_output = raw_output


class ResultAnalyst:
    def __init__(self, client: LLMClient, model: str, temperature: float = 0.1, max_tokens: int = 2048):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def analyze(
        self,
        spec: SimulationSpec,
        contract: EvaluationContract,
        variant: str,
        eval_summary: str,
        curve_texts: Dict[str, str],
        images: Optional[List[Tuple[str, str]]] = None,  # (label, data URL)
    ) -> AnalysisReport:
        text = self._build_text(spec, contract, eval_summary, curve_texts)
        used_vision = bool(images)
        if images:
            content: object = [{"type": "text", "text": text}]
            for label, data_url in images:
                content.append({"type": "text", "text": "Image: %s" % label})
                content.append({"type": "image_url", "image_url": {"url": data_url}})
        else:
            content = text

        try:
            raw = self._chat(content)
        except LLMError:
            if not images:
                raise
            # Endpoint/model rejected the multimodal request → text-only.
            used_vision = False
            raw = self._chat(text)

        report = self._parse(raw, contract)
        report.variant = variant
        report.used_vision = used_vision
        return report

    def _chat(self, content: object) -> str:
        return self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    def _build_text(
        self,
        spec: SimulationSpec,
        contract: EvaluationContract,
        eval_summary: str,
        curve_texts: Dict[str, str],
    ) -> str:
        targets = []
        for c in contract.checks:
            expected = c.expected_text or "(shown as a plot only)"
            targets.append(
                "- %s [%s] (%s): %s — expected %s"
                % (c.key, c.kind, c.location or "?", c.description, expected)
            )
        parts = [
            "Paper: %s" % spec.title,
            "Model: %s" % (spec.model_summary or spec.solver),
            "Target results:\n%s" % "\n".join(targets),
            "Numeric verification output:\n%s" % (eval_summary.strip() or "(not available)"),
        ]
        for key, csv_text in curve_texts.items():
            if csv_text:
                parts.append("Simulated curve %r (downsampled CSV):\n%s" % (key, csv_text))
            else:
                parts.append("Simulated curve %r: NO DATA FILE was produced." % key)
        return "\n\n".join(parts)

    def _parse(self, raw: str, contract: EvaluationContract) -> AnalysisReport:
        try:
            data = extract_json_object(raw)
        except ValueError as exc:
            raise AnalystError(str(exc), raw) from exc

        if data.get("verdict") not in ("match", "partial", "mismatch", "uncertain"):
            data["verdict"] = "uncertain"
        known_keys = {c.key for c in contract.checks}
        curves = []
        for item in data.get("curves") or []:
            if not isinstance(item, dict) or item.get("key") not in known_keys:
                continue  # hallucinated key
            verdict = item.get("verdict")
            if verdict not in ("match", "mismatch", "uncertain"):
                verdict = "uncertain"
            curves.append(
                CurveAssessment(
                    key=str(item["key"]),
                    verdict=verdict,
                    comment=str(item.get("comment") or ""),
                )
            )
        fixes = [str(f) for f in (data.get("suggested_fixes") or []) if str(f).strip()]
        return AnalysisReport(
            verdict=data["verdict"],
            diagnosis=str(data.get("diagnosis") or ""),
            suggested_fixes=fixes,
            curves=curves,
        )
