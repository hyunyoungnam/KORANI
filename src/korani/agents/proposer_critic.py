"""Proposer + Critic agents — escalation rung 2 only (stage F).

Settled design (CLAUDE.md): minimal LLM calls before code exists, and no
N-round debate — the Proposer drafts up to 2 revised implementation plans,
the Critic reviews them ONCE (strike or refine), and the survivors re-enter
the stage E variant machinery. Both roles share one model slot
(``models.proposer_critic``, a reasoning model when available).
"""

from __future__ import annotations

from typing import List

from korani.jsonutil import extract_json_array
from korani.llm import LLMClient
from korani.models import RevisionPlan, SimulationSpec

PROPOSER_PROMPT = """\
You are the Proposer agent of KORANI. Earlier attempts to reproduce a \
paper's simulation ran but did not match the paper's reported results (or \
failed outright). Propose UP TO 2 genuinely different implementation plans \
for the next attempt.

RULES:
1. Implementation-level changes only: submodel choice, parameter \
interpretation (units, geometry convention), operating-condition setup, \
numerics (mesh, tolerances, solver options). Stay within the paper's physics.
2. NEVER propose hardcoding, tuning, or fitting values to force agreement \
with the paper — that would be fabrication.
3. Each plan must address the diagnosis in the failure history, and the two \
plans must differ in their main hypothesis about what went wrong.

Respond with ONE JSON array and NOTHING else:
[{"name": "<short name>", "rationale": "<why this might fix it>",
  "changes": ["<concrete change 1>", "<concrete change 2>", ...]}]
"""

CRITIC_PROMPT = """\
You are the Critic agent of KORANI. Review the proposed revision plans in \
ONE round — there is no debate after this.

Strike a plan if it: fabricates or tunes results to match the paper, \
contradicts the SimulationSpec's physics, repeats an approach the failure \
history shows was already tried, or is too vague to implement. Otherwise \
keep it, optionally tightening its changes.

Respond with ONE JSON array and NOTHING else, one entry per plan in order:
[{"index": 1, "keep": true|false, "comment": "<one sentence>",
  "revised_changes": ["<only if you tightened the changes>"]}]
"""


class EscalationError(RuntimeError):
    def __init__(self, message: str, raw_output: str = ""):
        super().__init__(message)
        self.raw_output = raw_output


class ProposerCritic:
    def __init__(self, client: LLMClient, model: str, temperature: float = 0.3, max_tokens: int = 2048):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def revise_plans(self, spec: SimulationSpec, history: str) -> List[RevisionPlan]:
        """One Proposer call, one Critic review; returns 1-2 surviving plans."""
        context = (
            "SimulationSpec:\n%s\n\nFailure history (honest log):\n%s"
            % (spec.model_dump_json(exclude={"work_id"}, indent=2), history)
        )
        raw = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": PROPOSER_PROMPT},
                {"role": "user", "content": context},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        plans = self._parse_plans(raw)

        # ONE critic round; if the critic output is unusable, keep the
        # proposer's plans rather than aborting the escalation.
        try:
            raw_review = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": CRITIC_PROMPT},
                    {
                        "role": "user",
                        "content": context + "\n\nProposed plans:\n" + "\n".join(
                            "%d. %s — %s\n   changes: %s"
                            % (i + 1, p.name, p.rationale, "; ".join(p.changes))
                            for i, p in enumerate(plans)
                        ),
                    },
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            plans = self._apply_review(plans, raw_review)
        except Exception:
            pass
        if not plans:
            raise EscalationError("The Critic struck every proposed plan.", raw)
        return plans[:2]

    def _parse_plans(self, raw: str) -> List[RevisionPlan]:
        try:
            items = extract_json_array(raw)
        except ValueError as exc:
            raise EscalationError(str(exc), raw) from exc
        plans = []
        for item in items:
            if not isinstance(item, dict):
                continue
            changes = [str(c) for c in (item.get("changes") or []) if str(c).strip()]
            if not changes:
                continue
            plans.append(
                RevisionPlan(
                    name=str(item.get("name") or "plan%d" % (len(plans) + 1)),
                    rationale=str(item.get("rationale") or ""),
                    changes=changes,
                )
            )
        if not plans:
            raise EscalationError("Proposer output contained no usable plans.", raw)
        return plans[:2]

    def _apply_review(self, plans: List[RevisionPlan], raw_review: str) -> List[RevisionPlan]:
        review = extract_json_array(raw_review)
        kept = []
        for item in review:
            if not isinstance(item, dict):
                continue
            try:
                index = int(item.get("index", 0)) - 1
            except (TypeError, ValueError):
                continue
            if not (0 <= index < len(plans)) or not item.get("keep", False):
                continue
            plan = plans[index]
            revised = [str(c) for c in (item.get("revised_changes") or []) if str(c).strip()]
            if revised:
                plan = plan.model_copy(update={"changes": revised})
            kept.append(plan)
        return kept
