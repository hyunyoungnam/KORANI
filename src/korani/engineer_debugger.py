"""Stage E assembly: Engineer writes solver code; Debugger fixes failures.

- Requires an APPROVED evaluation contract (non-negotiable checkpoint) and
  holds the Engineer to its results.json / curve-CSV interface.
- Branch-on-ambiguity: ambiguities that survive the user clarification
  attempt fan out into up to ``budget.max_variants`` variants that resolve
  them differently — best-of-N single generation, NOT a tree.
- Budget the solver, not just tokens: every execution (initial runs and
  debug retries alike) draws from ``budget.max_solver_runs``.
- Right after a variant runs to completion, its numeric checks are scored
  with the stage D evaluate.py — the deterministic half of stage F. The
  Result Analyst (vision) and the escalation ladder are stage F work.
- When a Debugger fix turns a failing run into a working one, the fix is
  distilled into the failure playbook for future runs of that error class.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from korani import playbook
from korani.agents.debugger import Debugger, DebuggerError
from korani.agents.engineer import Engineer, EngineerError
from korani.knowledge import get_knowledge_module
from korani.llm import LLMClient, OpenAICompatClient
from korani.models import (
    EvaluationContract,
    SimulationSpec,
    StageEReport,
    VariantOutcome,
    VariantPlan,
)
from korani.runner import run_script
from korani.storage import Storage


class EngineerDebuggerError(RuntimeError):
    pass


class SolverBudget:
    """Per-task cap on simulation executions (KoCoScientist BudgetConfig,
    scaled down). Variants and debug retries all draw from the same pool."""

    def __init__(self, max_runs: int):
        self.max_runs = max_runs
        self.used = 0

    def try_consume(self) -> bool:
        if self.used >= self.max_runs:
            return False
        self.used += 1
        return True


def build_variant_plans(
    spec: SimulationSpec,
    user_resolutions: Optional[Dict[str, str]],
    max_variants: int,
) -> List[VariantPlan]:
    """Deterministic fan-out (no LLM): user decisions apply to every variant;
    surviving ambiguities get a different resolution per variant, taken from
    the ambiguity's candidates when it has any."""
    user_resolutions = user_resolutions or {}
    user_lines = [
        "%s: %s (user decision)" % (field, answer)
        for field, answer in user_resolutions.items()
    ]
    unresolved = [a for a in spec.ambiguities if a.field not in user_resolutions]
    n_variants = max(1, max_variants) if unresolved else 1

    plans: List[VariantPlan] = []
    seen = set()
    for k in range(n_variants):
        lines = list(user_lines)
        for a in unresolved:
            if a.candidates:
                choice = a.candidates[k % len(a.candidates)]
                lines.append("%s: %s" % (a.field, choice))
            elif k == 0:
                lines.append(
                    "%s: use the most standard/default resolution (%s)"
                    % (a.field, a.issue)
                )
            else:
                lines.append(
                    "%s: use a plausible ALTERNATIVE to the most standard "
                    "resolution (%s)" % (a.field, a.issue)
                )
        key = tuple(lines)
        if key in seen:  # all candidates exhausted → identical plan, drop it
            continue
        seen.add(key)
        plans.append(VariantPlan(name="variant%d" % (len(plans) + 1), resolutions=lines))
    return plans


def run_engineer_debugger(
    spec: SimulationSpec,
    contract: EvaluationContract,
    config: Dict,
    user_resolutions: Optional[Dict[str, str]] = None,
    client: Optional[LLMClient] = None,
    budget: Optional[SolverBudget] = None,
) -> StageEReport:
    """Generate, run, debug, and score code variants. ``client`` injectable
    for tests. Pass ``budget`` to share one solver pool with stage F's
    escalation ladder (the per-task cap spans both stages)."""
    if contract.status != "approved":
        raise EngineerDebuggerError(
            "The evaluation contract is still a draft — stage E only runs "
            "against a human-approved contract (rerun with --approve-contract "
            "after reviewing evaluate.py)."
        )
    if spec.solver not in ("pybamm", "devsim"):
        raise EngineerDebuggerError(
            "spec.solver is %r — no supported solver identified for this "
            "paper (PyBaMM/DEVSIM only)." % spec.solver
        )

    if client is None:
        client = OpenAICompatClient(
            base_url=config["llm"]["base_url"],
            api_key=config["llm"].get("api_key", "not-needed"),
        )
    data_dir = config.get("data_dir", "data")
    budget_cfg = config.get("budget", {})
    if budget is None:
        budget = SolverBudget(budget_cfg.get("max_solver_runs", 6))
    max_debug_retries = budget_cfg.get("max_debug_retries", 2)
    max_variants = budget_cfg.get("max_variants", 2)
    timeout_s = config.get("execution", {}).get("timeout_seconds", 900)

    engineer = Engineer(client=client, model=config["models"]["engineer"])
    debugger = Debugger(client=client, model=config["models"]["debugger"])
    plans = build_variant_plans(spec, user_resolutions, max_variants)

    work_id = contract.work_id or spec.work_id
    session = time.strftime("%Y%m%d-%H%M%S")
    session_dir = Path(data_dir) / "runs" / (work_id or "unknown") / session

    outcomes: List[VariantOutcome] = []
    for plan in plans:
        outcome = _run_variant(
            plan, spec, contract, engineer, debugger, budget,
            max_debug_retries, session_dir / plan.name, timeout_s, data_dir,
        )
        if outcome.status == "success":
            score_outcome(outcome, data_dir, work_id)
        outcomes.append(outcome)

    report = StageEReport(
        work_id=work_id,
        solver=spec.solver,
        variants=outcomes,
        solver_runs_used=budget.used,
        solver_runs_budget=budget.max_runs,
        best_variant=_pick_best(outcomes),
        session_dir=str(session_dir),
    )
    _persist_report(report, spec, session_dir, data_dir)
    return report


# ── variant loop ───────────────────────────────────────────────────


def _run_variant(
    plan: VariantPlan,
    spec: SimulationSpec,
    contract: EvaluationContract,
    engineer: Engineer,
    debugger: Debugger,
    budget: SolverBudget,
    max_debug_retries: int,
    vdir: Path,
    timeout_s: float,
    data_dir: str,
) -> VariantOutcome:
    vdir.mkdir(parents=True, exist_ok=True)
    outcome = VariantOutcome(
        name=plan.name, status="engineer_error", resolutions=plan.resolutions
    )
    try:
        code = engineer.generate(spec, contract, plan)
    except EngineerError as exc:
        outcome.error_tail = str(exc)
        return outcome

    script = vdir / "simulate.py"
    debug_used = 0
    fixed_signature = None  # error class of the failure the last fix targeted
    fixed_error_tail = ""
    template_context = _debug_template_context(spec, contract)
    while True:
        script.write_text(code, encoding="utf-8")
        outcome.code_path = str(script)
        if not budget.try_consume():
            outcome.status = "budget_exhausted"
            return outcome
        outcome.attempts += 1

        result = run_script(str(script), cwd=str(vdir), timeout_s=timeout_s)
        results_file = vdir / contract.results_file
        if not result.timed_out and result.exit_code == 0 and results_file.exists():
            outcome.status = "success"
            outcome.results_path = str(results_file)
            outcome.error_tail = ""
            if fixed_signature:  # a Debugger fix made this pass → harvest it
                playbook.record(
                    debugger.client, debugger.model, data_dir,
                    spec.solver, fixed_signature, fixed_error_tail, code,
                )
            return outcome

        if result.timed_out:
            error_tail = result.stderr_tail
            outcome.status = "timeout"
        elif result.exit_code == 0:
            error_tail = (
                "The script exited 0 but did not write %r — it must create "
                "that file at the end." % contract.results_file
            )
            outcome.status = "solver_error"
        else:
            error_tail = result.stderr_tail or (
                "exit code %s with empty stderr" % result.exit_code
            )
            outcome.status = "solver_error"
        outcome.error_tail = error_tail

        if debug_used >= max_debug_retries:  # bounded retries — escalation is stage F work
            return outcome
        debug_used += 1

        fixed_signature = playbook.error_signature(error_tail)
        fixed_error_tail = error_tail
        hint = playbook.lookup(data_dir, spec.solver, fixed_signature)
        try:
            code = debugger.debug(
                code,
                error_tail,
                spec.solver,
                hint=hint,
                template_context=template_context,
            )
        except DebuggerError as exc:
            outcome.error_tail += "\n[Debugger] %s" % exc
            return outcome


def _debug_template_context(spec: SimulationSpec, contract: EvaluationContract) -> str:
    try:
        context = get_knowledge_module(spec.solver).build_template_context(spec, contract)
    except ValueError:
        return ""
    return (
        "\n\nSolver template context to preserve during the fix (%s):\n%s\n\n"
        "Template scaffold:\n```python\n%s\n```"
        % (context.name, context.instructions, context.template)
    )


# ── scoring & persistence ──────────────────────────────────────────


_EVAL_LINE = re.compile(r"^\[(PASS|FAIL|MISSING|INVALID|ANALYST)")


def run_evaluate(
    data_dir: str, work_id: Optional[str], results_path: str
) -> Tuple[Optional[int], str]:
    """Run the stage D evaluate.py against a results file. Returns
    (exit_code, stdout); (None, "") when the script is missing or fails to
    run. Cheap comparison — never draws from the solver budget."""
    eval_script = Path(data_dir) / "evaluations" / (work_id or "unknown") / "evaluate.py"
    if not eval_script.exists():
        return None, ""
    try:
        proc = subprocess.run(
            [sys.executable, str(eval_script), results_path],
            capture_output=True, text=True, timeout=120,
            encoding="utf-8", errors="replace",
        )
    except (OSError, subprocess.TimeoutExpired):
        return None, ""
    return proc.returncode, proc.stdout or ""


def score_outcome(
    outcome: VariantOutcome, data_dir: str, work_id: Optional[str]
) -> str:
    """Fill the outcome's eval_* fields from an evaluate.py run; returns the
    evaluate.py stdout for the Result Analyst."""
    if not outcome.results_path:
        return ""
    exit_code, stdout = run_evaluate(data_dir, work_id, outcome.results_path)
    if exit_code is None:
        return ""
    outcome.eval_exit = exit_code
    outcome.eval_passed = outcome.eval_failed = outcome.eval_deferred = 0
    for line in stdout.splitlines():
        match = _EVAL_LINE.match(line.strip())
        if not match:
            continue
        tag = match.group(1)
        if tag == "PASS":
            outcome.eval_passed += 1
        elif tag == "ANALYST":
            outcome.eval_deferred += 1
        else:
            outcome.eval_failed += 1
    return stdout


def _pick_best(outcomes: List[VariantOutcome]) -> Optional[str]:
    scored = [o for o in outcomes if o.status == "success" and o.eval_exit is not None]
    if not scored:
        return None
    best = max(scored, key=lambda o: (o.eval_passed, -o.eval_failed))
    return best.name


def _persist_report(
    report: StageEReport, spec: SimulationSpec, session_dir: Path, data_dir: str
) -> None:
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "report.json").write_text(
        json.dumps(report.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if not report.work_id:
        return
    spec_sha = hashlib.sha256(spec.model_dump_json().encode("utf-8")).hexdigest()
    with Storage(data_dir) as storage:
        for outcome in report.variants:
            storage.save_run(report.work_id, spec_sha, report.solver, outcome)
