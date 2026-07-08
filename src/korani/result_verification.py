"""Stage F assembly: Result Analyst verdict + escalation ladder.

    verify best variant → match? → done (rungs_used = 0)
      rung 1: ONE Debugger retry guided by the analyst's diagnosis
      rung 2: ONE Proposer↔Critic round → up to 2 revised plans → the
              stage E variant machinery runs them
      rung 3: stop and report honestly what failed and why

Acceptance policy: a reproduction is a "match" when the numeric checks pass
(evaluate.py exit 0) AND the Result Analyst does not call the curves a
mismatch. An analyst verdict of "uncertain" with passing numerics is
accepted but recorded honestly in the history; "partial"/"mismatch"
escalates. Solver executions on every rung draw from the same per-task
budget as stage E (pass the shared ``budget``).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from korani.agents.debugger import Debugger, DebuggerError
from korani.agents.engineer import Engineer
from korani.agents.proposer_critic import EscalationError, ProposerCritic
from korani.agents.result_analyst import AnalystError, ResultAnalyst
from korani.figures import (
    find_figure_page_png,
    png_data_url,
    read_curve_csv,
    render_curve_png,
)
from korani.llm import LLMClient, client_for_role
from korani.models import (
    AnalysisReport,
    EvaluationContract,
    SimulationSpec,
    StageEReport,
    StageFReport,
    VariantOutcome,
    VariantPlan,
)
from korani.engineer_debugger import SolverBudget, _run_variant, score_outcome
from korani.runner import run_script
from korani.storage import Storage


class ResultVerificationError(RuntimeError):
    pass


def run_result_verification(
    spec: SimulationSpec,
    contract: EvaluationContract,
    e_report: StageEReport,
    config: Dict,
    budget: Optional[SolverBudget] = None,
    client: Optional[LLMClient] = None,
) -> StageFReport:
    """Verify the stage E outcome and climb the escalation ladder if needed.
    ``budget`` should be the same pool stage E used; ``client`` injectable
    for tests."""
    data_dir = config.get("data_dir", "data")
    budget_cfg = config.get("budget", {})
    if budget is None:
        budget = SolverBudget(budget_cfg.get("max_solver_runs", 6))
        budget.used = e_report.solver_runs_used  # continue stage E's pool
    max_debug_retries = budget_cfg.get("max_debug_retries", 2)
    timeout_s = config.get("execution", {}).get("timeout_seconds", 900)

    work_id = e_report.work_id or contract.work_id or spec.work_id
    session_dir = Path(
        e_report.session_dir
        or Path(data_dir) / "runs" / (work_id or "unknown") / time.strftime("%Y%m%d-%H%M%S")
    )
    analysis_dir = session_dir / "analysis"

    analyst_client, analyst_model = client_for_role(config, "result_analyst", client)
    debugger_client, debugger_model = client_for_role(config, "debugger", client)
    engineer_client, engineer_model = client_for_role(config, "engineer", client)
    analyst = ResultAnalyst(client=analyst_client, model=analyst_model)
    debugger = Debugger(client=debugger_client, model=debugger_model)
    engineer = Engineer(client=engineer_client, model=engineer_model)
    pc_client, pc_model = client_for_role(config, "proposer_critic", client)
    proposer_critic = ProposerCritic(client=pc_client, model=pc_model)

    report = StageFReport(work_id=work_id, solver_runs_budget=budget.max_runs)
    history = report.history
    had_success = False

    def finish(verdict, final=None, rungs=0):
        report.verdict = verdict
        report.final_variant = final
        report.rungs_used = rungs
        report.solver_runs_used = budget.used
        _persist_report(report, session_dir, data_dir)
        return report

    def verify(outcome: VariantOutcome) -> Tuple[bool, AnalysisReport]:
        """Numeric rescore + analyst review; persists the analysis."""
        eval_stdout = score_outcome(outcome, data_dir, work_id)
        try:
            analysis = _analyze(
                analyst, spec, contract, outcome, eval_stdout,
                config, data_dir, work_id, analysis_dir,
            )
        except AnalystError as exc:
            # A broken analyst must not crash the ladder — record it and
            # fall back to numeric-only judgment, stated honestly.
            analysis = AnalysisReport(
                variant=outcome.name,
                verdict="uncertain",
                diagnosis="Result Analyst output was unusable: %s" % exc,
            )
        report.analysis = analysis
        history.append(
            "%s: numeric %s (%d pass / %d fail), analyst verdict %r%s"
            % (
                outcome.name,
                "PASS" if outcome.eval_exit == 0 else "FAIL",
                outcome.eval_passed, outcome.eval_failed,
                analysis.verdict,
                " — " + analysis.diagnosis if analysis.diagnosis else "",
            )
        )
        accepted = outcome.eval_exit == 0 and analysis.verdict in ("match", "uncertain")
        if accepted and analysis.verdict == "uncertain":
            history.append(
                "%s: curves could not be fully verified (analyst uncertain) — "
                "accepted on numeric checks; review the curve CSVs yourself."
                % outcome.name
            )
        return accepted, analysis

    # ── verify stage E's best variant ──────────────────────────────
    best = _find(e_report, e_report.best_variant)
    if best is not None:
        had_success = True
        accepted, analysis = verify(best)
        if accepted:
            return finish("match", best.name, rungs=0)
    else:
        history.append(
            "stage E produced no successful variant (%s)"
            % ", ".join("%s: %s" % (v.name, v.status) for v in e_report.variants)
        )
        analysis = None

    # ── rung 1: guided Debugger retry on the best variant ─────────
    if best is not None and analysis is not None:
        report.rungs_used = 1
        history.append("rung 1: Debugger retry guided by the analyst's diagnosis")
        outcome = _guided_retry(
            best, analysis, spec, contract, debugger, budget,
            session_dir / "rung1", timeout_s, data_dir, work_id, history,
        )
        if outcome is not None and outcome.status == "success":
            had_success = True
            accepted, _ = verify(outcome)
            if accepted:
                return finish("match", outcome.name, rungs=1)
        if budget.used >= budget.max_runs:
            history.append("solver budget exhausted — cannot climb further")
            return finish("mismatch" if had_success else "no_result", rungs=report.rungs_used)

    # ── rung 2: ONE Proposer↔Critic round → revised plans → E ─────
    report.rungs_used = 2
    history.append("rung 2: one Proposer↔Critic round for revised plans")
    try:
        plans = proposer_critic.revise_plans(spec, "\n".join(history))
    except EscalationError as exc:
        history.append("rung 2 aborted: %s" % exc)
        plans = []
    for i, plan in enumerate(plans, start=1):
        history.append(
            "rung 2 plan %d (%s): %s" % (i, plan.name, "; ".join(plan.changes))
        )
        variant_plan = VariantPlan(
            name="rung2_plan%d" % i, resolutions=list(plan.changes)
        )
        outcome = _run_variant(
            variant_plan, spec, contract, engineer, debugger, budget,
            max_debug_retries, session_dir / variant_plan.name, timeout_s, data_dir,
        )
        _save_run(outcome, spec, data_dir, work_id)
        if outcome.status == "success":
            had_success = True
            accepted, _ = verify(outcome)
            if accepted:
                return finish("match", outcome.name, rungs=2)
        else:
            history.append(
                "%s: %s%s"
                % (
                    outcome.name, outcome.status,
                    " — " + outcome.error_tail.strip().splitlines()[-1]
                    if outcome.error_tail.strip() else "",
                )
            )
        if outcome.status == "budget_exhausted":
            history.append("solver budget exhausted — cannot climb further")
            break

    # ── rung 3: stop and report honestly ──────────────────────────
    report.rungs_used = 3
    history.append(
        "rung 3: stopping — the reproduction %s. All artifacts and this "
        "history are preserved for human review."
        % ("does not match the paper's reported results" if had_success
           else "never produced a complete run")
    )
    return finish("mismatch" if had_success else "no_result", rungs=3)


# ── helpers ────────────────────────────────────────────────────────


def _find(e_report: StageEReport, name: Optional[str]) -> Optional[VariantOutcome]:
    for v in e_report.variants:
        if name is not None and v.name == name and v.status == "success":
            return v
    return None


def _guided_retry(
    best: VariantOutcome,
    analysis: AnalysisReport,
    spec: SimulationSpec,
    contract: EvaluationContract,
    debugger: Debugger,
    budget: SolverBudget,
    rung_dir: Path,
    timeout_s: float,
    data_dir: str,
    work_id: Optional[str],
    history: List[str],
) -> Optional[VariantOutcome]:
    """One guided fix + one run. Not a crash fix: the script works but its
    results deviate, so the Debugger gets the diagnosis instead of a
    traceback."""
    try:
        code = Path(best.code_path).read_text(encoding="utf-8")
    except (OSError, TypeError):
        history.append("rung 1 skipped: variant code unavailable")
        return None
    guidance = (
        "The script runs to completion, but its results DO NOT match the "
        "paper's reported values.\n\nAnalyst diagnosis: %s\n\nSuggested "
        "fixes:\n%s\n\nRevise the script accordingly."
        % (
            analysis.diagnosis or "(none)",
            "\n".join("- %s" % f for f in analysis.suggested_fixes) or "(none)",
        )
    )
    try:
        code = debugger.debug(code, guidance, spec.solver)
    except DebuggerError as exc:
        history.append("rung 1 failed: Debugger returned no usable code (%s)" % exc)
        return None

    rung_dir.mkdir(parents=True, exist_ok=True)
    script = rung_dir / "simulate.py"
    script.write_text(code, encoding="utf-8")
    outcome = VariantOutcome(
        name="rung1", status="solver_error", code_path=str(script),
        resolutions=["guided by analyst diagnosis"],
    )
    if not budget.try_consume():
        outcome.status = "budget_exhausted"
        history.append("rung 1 not run: solver budget exhausted")
        _save_run(outcome, spec, data_dir, work_id)
        return outcome
    outcome.attempts = 1
    result = run_script(str(script), cwd=str(rung_dir), timeout_s=timeout_s)
    results_file = rung_dir / contract.results_file
    if not result.timed_out and result.exit_code == 0 and results_file.exists():
        outcome.status = "success"
        outcome.results_path = str(results_file)
    else:
        outcome.status = "timeout" if result.timed_out else "solver_error"
        outcome.error_tail = result.stderr_tail
        history.append(
            "rung 1 run failed (%s)%s"
            % (
                outcome.status,
                " — " + outcome.error_tail.strip().splitlines()[-1]
                if outcome.error_tail.strip() else "",
            )
        )
    _save_run(outcome, spec, data_dir, work_id)
    return outcome


def _analyze(
    analyst: ResultAnalyst,
    spec: SimulationSpec,
    contract: EvaluationContract,
    outcome: VariantOutcome,
    eval_stdout: str,
    config: Dict,
    data_dir: str,
    work_id: Optional[str],
    analysis_dir: Path,
) -> AnalysisReport:
    analysis_cfg = config.get("analysis", {})
    use_vision = analysis_cfg.get("use_vision", True)
    max_points = analysis_cfg.get("max_curve_points", 40)
    variant_dir = Path(outcome.code_path).parent if outcome.code_path else None
    analysis_dir.mkdir(parents=True, exist_ok=True)

    curve_checks = [
        c for c in contract.checks
        if not (c.kind == "numeric" and c.expected_value is not None)
    ]
    curve_texts: Dict[str, str] = {}
    images: List[Tuple[str, str]] = []
    for check in curve_checks:
        csv_path = variant_dir / ("%s.csv" % check.key) if variant_dir else None
        if csv_path is None or not csv_path.exists():
            curve_texts[check.key] = ""
            continue
        curve_texts[check.key] = read_curve_csv(str(csv_path), max_points) or ""
        if use_vision:
            png = render_curve_png(
                str(csv_path),
                str(analysis_dir / ("%s_%s.png" % (outcome.name, check.key))),
                check.key,
            )
            url = png_data_url(png) if png else None
            if url:
                images.append(("simulated %s" % check.key, url))

    if use_vision and work_id:
        pdf_path = _paper_pdf(data_dir, work_id)
        if pdf_path:
            seen_locations = set()
            for check in curve_checks:
                if not check.location or check.location in seen_locations:
                    continue
                seen_locations.add(check.location)
                page_png = find_figure_page_png(
                    pdf_path, check.location,
                    str(analysis_dir / ("paper_%s.png" % check.key)),
                )
                url = png_data_url(page_png) if page_png else None
                if url:
                    images.append(("paper page containing %s" % check.location, url))

    analysis = analyst.analyze(
        spec, contract, outcome.name, eval_stdout, curve_texts, images or None
    )
    report_path = analysis_dir / ("%s.json" % outcome.name)
    report_path.write_text(
        json.dumps(analysis.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if work_id:
        with Storage(data_dir) as storage:
            storage.save_analysis(work_id, outcome.name, analysis.verdict, str(report_path))
    return analysis


def _paper_pdf(data_dir: str, work_id: str) -> Optional[str]:
    with Storage(data_dir) as storage:
        for asset in storage.get_assets_for_work(work_id):
            if asset["format"] == "pdf" and Path(asset["path"]).exists():
                return asset["path"]
    return None


def _save_run(
    outcome: VariantOutcome, spec: SimulationSpec, data_dir: str, work_id: Optional[str]
) -> None:
    if not work_id:
        return
    import hashlib

    spec_sha = hashlib.sha256(spec.model_dump_json().encode("utf-8")).hexdigest()
    with Storage(data_dir) as storage:
        storage.save_run(work_id, spec_sha, spec.solver, outcome)


def _persist_report(report: StageFReport, session_dir: Path, data_dir: str) -> None:
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "result_verification_report.json").write_text(
        json.dumps(report.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
