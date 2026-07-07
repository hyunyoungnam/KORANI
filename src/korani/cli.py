"""KORANI CLI — the full A–F pipeline, for testing against a local model server.

Usage:
    # Mode B: vague idea → interpret → search → triage shortlist → you pick
    python -m korani.cli "저온에서 배터리 열화를 줄이는 설계를 시뮬레이션하고 싶은데..."

    # Mode A: paper attached → spec → contract (approval) → code + run
    python -m korani.cli "이 논문의 소자 시뮬레이션을 재현해줘" --paper paper.pdf

    # Stop after stage A
    python -m korani.cli "..." --interpret-only

    # Non-interactive runs: pick from the shortlist / approve the contract
    python -m korani.cli "..." --pick 1 --approve-contract
"""

from __future__ import annotations

import argparse
import sys

from korani.agents.evaluator import EvaluatorError
from korani.agents.interpreter import InterpreterError, build_interpreter
from korani.agents.paper_triage import TriageError
from korani.agents.spec_extractor import SpecExtractionError
from korani.config import load_config
from korani.llm import LLMError
from korani.models import EvaluationContract, Shortlist, SimulationSpec, TaskSpec


def _print_spec(spec: TaskSpec) -> None:
    print()
    print("=" * 60)
    print(f"  KORANI  |  mode {spec.mode}  |  domain: {spec.domain}")
    print("=" * 60)
    print(f"\n[응답]\n{spec.reply_ko}")
    print(f"\n[Task (EN)]\n{spec.task_en}")
    if spec.paper_path:
        print(f"\n[Paper]\n{spec.paper_path}")
    if spec.search_queries_en:
        print("\n[Search queries]")
        for q in spec.search_queries_en:
            print(f"  - {q}")
    if spec.ambiguous_terms:
        print("\n[⚠ 확인 필요한 용어]")
        for t in spec.ambiguous_terms:
            candidates = ", ".join(t.candidates_en) or "?"
            print(f"  - {t.term_ko} → {candidates} ({t.reason})")
    if spec.clarifying_questions_ko:
        print("\n[질문]")
        for q in spec.clarifying_questions_ko:
            print(f"  - {q}")
    print()


def _print_shortlist(shortlist: Shortlist) -> None:
    print("-" * 60)
    print("  후보 논문 (reproducibility 순위)")
    print("-" * 60)
    if not shortlist.entries:
        print("\n검색 결과가 없습니다. 질문을 더 구체적으로 해보세요.\n")
        return
    for rank, entry in enumerate(shortlist.entries, start=1):
        c = entry.candidate
        authors = ", ".join(c.authors[:3]) + (" et al." if len(c.authors) > 3 else "")
        pdf = "PDF ✓" if c.pdf_url else "PDF ✗"
        print(f"\n  [{rank}] ({entry.score:.1f}/10, {entry.solver_fit}, {pdf}) {c.title}")
        print(f"      {authors} ({c.year}) | citations: {c.citation_count}")
        if entry.rationale:
            print(f"      → {entry.rationale}")
        if c.doi:
            print(f"      doi: {c.doi}")
    print()


def _print_simspec(spec: SimulationSpec, spec_file: str) -> None:
    print("-" * 60)
    print(f"  SimulationSpec  |  solver: {spec.solver}  |  domain: {spec.domain}")
    print("-" * 60)
    print(f"\n[논문]\n{spec.title}")
    print(f"\n[Model]\n{spec.model_summary}")
    if spec.geometry:
        print(f"\n[Geometry]\n{spec.geometry}")
    n_missing = sum(1 for p in spec.parameters if p.source == "missing")
    print(f"\n[Parameters] {len(spec.parameters)} extracted ({n_missing} missing)")
    for p in spec.parameters[:8]:
        mark = " ⚠ MISSING" if p.source == "missing" else ""
        print(f"  - {p.name}: {p.value or '?'} {p.units or ''}{mark}")
    if len(spec.parameters) > 8:
        print(f"  ... (+{len(spec.parameters) - 8} more, see spec file)")
    if spec.operating_conditions:
        print("\n[Operating conditions]")
        for c in spec.operating_conditions:
            print(f"  - {c}")
    if spec.target_results:
        print("\n[검증 목표 (target results)]")
        for t in spec.target_results:
            loc = f" ({t.location})" if t.location else ""
            print(f"  - {t.description}{loc}")
    if spec.ambiguities:
        print("\n[⚠ 모호한 항목 — 확인 필요]")
        for a in spec.ambiguities:
            print(f"  - {a.field}: {a.issue}")
            if a.candidates:
                print(f"      후보: {', '.join(a.candidates)}")
    print(f"\n저장됨: {spec_file}")


def _print_contract(contract: EvaluationContract, script_path: str) -> None:
    print("-" * 60)
    print(f"  평가 계약 (stage D)  |  checks: {len(contract.checks)}  |  {contract.status}")
    print("-" * 60)
    for c in contract.checks:
        loc = f" ({c.location})" if c.location else ""
        if c.kind == "numeric" and c.expected_value is not None:
            print(f"  [numeric] {c.key}{loc}: {c.description}")
            print(f"      기대값 {c.expected_text or c.expected_value} ± {c.rel_tol * 100:.0f}%")
        else:
            print(f"  [{c.kind}] {c.key}{loc}: {c.description} → stage F Result Analyst")
    print(f"\nevaluate.py: {script_path}")


def _run_spec_extraction_and_print(spec, config, candidate=None, pdf_path=None):
    """Returns (exit_code, simspec, work_id); simspec is None on failure."""
    from korani.fulltext import FulltextError
    from korani.spec_extraction import SpecExtractionStageError, run_spec_extraction

    print("논문 분석 중 (PDF 파싱 + SimulationSpec 추출)...", file=sys.stderr)
    try:
        simspec, work_id, spec_file = run_spec_extraction(
            spec, config, candidate=candidate, pdf_path=pdf_path
        )
    except (SpecExtractionStageError, FulltextError) as exc:
        print(f"\n[Stage C] {exc}", file=sys.stderr)
        return 1, None, None
    except LLMError as exc:
        print(f"\n[LLM error] {exc}", file=sys.stderr)
        return 1, None, None
    except SpecExtractionError as exc:
        print(f"\n[Spec parse error] {exc}", file=sys.stderr)
        if exc.raw_output:
            print(f"\n--- raw model output ---\n{exc.raw_output}", file=sys.stderr)
        return 2, None, None
    _print_simspec(simspec, spec_file)
    return 0, simspec, work_id


def _confirm_contract(approve_flag: bool) -> bool:
    """Human approval checkpoint (non-negotiable). ``--approve-contract`` is
    an explicit human decision made on the command line; otherwise ask."""
    if approve_flag:
        return True
    if not sys.stdin.isatty():
        return False
    try:
        answer = input("평가 계약을 승인하시겠습니까? evaluate.py 검토 후 입력 (y/N): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return answer in ("y", "yes")


def _run_evaluation_contract_and_print(simspec, work_id, config, approve_flag: bool):
    """Returns (exit_code, contract); contract is None on failure and stays
    a draft unless the human approved it."""
    from korani.evaluation_contract import (
        EvaluationContractError,
        approve_contract,
        run_evaluation_contract,
    )

    print("평가 계약 작성 중 (evaluate.py 초안)...", file=sys.stderr)
    try:
        contract, script_path, _ = run_evaluation_contract(simspec, work_id, config)
    except EvaluationContractError as exc:
        print(f"\n[Stage D] {exc}", file=sys.stderr)
        return 1, None
    except LLMError as exc:
        print(f"\n[LLM error] {exc}", file=sys.stderr)
        return 1, None
    except EvaluatorError as exc:
        print(f"\n[Evaluator parse error] {exc}", file=sys.stderr)
        if exc.raw_output:
            print(f"\n--- raw model output ---\n{exc.raw_output}", file=sys.stderr)
        return 2, None
    _print_contract(contract, script_path)
    if _confirm_contract(approve_flag):
        approve_contract(contract, config)
        print("\n계약이 승인되었습니다 (approved). 이 계약으로 검증합니다.")
    else:
        print("\n계약은 초안(draft)으로 저장되었습니다. evaluate.py를 검토한 뒤")
        print("--approve-contract 옵션으로 승인하세요. stage E는 승인된 계약만 사용합니다.\n")
    return 0, contract


def _collect_ambiguity_resolutions(simspec) -> dict:
    """Stage E user clarification attempt: ambiguities answered here apply to
    every variant; the rest fan out via branch-on-ambiguity."""
    if not simspec.ambiguities or not sys.stdin.isatty():
        return {}
    print("모호한 항목을 지금 확정할 수 있습니다 (Enter = 건너뛰고 변형(variant)으로 처리):")
    resolutions = {}
    for a in simspec.ambiguities:
        candidates = f" [후보: {', '.join(a.candidates)}]" if a.candidates else ""
        try:
            answer = input(f"  {a.field} — {a.issue}{candidates}\n  → ").strip()
        except (EOFError, KeyboardInterrupt):
            return resolutions
        if answer:
            resolutions[a.field] = answer
    return resolutions


def _print_execution_report(report) -> None:
    print("-" * 60)
    print(
        f"  실행 결과 (stage E)  |  solver: {report.solver}"
        f"  |  budget: {report.solver_runs_used}/{report.solver_runs_budget}"
    )
    print("-" * 60)
    for v in report.variants:
        mark = "✓" if v.status == "success" else "✗"
        print(f"\n  [{mark}] {v.name}: {v.status} (실행 {v.attempts}회)")
        for r in v.resolutions:
            print(f"      가정: {r}")
        if v.code_path:
            print(f"      코드: {v.code_path}")
        if v.status == "success" and v.eval_exit is not None:
            verdict = "일치" if v.eval_exit == 0 else "불일치"
            print(
                f"      수치 검증: {verdict} — {v.eval_passed} 통과, {v.eval_failed} 실패,"
                f" {v.eval_deferred} 건은 Result Analyst 확인 대상"
            )
        elif v.error_tail:
            tail = v.error_tail.strip().splitlines()[-3:]
            for line in tail:
                print(f"      | {line}")
    print()


def _print_verification_report(report) -> None:
    print("-" * 60)
    print(
        f"  검증 결과 (stage F)  |  rungs: {report.rungs_used}"
        f"  |  budget: {report.solver_runs_used}/{report.solver_runs_budget}"
    )
    print("-" * 60)
    for line in report.history:
        print(f"  · {line}")
    if report.analysis is not None:
        a = report.analysis
        vision = " (vision)" if a.used_vision else " (text-only)"
        print(f"\n  Result Analyst{vision}: {a.verdict}")
        if a.diagnosis:
            print(f"    진단: {a.diagnosis}")
        for c in a.curves:
            print(f"    곡선 {c.key}: {c.verdict} — {c.comment}")
    print()
    if report.verdict == "match":
        print(f"✅ 재현 성공 — {report.final_variant} 변형이 논문 결과와 일치합니다.")
    elif report.verdict == "mismatch":
        print("❌ 불일치 — 코드는 실행되지만 논문 결과와 일치하지 않습니다.")
        print("   위 이력이 시도한 내용 전부입니다. 산출물을 직접 검토해 주세요.")
    else:
        print("❌ 실패 — 완전한 실행을 얻지 못했습니다. 위 이력을 검토해 주세요.")
    print()


def _run_engineer_debugger_and_print(simspec, contract, config, budget):
    """Returns (exit_code, report); report is None on failure."""
    from korani.agents.debugger import DebuggerError
    from korani.agents.engineer import EngineerError
    from korani.engineer_debugger import EngineerDebuggerError, run_engineer_debugger

    resolutions = _collect_ambiguity_resolutions(simspec)
    print("코드 생성 및 실행 중 (Engineer/Debugger, solver budget 적용)...", file=sys.stderr)
    try:
        report = run_engineer_debugger(
            simspec, contract, config, user_resolutions=resolutions, budget=budget
        )
    except EngineerDebuggerError as exc:
        print(f"\n[Stage E] {exc}", file=sys.stderr)
        return 1, None
    except LLMError as exc:
        print(f"\n[LLM error] {exc}", file=sys.stderr)
        return 1, None
    except (EngineerError, DebuggerError) as exc:
        print(f"\n[Stage E parse error] {exc}", file=sys.stderr)
        raw = getattr(exc, "raw_output", "")
        if raw:
            print(f"\n--- raw model output ---\n{raw}", file=sys.stderr)
        return 2, None
    _print_execution_report(report)
    return 0, report


def _run_result_verification_and_print(simspec, contract, e_report, config, budget) -> int:
    from korani.result_verification import ResultVerificationError, run_result_verification

    print("결과 분석 중 (Result Analyst + escalation ladder)...", file=sys.stderr)
    try:
        report = run_result_verification(simspec, contract, e_report, config, budget=budget)
    except ResultVerificationError as exc:
        print(f"\n[Stage F] {exc}", file=sys.stderr)
        return 1
    except LLMError as exc:
        print(f"\n[LLM error] {exc}", file=sys.stderr)
        return 1
    _print_verification_report(report)
    return 0


def _run_remaining_pipeline(spec, config, approve_flag: bool, candidate=None, pdf_path=None) -> int:
    """From spec extraction through result verification (stages C-F)."""
    code, simspec, work_id = _run_spec_extraction_and_print(
        spec, config, candidate=candidate, pdf_path=pdf_path
    )
    if simspec is None:
        return code
    code, contract = _run_evaluation_contract_and_print(simspec, work_id, config, approve_flag)
    if contract is None:
        return code
    if contract.status != "approved":
        return 0  # draft saved; stage E requires approval (checkpoint)

    # One solver budget spans stage E and stage F's escalation ladder.
    from korani.engineer_debugger import SolverBudget

    budget = SolverBudget(config.get("budget", {}).get("max_solver_runs", 6))
    code, e_report = _run_engineer_debugger_and_print(simspec, contract, config, budget)
    if e_report is None:
        return code
    return _run_result_verification_and_print(simspec, contract, e_report, config, budget)


def _pick_from_shortlist(shortlist: Shortlist, pick: int = None) -> int:
    """Return the chosen 1-based rank, or 0 for no selection."""
    n = len(shortlist.entries)
    if pick is not None:
        return pick if 1 <= pick <= n else 0
    if not sys.stdin.isatty():
        return 0
    try:
        answer = input(f"재현할 논문 번호를 선택하세요 (1-{n}, 건너뛰려면 Enter): ").strip()
    except (EOFError, KeyboardInterrupt):
        return 0
    if answer.isdigit() and 1 <= int(answer) <= n:
        return int(answer)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="korani", description="KORANI — Korean co-scientist (stages A-F test CLI)"
    )
    parser.add_argument("question", help="연구 요청 (한국어)")
    parser.add_argument("--paper", default=None, help="논문 PDF 경로 (Mode A)")
    parser.add_argument("--config", default=None, help="config.yaml 경로")
    parser.add_argument(
        "--interpret-only", action="store_true", help="stage A까지만 실행"
    )
    parser.add_argument(
        "--pick", type=int, default=None, help="shortlist에서 비대화식으로 선택할 번호"
    )
    parser.add_argument(
        "--approve-contract",
        action="store_true",
        help="stage D 평가 계약을 비대화식으로 승인 (검토했다는 명시적 결정)",
    )
    args = parser.parse_args(argv)

    # Windows console may default to cp949; force UTF-8 for Korean output.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    config = load_config(args.config)
    interpreter = build_interpreter(config)
    print(
        f"Interpreter model: {interpreter.model} @ {config['llm']['base_url']}",
        file=sys.stderr,
    )

    try:
        spec = interpreter.interpret(args.question, paper_path=args.paper)
    except LLMError as exc:
        print(f"\n[LLM error] {exc}", file=sys.stderr)
        return 1
    except InterpreterError as exc:
        print(f"\n[Parse error] {exc}", file=sys.stderr)
        if exc.raw_output:
            print(f"\n--- raw model output ---\n{exc.raw_output}", file=sys.stderr)
        return 2

    _print_spec(spec)

    if args.interpret_only:
        return 0

    # ── Mode A: local PDF → stages C→D→E→F directly ──
    if spec.mode == "A":
        return _run_remaining_pipeline(
            spec, config, args.approve_contract, pdf_path=spec.paper_path
        )

    # ── Stage B: search and triage ──
    from korani.search_and_triage import run_search_and_triage  # deferred: pulls in httpx

    print("논문 검색 중 (OpenAlex, Semantic Scholar)...", file=sys.stderr)
    try:
        shortlist = run_search_and_triage(spec, config)
    except LLMError as exc:
        print(f"\n[LLM error] {exc}", file=sys.stderr)
        return 1
    except TriageError as exc:
        print(f"\n[Triage parse error] {exc}", file=sys.stderr)
        if exc.raw_output:
            print(f"\n--- raw model output ---\n{exc.raw_output}", file=sys.stderr)
        return 2

    _print_shortlist(shortlist)
    if not shortlist.entries:
        return 0

    chosen = _pick_from_shortlist(shortlist, pick=args.pick)
    if not chosen:
        return 0
    picked = shortlist.entries[chosen - 1].candidate
    print(f"\n선택됨: {picked.title}\n")
    return _run_remaining_pipeline(spec, config, args.approve_contract, candidate=picked)


if __name__ == "__main__":
    sys.exit(main())
