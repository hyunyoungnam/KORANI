"""KORANI CLI — stages A + B, for testing against a local model server.

Usage:
    # Mode B: vague idea → interpret → search → triage shortlist → you pick
    python -m korani.cli "저온에서 배터리 열화를 줄이는 설계를 시뮬레이션하고 싶은데..."

    # Mode A: paper attached → interpret only (stage C not yet implemented)
    python -m korani.cli "이 논문의 소자 시뮬레이션을 재현해줘" --paper paper.pdf

    # Stop after stage A
    python -m korani.cli "..." --interpret-only
"""

from __future__ import annotations

import argparse
import sys

from korani.agents.interpreter import InterpreterError, build_interpreter
from korani.agents.paper_triage import TriageError
from korani.agents.spec_extractor import SpecExtractionError
from korani.config import load_config
from korani.llm import LLMError
from korani.models import Shortlist, SimulationSpec, TaskSpec


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
    print("(stage D — 평가 스크립트 작성 + 승인 — 는 아직 구현되지 않았습니다)\n")


def _run_stage_c_and_print(spec, config, candidate=None, pdf_path=None) -> int:
    from korani.fulltext import FulltextError
    from korani.stage_c import StageCError, run_stage_c

    print("논문 분석 중 (PDF 파싱 + SimulationSpec 추출)...", file=sys.stderr)
    try:
        simspec, work_id, spec_file = run_stage_c(
            spec, config, candidate=candidate, pdf_path=pdf_path
        )
    except (StageCError, FulltextError) as exc:
        print(f"\n[Stage C] {exc}", file=sys.stderr)
        return 1
    except LLMError as exc:
        print(f"\n[LLM error] {exc}", file=sys.stderr)
        return 1
    except SpecExtractionError as exc:
        print(f"\n[Spec parse error] {exc}", file=sys.stderr)
        if exc.raw_output:
            print(f"\n--- raw model output ---\n{exc.raw_output}", file=sys.stderr)
        return 2
    _print_simspec(simspec, spec_file)
    return 0


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
        prog="korani", description="KORANI — Korean co-scientist (stages A-B test CLI)"
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

    # ── Mode A: local PDF → stage C directly ──
    if spec.mode == "A":
        return _run_stage_c_and_print(spec, config, pdf_path=spec.paper_path)

    # ── Stage B: search and triage ──
    from korani.stage_b import run_stage_b  # deferred: pulls in httpx

    print("논문 검색 중 (OpenAlex, Semantic Scholar)...", file=sys.stderr)
    try:
        shortlist = run_stage_b(spec, config)
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
    return _run_stage_c_and_print(spec, config, candidate=picked)


if __name__ == "__main__":
    sys.exit(main())
