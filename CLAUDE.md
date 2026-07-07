# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**KORANI** (KISTI Open Research Agentic Network Intelligence) — a multi-agent
AI framework that acts as a "co-scientist" for engineers and researchers in
Korea's semiconductor, battery, and manufacturing industries. Named after the
고라니 (Korean water deer, `Image/KORANI.jpg`).

KORANI does **not** try to replace the commercial simulation software these
industries already trust (TCAD, CFD, battery modeling tools). Instead it wraps
a team of specialized LLM agents around the tools people already use, in two
layers:

1. **Workflow layer** (current focus) — automates the tedious work around
   simulation: searching literature and past runs, generating input files,
   debugging solver errors.
2. **Optimization layer** (future) — an LLM-supervised optimization loop that
   searches for design parameters, calling the existing solver as a **black
   box**. Planned approach: classical sample-efficient methods (Bayesian
   optimization / DOE) propose candidate points; LLM agents interpret results,
   prune physically implausible regions, and reformulate the search when it
   stalls. **No evolutionary/tree search** — settled.

The goal is fewer wasted simulation runs and faster time-to-insight, **not** a
faster physics engine.

## First objective

The user asks in Korean to rebuild the Python code behind a research paper.
Agents find the paper (local DB or web), analyze it, and produce runnable
Python code using open-source solvers, verified against the paper's reported
results.

## Architecture: linear pipeline with escalation (decided)

A single linear pipeline, not an agent conversation and not a search tree.
Cheap and deterministic by default; extra LLM machinery only on failure.

Two entry modes at stage A/B:

- **Mode A — paper attached:** stage B collapses to "index the uploaded PDF";
  go straight to Extract.
- **Mode B — vague idea, no paper:** search-and-triage funnel — Search
  Planner expands the idea into multi-provider queries → literature pipeline
  returns candidates → Paper Triage ranks them by **reproducibility**
  (parameters tabulated? geometry fully specified? data/code available?
  physics solvable in DEVSIM/PyBaMM?) → **user picks the paper** → continue.

```
Korean request
   │
A. Understand   Interpreter parses the request (Korean→English boundary),
                  routes Mode A vs Mode B
B. Acquire      Mode A: index uploaded PDF
                  Mode B: search-and-triage funnel (user picks the paper)
C. Extract      ★ paper → SimulationSpec: equations, geometry, BCs, material
                  parameters, numerical settings, reported target results.
                  Ambiguous/missing fields are explicitly marked.
D. Contract     Evaluator drafts evaluate.py FROM the paper's reported
                  figures/tables → human approval checkpoint
E. Implement    Engineer writes DEVSIM/PyBaMM code from the spec;
                  Debugger fixes solver errors (bounded retries).
                  ★ Branch-on-ambiguity: if spec ambiguity survives a user
                  clarification attempt, produce 2-3 variants that resolve it
                  differently (best-of-N, single generation — NOT a tree).
F. Verify       execute → Result Analyst compares output to paper's results;
                  best variant wins
   │
   ├─ match    → return code + results to user (in Korean); log run
   └─ mismatch → escalation ladder:
        rung 1: Debugger retry guided by the analyst's diagnosis
        rung 2: ONE Proposer↔Critic debate round → up to 2 revised plans → E
        rung 3: stop and report honestly what failed and why
```

The **SimulationSpec** (stage C) is KORANI's core novel artifact — a typed,
human-reviewable intermediate representation connecting the literature side to
the engineering side. Design effort goes here first.

### Settled design decisions

- **No evolutionary tree search** (AgenticSciML Phase 3). Reproduction is a
  convergent task; the verify loop (E↔F) + bounded branch-on-ambiguity
  replaces the tree. Revisit only if Analysis Base data shows the linear
  path repeatedly failing on a problem class.
- **Minimal LLM calls before code exists.** No N-round debate by default;
  Proposer/Critic appear only at escalation rung 2, for one round.
- **No Co-STORM discourse engine.** We reuse KoCoScientist's `src/app/`
  literature pipeline only — not the rotating-expert conversation, moderator,
  mindmap, or LATS planner (those serve open-ended survey writing; even
  Mode B is a convergent funnel, not breadth exploration). Reversible: Co-STORM
  lives in a separate package and could return later as an optional
  literature-review module.
- **No report generation** for now. Output = working code + verification
  result + short summary.
- **No curated Knowledge Base at start.** SOTA LLMs know PyBaMM well and
  DEVSIM adequately; paper-specific knowledge comes from retrieval. Harvest a
  **failure playbook** automatically instead: when the Debugger resolves a
  recurring solver-error class, distill it into a small markdown entry
  (AgenticSciML KB format: Problem / Issue / Fix / Critical parameters) that
  future Debugger calls retrieve.
- **Budget the solver, not just tokens.** Per-task cap on simulation
  executions (adapt KoCoScientist's `BudgetConfig`). Branch-on-ambiguity
  variants count against this budget.
- **CLI only — not a product yet.** KoCoScientist's FastAPI + WebSocket web
  UI (its user-friendliness layer) is deliberately NOT carried over. The
  entry point is `python -m korani.cli`. Revisit a web layer only after the
  full A–F pipeline works end-to-end.
- **Language routing:** agents reason in English internally; Korean at the
  user boundary (KoCoScientist policy, adopted as-is).

## Models: free open-source first (current phase)

During the build/test phase all agents run on **free open-weight models**
served locally behind an **OpenAI-compatible endpoint** (vLLM / Ollama), so a
single `LLMClient` implementation covers every model. Once the architecture
is completely built, a commercial (non-free) model may be swapped in per-role
where benchmarks justify it — that is an explicit user decision made later,
and it must remain a `config.yaml` change only. Open models also carry a
data-sovereignty advantage (nothing leaves local infra) worth preserving.

- **KONI** (KISTI Open Natural Intelligence — `KISTI-KONI` org on Hugging
  Face, Llama-based, Korean science/tech domain-tuned; latest instruct:
  `KONI-Llama3.1-8B-Instruct-20241024`) powers the Korean boundary.
  Implemented in `src/korani/agents/interpreter.py`.
- Per-agent model assignment lives in `config.yaml` (model registry pattern:
  role → model name + endpoint). Never hardcode a model into agent code.
- The two **frontier-gap risk stages** are Spec Extractor (long-document
  structured extraction) and Result Analyst (vision over plots). Benchmark
  open models on a small eval set (~5 papers) before locking the assignment;
  if an open model underperforms there, escalate model size first, not
  pipeline complexity.

### Agent roster and default model tiers

| # | Agent | Stage | Default model tier (open-weight) |
|---|---|---|---|
| 1 | Interpreter | A | **KONI** (Korean boundary: intake, clarifying Q&A, output rendering) |
| 2 | Search Planner | B (Mode B) | small: KONI or Qwen3-8B |
| 3 | Paper Triage | B (Mode B) | mid: Qwen3-32B |
| 4 | Spec Extractor | C | strongest available (e.g. Qwen3-32B+ / DeepSeek-R1-class) — ⚠ risk stage |
| 5 | Evaluator | D | mid-strong: Qwen3-32B |
| 6 | Engineer | E | coding model: Qwen3-Coder / Qwen2.5-Coder-32B |
| 7 | Debugger | E | coding model, smaller ok |
| 8 | Result Analyst | F | **vision** model: Qwen2.5-VL/Qwen3-VL class — ⚠ risk stage |
| 9 | Proposer + Critic | escalation only | reasoning model (thinking mode) |

Deliberately absent: Moderator, Selector ensemble, Root Engineer baseline,
expert personas. Optional future role: Data Analyst (EDA on user-provided
measurement data), placeholder only.

Hardware reality: 8B models run on one 24 GB consumer GPU; 32B-class needs
~48 GB (or 2×24 GB, or 4-bit quantization); vision and coding models can share
a server with model swapping if concurrency is low.

## Persistence

Extend the KoCoScientist SQLite schema with a **`simulation_runs`** table:
inputs, solver + version, SimulationSpec hash, score, artifact paths, analyst
report link. This powers "search past experiments" and gives the future
optimization layer its memory. Analysis reports persist as files referenced
from the DB row (AgenticSciML's Analysis Base pattern).

## Reference codebases (read-only)

- **`../KoCoScientist_v2.1/`** — reference ONLY. Never modify it; all KORANI
  work happens inside the `KORANI/` folder.
  Borrow from `src/app/` (NOT `src/costorm/`):
  - SQLite DB (`works`, `provider_records`, `fulltext_assets` sha256-deduped,
    `chunks` with embeddings; async SQLAlchemy + aiosqlite, WAL).
  - Stage 0–3 orchestrator (`src/app/core/orchestrator.py`): query plan →
    parallel provider search (OpenAlex / Semantic Scholar / CORE) → dedup →
    rerank → OA resolve → download → parse → chunk → index.
  - Chunks triple-indexed: FAISS (vector) + BM25 (lexical) as the search
    layer, SQL as source of truth; hybrid retrieval in `search_in_documents`.
  - 5 cost-tiered agent tools (`src/app/retrieval_tools/tools.py`) — cheap
    metadata tools narrow candidates before expensive fulltext tools run.
  - Protocol interfaces (`LLMClient`, `Retriever`, `Embedder`, `Translator`)
    as swap/test seams.
  - Typed event stream (agent steps → WebSocket → UI "thinking" blocks) for
    transparency, if/when a UI is added.
- **`ReferencePaper/AgenticSciML.pdf`** — source of the evaluation contract,
  Engineer/Debugger/Result-Analyst roles, Analysis Base, and KB entry format.
- **`ReferencePaper/CrunchGPT.pdf`** — secondary reference.

Literature providers are OpenAlex / Semantic Scholar / CORE for now. Adding a
new source later (e.g. a domestic Korean database) is a provider-layer
change only — a new client in the `src/app/providers/` pattern — so nothing
in the architecture depends on this decision.

## Target domains & solvers

| Domain | Solver | Notes |
|---|---|---|
| Semiconductor (TCAD) | **DEVSIM** | open-source device simulator, Python API |
| Battery | **PyBaMM** | open-source battery modeling, Python |

The solver is always called as a black box. Do not train neural surrogates to
replace solver physics. PyBaMM runs take seconds and small DEVSIM
drift-diffusion runs seconds-to-minutes — running 2-3 candidate variants is
often cheaper than an extra LLM debate cycle.

## Non-negotiables

- Journal paper text is copyrighted. Don't store or reproduce full paper text
  beyond what's needed for retrieval; summarize/extract rather than copy.
- Korean technical terms must not be silently mistranslated or guessed at —
  flag ambiguous terms for human review instead of picking a translation.
- All KORANI work stays inside the `KORANI/` folder. `KoCoScientist_v2.1/`
  is reference material only — never edit it.
- Human approval checkpoint before the evaluation contract is locked in.
- Report verification outcomes honestly — a mismatch against the paper's
  results is reported as a mismatch, never papered over.
- Free open-weight models during the build phase; introducing a paid model
  later is the user's explicit decision and must never require code changes
  (config.yaml only).

## Implementation status

```
src/korani/
├── models.py                  # TaskSpec, PaperCandidate, Shortlist,
│                              #   ★ SimulationSpec (+ ParameterEntry,
│                              #   TargetResult, AmbiguousField)
├── config.py                  # config.yaml + env overrides (KORANI_LLM_BASE_URL,
│                              #   KORANI_LLM_API_KEY, KORANI_MODEL_<ROLE>)
├── llm.py                     # LLMClient protocol + OpenAICompatClient (vLLM/Ollama)
├── jsonutil.py                # shared defensive JSON extraction (all agents)
├── cli.py                     # python -m korani.cli "질문" [--paper x.pdf]
│                              #   [--interpret-only] [--pick N]
├── search.py                  # ✅ stage B: fan-out, DOI/title dedup+merge,
│                              #   heuristic pre-rank, candidate cap (no LLM)
├── stage_b.py                 # ✅ stage B assembly (run_stage_b)
├── stage_c.py                 # ✅ stage C assembly: acquire PDF → parse →
│                              #   extract spec → persist (run_stage_c)
├── fulltext.py                # ✅ PDF download (%PDF-verified, sha256),
│                              #   PyMuPDF text extraction, references trim
├── storage.py                 # ✅ SQLite (sqlite3+WAL): works (DOI upsert),
│                              #   fulltext_assets (sha256 dedup),
│                              #   simulation_specs; runs table joins at stage F
├── providers/                 # ✅ OpenAlex + Semantic Scholar (PaperProvider
│                              #   protocol; safe_search isolates failures;
│                              #   verify_ssl escape hatch for corp proxies)
└── agents/
    ├── interpreter.py         # ✅ stage A: KONI-powered Korean intake
    ├── search_planner.py      # ✅ stage B: query expansion (falls back to
    │                          #   interpreter queries on any LLM failure)
    ├── paper_triage.py        # ✅ stage B: batched reproducibility scoring;
    │                          #   user picks — never auto-selects
    └── spec_extractor.py      # ✅ stage C: paper text → SimulationSpec;
                               #   ⚠ RISK STAGE — honesty rules in prompt
tests/                         # 45 offline tests (stub LLM + httpx.MockTransport
                               #   + real tiny PDFs + real SQLite in tmp_path)
```

Conventions established by stage A — follow them for stages B–F:
- Agents take an `LLMClient` + model name in the constructor (DI); tests use
  stub clients, never a live server.
- Deterministic decisions (e.g. Mode A/B from paper attachment) are made in
  code, never delegated to the LLM.
- LLM JSON output is parsed defensively (`_extract_json` tolerates prose and
  markdown fences) and validated through pydantic models.
- Per-role models come from the `models:` registry in config.yaml.

Stage B notes:
- Triage scores REPRODUCIBILITY (0-10), not relevance; `solver_fit` maps each
  candidate to pybamm/devsim/none. One batched LLM call for all candidates.
- Metadata search is live-verified against OpenAlex; Semantic Scholar's
  anonymous pool rate-limits (429) under load — `safe_search` degrades
  gracefully. A backoff/retry is a known TODO, not a bug.

Stage C notes:
- The SimulationSpec (models.py) is the pipeline's contract: parameter values
  are strings copied verbatim WITH units (no silent conversion — the Engineer
  converts, so extraction errors stay visible to the human reviewer);
  `source: paper|assumed|missing` per parameter; `ambiguities[]` feeds both
  human review and stage E branch-on-ambiguity; `target_results[]` cites
  concrete figures/tables and becomes the Evaluator's input at stage D.
- Extractor honesty rules live in the system prompt: never invent values,
  missing → source="missing" + an ambiguities entry. Enum fields and work_id
  are normalized/stripped in code — the LLM cannot claim persistence identity.
- Paper text is trimmed (references cut, `extraction.max_chars` cap, default
  24k chars) to fit small local models. Chunked/two-pass extraction for long
  papers is a known TODO tied to the risk-stage benchmark.
- Persistence: data_dir/korani.db (works, fulltext_assets, simulation_specs) +
  papers/ (sha-named PDFs) + specs/{work_id}.json (human-reviewable copy).
- Mode B candidates without an OA pdf_url fail with clear guidance to obtain
  the PDF manually and rerun with --paper (copyright non-negotiable: KORANI
  never scrapes paywalled fulltext).

## Status

Architecture decided (linear pipeline, two entry modes, branch-on-ambiguity,
open-weight-first model policy, CLI only). Stages A (Interpreter/KONI),
B (Search Planner → OpenAlex/S2 → Paper Triage → user pick), and
C (PDF acquire/parse → Spec Extractor → SQLite persistence) are implemented
and tested; the CLI runs A→B→C end-to-end. Next steps: stage D (Evaluator
drafts evaluate.py from the spec's target_results → human approval); stage E
Engineer/Debugger with DEVSIM/PyBaMM solver adapters and error taxonomies;
benchmark open models on the two risk stages (Spec Extractor, Result
Analyst) with ~5 real papers.
