<p align="center">
  <img src="Image/KORANI.jpg" alt="KORANI(고라니)" width="600">
</p>

<h1 align="center">KORANI</h1>
<p align="center"><b>KISTI Open Research Agentic Network Intelligence (KORANI)</b></p>
<p align="center">한국 반도체·배터리 연구자를 위한 AI 코사이언티스트</p>

---

## What is KORANI?

KORANI is a multi-agent AI framework that acts as a **co-scientist** for
engineers and researchers in Korea's semiconductor, battery, and manufacturing
industries. The library name comes from KORANI (고라니), an animal found
almost nowhere else on Earth but thriving in Korea.

It does **not** replace the simulation software these industries already trust
(TCAD, CFD, battery modeling tools). Instead, it wraps a team of specialized
LLM agents around the tools people already use:

- **Workflow layer** — automates the tedious work around simulation:
  searching literature and past experiments, generating input files,
  debugging solver errors.
- **Optimization layer** *(planned)* — an LLM-supervised loop that searches
  design parameters, calling the existing solver as a black box.

**The goal is fewer wasted simulation runs and faster time-to-insight — not a
faster physics engine.**

## What can it do? (first objective)

Ask in Korean to rebuild the Python code behind a research paper:

> "이 논문의 소자 시뮬레이션을 DEVSIM으로 재현해줘"

KORANI then:

1. Finds the paper — in its local literature database or on the web
2. Extracts a structured **SimulationSpec** — equations, geometry, boundary
   conditions, material parameters, and the paper's reported results
3. Drafts an evaluation script from the paper's reported figures/tables —
   **you approve it before anything runs**
4. Writes runnable Python code using open-source solvers
5. Executes, compares the output against the paper, and iterates until it
   matches — or tells you honestly why it doesn't

| Domain | Solver |
|---|---|
| Semiconductor / TCAD | [DEVSIM](https://devsim.org) |
| Battery | [PyBaMM](https://pybamm.org) |

## How it works

A **linear pipeline with an escalation ladder** — cheap and deterministic by
default, extra LLM machinery only when a step fails. No agent debate before
code exists, no evolutionary search.

```
Korean request
   │
   A. Understand    task spec (Korean → English internally)
   B. Acquire       literature search: local DB or OpenAlex / Semantic Scholar / CORE
   C. Extract       paper → SimulationSpec (human-reviewable)
   D. Contract      evaluate.py drafted from the paper's results → human approval
   E. Implement     Engineer writes DEVSIM/PyBaMM code; Debugger fixes errors
   F. Verify        run → compare against the paper's reported results
   │
   ├─ match     →  code + results returned in Korean; run logged to the DB
   └─ mismatch  →  ① guided debug retry  ② one Proposer↔Critic round  ③ honest failure report
```

Design lineage:

- Multi-agent roles (Evaluator, Engineer, Debugger, Result Analyst) and the
  evaluation-contract idea come from **AgenticSciML**
  (Jiang & Karniadakis, *npj Artificial Intelligence*, 2026).
- The literature search pipeline (SQLite + FAISS + BM25 hybrid retrieval,
  cost-tiered search tools, Korean/English language routing) follows the
  **KoCoScientist** project.

## Models: free & open-source first

While the architecture is being built, KORANI runs on **free open-weight
LLMs, self-hosted** — zero token cost, and no user/paper data leaves your
machine. The LLM client is provider-agnostic (any OpenAI-compatible
endpoint), so a commercial model can be swapped in later through
`config.yaml` alone, once the architecture is complete and testing shows
where a stronger model is worth paying for.

- **[KONI](https://huggingface.co/KISTI-KONI)** (KISTI Open Natural
  Intelligence — KISTI's Korean science/tech LLM, Llama-based) handles the
  Korean user boundary — implemented now
- Qwen3 / Qwen3-Coder / Qwen2.5-VL-class models are planned for extraction,
  coding, and plot analysis
- Per-agent model assignment lives in `config.yaml`; agent code never
  hardcodes a model

KORANI is currently a **CLI, not a product** — no web server, no UI.
(KoCoScientist's FastAPI + WebSocket web interface is a user-friendliness
layer we deliberately haven't carried over yet.)

## Try it: ask in Korean

**1. Install** (Python ≥ 3.9)

```bash
cd KORANI
pip install -e .
```

**2. Serve KONI locally** — either server works; both expose an
OpenAI-compatible endpoint:

```bash
# Option 1 — Ollama (easiest): pull a community KONI GGUF from Hugging Face
ollama pull hf.co/RichardErkhov/KISTI-KONI_-_KONI-Llama3-8B-Instruct-20240729-gguf:Q4_K_M
# then set `models.interpreter` in config.yaml to that model name

# Option 2 — vLLM (needs a GPU, serves the latest original weights)
vllm serve KISTI-KONI/KONI-Llama3.1-8B-Instruct-20241024
# then set llm.base_url to http://localhost:8000/v1 in config.yaml
```

> No KONI download handy? Any Korean-capable open model works for a first
> smoke test — e.g. `ollama pull exaone3.5` — just change
> `models.interpreter` in `config.yaml`.

**3. Ask** — vague idea, no paper (Mode B):

```bash
python -m korani.cli "저온에서 배터리 열화를 줄이는 설계를 시뮬레이션하고 싶은데, 어떤 논문을 재현하면 좋을까?"
```

```
============================================================
  KORANI  |  mode B  |  domain: battery
============================================================

[응답]
저온 환경에서의 배터리 열화 저감 설계 시뮬레이션 과제로 이해했습니다.

[Task (EN)]
Design and simulate strategies to reduce battery degradation at low temperature

[Search queries]
  - low temperature lithium-ion battery degradation model
  - battery aging simulation cold climate PyBaMM

[질문]
  - 대상 셀 화학조성(NMC811 등)이 정해져 있나요?
```

Or attach a paper (Mode A — search is skipped):

```bash
python -m korani.cli "이 논문의 소자 시뮬레이션을 DEVSIM으로 재현해줘" --paper paper.pdf
```

In **Mode B** (no paper attached), KORANI continues past interpretation:
the Search Planner expands your question into English queries, OpenAlex and
Semantic Scholar are searched in parallel (free, no API key), duplicates are
merged, and the Paper Triage agent ranks candidates by **reproducibility** —
can this paper's simulation actually be rebuilt in PyBaMM/DEVSIM and verified?
You pick from the shortlist:

```
------------------------------------------------------------
  후보 논문 (reproducibility 순위)
------------------------------------------------------------

  [1] (9.0/10, pybamm, PDF ✓) Lithium-ion battery degradation: how to model it
      O'Kane, S., Ai, W., Madabattula, G. et al. (2022) | citations: 349
      → Full DFN+degradation parameter set and validation plots; PyBaMM-native.
  ...
재현할 논문 번호를 선택하세요 (1-5, 건너뛰려면 Enter):
```

After you pick a paper (or attach one with `--paper`), **stage C** takes over:
the PDF is downloaded (open-access only) or read locally, parsed, and the
Spec Extractor produces a **SimulationSpec** — the equations, geometry,
parameters (with anything missing explicitly marked, never invented),
operating conditions, and the figures/tables the reproduction will be
verified against. The spec is saved to `data/specs/` and SQLite for review.

**4. Run the tests** (offline, no model server needed):

```bash
python -m pytest tests/ -q
```

> Currently implemented: **stage A (Interpreter)**, **stage B
> (search-and-triage)**, and **stage C (SimulationSpec extraction)**.
> Stages D–F are under development — see `CLAUDE.md` for the full
> architecture and design decisions.
> Tip: set `search.mailto` in `config.yaml` (OpenAlex polite pool); behind a
> corporate proxy with SSL inspection, set `SSL_CERT_FILE` to your company CA
> or, for local testing only, `search.verify_ssl: false`.

## Repository layout

```
KORANI/
├── CLAUDE.md                 # architecture & design decisions (source of truth)
├── README.md
├── config.yaml               # LLM endpoint + per-agent model registry
├── pyproject.toml
├── requirements.txt
├── src/korani/
│   ├── agents/
│   │   ├── interpreter.py    # stage A: Korean intake (KONI)
│   │   ├── search_planner.py # stage B: query expansion
│   │   ├── paper_triage.py   # stage B: reproducibility ranking
│   │   └── spec_extractor.py # stage C: paper → SimulationSpec
│   ├── providers/            # OpenAlex + Semantic Scholar clients
│   ├── search.py             # fan-out, dedup, pre-rank (no LLM)
│   ├── stage_b.py            # stage B assembly
│   ├── stage_c.py            # stage C assembly (download/parse/extract/persist)
│   ├── fulltext.py           # PDF download + PyMuPDF text extraction
│   ├── storage.py            # SQLite: works, fulltext_assets, simulation_specs
│   ├── cli.py                # python -m korani.cli "질문"
│   ├── config.py             # config.yaml + env overrides
│   ├── llm.py                # provider-agnostic OpenAI-compatible client
│   └── models.py             # TaskSpec, Shortlist, SimulationSpec, ...
├── tests/                    # 45 offline tests (stub LLM + mock HTTP + tiny PDFs)
├── Image/
│   └── KORANI.jpg            # the mascot
└── ReferencePaper/
    ├── AgenticSciML.pdf      # multi-agent framework reference
    └── CrunchGPT.pdf         # secondary reference
```

## References

- Jiang, Q. & Karniadakis, G. *AgenticSciML: collaborative multi-agent systems
  for emergent discovery in scientific machine learning.* npj Artif. Intell.
  (2026). https://doi.org/10.1038/s44387-026-00102-5
- KoCoScientist — Co-STORM-based literature search & research discourse system
  (internal reference codebase)
- [DEVSIM](https://devsim.org) — open-source TCAD device simulator
- [PyBaMM](https://pybamm.org) — Python Battery Mathematical Modelling
