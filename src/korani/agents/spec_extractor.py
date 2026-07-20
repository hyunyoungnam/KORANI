"""Spec Extractor agent — stage C.

Turns parsed paper text into a SimulationSpec, KORANI's core artifact.
This is a flagged RISK STAGE (CLAUDE.md): a wrong parameter here poisons
everything downstream, so the prompt's honesty rules mirror the project
non-negotiables — missing values are marked missing, never invented, and
every uncertainty lands in ``ambiguities`` for human review and stage E
branch-on-ambiguity.

Papers longer than the per-call character budget get a chunked two-pass
extraction: pass 1 extracts a partial spec from each overlapping chunk,
pass 2 merges the partials in one more LLM call. If the merge call fails,
``merge_partial_specs`` merges deterministically in code instead — cruder on
free-text fields, but it never invents or drops a value.
"""

from __future__ import annotations

from collections import Counter
import re
from typing import List, Optional, Tuple

from korani.fulltext import chunk_text
from korani.jsonutil import extract_json_object
from korani.llm import LLMClient, LLMError
from korani.models import ParameterEntry, SimulationSpec, TaskSpec

SYSTEM_PROMPT = """\
You are the Spec Extractor agent of KORANI. Read the research paper text and \
produce a SimulationSpec JSON so an engineer agent can REPRODUCE the paper's \
simulation with DEVSIM for semiconductor TCAD (drift-diffusion class device \
simulation).

HONESTY RULES (critical - violating them poisons the whole pipeline):
1. NEVER invent a parameter value. If it is not stated in the text, set its \
"value" to null, "source" to "missing", and add an entry to "ambiguities".
2. Copy values EXACTLY as written, with units, into strings (e.g. \
"1.2e-14 m^2 s^-1"). Do not convert units.
3. "target_results" must cite concrete reported results (figure/table \
numbers) that a reproduction can be verified against.
4. Anything underspecified (mesh, tolerances, unclear boundary conditions, \
parameter values only shown in plots) goes into "ambiguities" with candidate \
resolutions if you can think of any.

Respond with ONE JSON object and NOTHING else:
{
 "title": "<paper title>",
 "domain": "semiconductor" | "unknown",
 "solver": "devsim" | "none",
 "model_summary": "<one/two sentences describing the device model>",
 "governing_equations": ["<name/description of each equation or submodel>"],
 "geometry": "<domain geometry / cell format / device structure>",
 "materials": ["<semiconductor, dielectric, and contact materials>"],
 "parameters": [{"name": ..., "symbol": ..., "value": ..., "units": ...,
                 "source": "paper"|"assumed"|"missing", "notes": ...}],
 "operating_conditions": ["<bias sweep / temperature / illumination ...>"],
 "numerical_settings": ["<mesh, solver tolerances, timestep - if given>"],
 "target_results": [{"description": ..., "location": "Figure 3",
                     "quantity": ..., "value": ...}],
 "ambiguities": [{"field": ..., "issue": ..., "candidates": [...]}]
}
"""

MERGE_PROMPT = """\
You are the Spec Extractor agent of KORANI. Partial SimulationSpec JSON \
objects were extracted from consecutive parts of ONE research paper. Merge \
them into a single SimulationSpec JSON with the same schema.

MERGE RULES:
1. Deduplicate parameters describing the same physical quantity. When \
duplicates disagree, prefer source "paper" with a concrete value, and record \
the disagreement in that parameter's "notes".
2. A parameter marked "missing" in one part but given a concrete "paper" \
value in another part is NOT missing — keep the concrete value and drop any \
"ambiguities" entry that the other part resolves.
3. NEVER invent values — only reorganize what the partial specs contain.
4. Union the list fields without duplicates; keep every distinct \
target_result with its figure/table citation.

Respond with ONE JSON object (the same SimulationSpec schema as the partial \
specs) and NOTHING else.
"""


class SpecExtractionError(RuntimeError):
    def __init__(self, message: str, raw_output: str = ""):
        super().__init__(message)
        self.raw_output = raw_output


class SpecExtractor:
    def __init__(self, client: LLMClient, model: str, temperature: float = 0.1, max_tokens: int = 4096):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def extract(
        self,
        task: TaskSpec,
        paper_text: str,
        max_chars: int = 24000,
        chunk_overlap: int = 2000,
        max_chunks: int = 6,
    ) -> SimulationSpec:
        """Single-pass extraction, or chunked two-pass for long papers."""
        chunks = chunk_text(paper_text, max_chars, overlap=chunk_overlap)
        if len(chunks) > max_chunks:  # cost cap — the tail is usually appendix
            chunks = chunks[:max_chunks]
        if len(chunks) == 1:
            return self._extract_call(task, chunks[0])

        # Pass 1: partial spec per chunk. A parse failure on one chunk must
        # not sink the paper; an LLMError (endpoint down) is fatal and
        # propagates immediately.
        partials: List[SimulationSpec] = []
        last_error: Optional[SpecExtractionError] = None
        for i, chunk in enumerate(chunks, start=1):
            try:
                partials.append(self._extract_call(task, chunk, part=(i, len(chunks))))
            except SpecExtractionError as exc:
                last_error = exc
        if not partials:
            raise SpecExtractionError(
                "All %d chunk extractions failed; last error: %s" % (len(chunks), last_error),
                last_error.raw_output if last_error else "",
            )
        if len(partials) == 1:
            return partials[0]

        # Pass 2: LLM merge, deterministic code merge as fallback.
        try:
            return self._merge_call(partials)
        except (SpecExtractionError, LLMError):
            return merge_partial_specs(partials)

    # ── LLM calls ──────────────────────────────────────────────────

    def _extract_call(
        self, task: TaskSpec, paper_text: str, part: Optional[Tuple[int, int]] = None
    ) -> SimulationSpec:
        part_note = ""
        if part is not None:
            part_note = (
                "NOTE: this is PART %d of %d of the paper text. Other parts are "
                "extracted separately and merged later, so information absent "
                "here may simply be in another part.\n\n" % part
            )
        raw = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Research task: {task.task_en}\n"
                        f"Expected domain: {task.domain}\n\n"
                        f"{part_note}Paper text:\n{paper_text}"
                    ),
                },
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return self._parse(raw)

    def _merge_call(self, partials: List[SimulationSpec]) -> SimulationSpec:
        listing = "\n\n".join(
            "--- partial spec %d of %d ---\n%s"
            % (i + 1, len(partials), p.model_dump_json(exclude={"work_id"}))
            for i, p in enumerate(partials)
        )
        raw = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": MERGE_PROMPT},
                {"role": "user", "content": listing},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return self._parse(raw)

    def _parse(self, raw: str) -> SimulationSpec:
        try:
            data = extract_json_object(raw)
        except ValueError as exc:
            raise SpecExtractionError(str(exc), raw) from exc

        # Normalize enums defensively; drop unknown keys via pydantic.
        if data.get("domain") not in ("semiconductor", "unknown"):
            data["domain"] = "unknown"
        if data.get("solver") not in ("devsim", "none"):
            data["solver"] = "none"
        data.pop("work_id", None)
        for key in (
            "governing_equations",
            "materials",
            "parameters",
            "operating_conditions",
            "numerical_settings",
            "target_results",
            "ambiguities",
        ):
            if data.get(key) is None:
                data[key] = []
        for ambiguity in data.get("ambiguities", []):
            if isinstance(ambiguity, dict) and ambiguity.get("candidates") is None:
                ambiguity["candidates"] = []
        for key in (
            "governing_equations",
            "materials",
            "operating_conditions",
            "numerical_settings",
        ):
            normalized = []
            for item in data.get(key, []):
                if isinstance(item, str):
                    normalized.append(item)
                elif isinstance(item, dict):
                    normalized.append("; ".join(
                        "%s: %s" % (k, v) for k, v in item.items() if v is not None
                    ))
                elif item is not None:
                    normalized.append(str(item))
            data[key] = normalized
        for target in data.get("target_results", []):
            if not isinstance(target, dict):
                continue
            for key in ("description", "location", "quantity", "value"):
                value = target.get(key)
                if isinstance(value, list):
                    target[key] = ", ".join(str(item) for item in value)
                elif isinstance(value, dict):
                    target[key] = "; ".join(
                        "%s: %s" % (k, v) for k, v in value.items() if v is not None
                    )
                elif value is not None and not isinstance(value, str):
                    target[key] = str(value)

        try:
            return SimulationSpec(**data)
        except Exception as exc:
            raise SpecExtractionError(
                f"Model JSON does not fit SimulationSpec: {exc}", raw
            ) from exc


# ── deterministic merge fallback ───────────────────────────────────


def _union(seqs: List[List[str]]) -> List[str]:
    seen, out = set(), []
    for seq in seqs:
        for item in seq:
            key = item.strip().lower()
            if key and key not in seen:
                seen.add(key)
                out.append(item)
    return out


def _longest(values: List[str]) -> str:
    values = [v for v in values if v]
    return max(values, key=len) if values else ""


def _param_rank(p: ParameterEntry) -> int:
    if p.value and p.source == "paper":
        return 3
    if p.value:
        return 2
    if p.source == "assumed":
        return 1
    return 0


def merge_partial_specs(parts: List[SimulationSpec]) -> SimulationSpec:
    """Merge chunk-level partial specs in code (LLM merge fallback).

    Dedup-union everywhere; for duplicate parameters the best-sourced entry
    wins (paper value > any value > assumed > missing), and ambiguities that
    another chunk resolved with a concrete paper value are dropped.
    """
    params = {}
    for part in parts:
        for entry in part.parameters:
            key = entry.name.strip().lower() or (entry.symbol or "").strip().lower()
            if not key:
                key = "unnamed-%d" % len(params)
            prev = params.get(key)
            if prev is None or _param_rank(entry) > _param_rank(prev):
                params[key] = entry
    resolved = {k for k, v in params.items() if v.value and v.source == "paper"}

    targets, seen_t = [], set()
    for part in parts:
        for t in part.target_results:
            key = (t.location.strip().lower(), re.sub(r"\W+", "", t.description.lower()))
            if key not in seen_t:
                seen_t.add(key)
                targets.append(t)

    ambiguities, seen_a = [], set()
    for part in parts:
        for a in part.ambiguities:
            key = a.field.strip().lower()
            if key in seen_a or key in resolved:
                continue
            seen_a.add(key)
            ambiguities.append(a)

    domains = [p.domain for p in parts if p.domain != "unknown"]
    solvers = [p.solver for p in parts if p.solver != "none"]
    return SimulationSpec(
        title=_longest([p.title for p in parts]),
        domain=Counter(domains).most_common(1)[0][0] if domains else "unknown",
        solver=Counter(solvers).most_common(1)[0][0] if solvers else "none",
        model_summary=_longest([p.model_summary for p in parts]),
        governing_equations=_union([p.governing_equations for p in parts]),
        geometry=_longest([p.geometry for p in parts]),
        materials=_union([p.materials for p in parts]),
        parameters=list(params.values()),
        operating_conditions=_union([p.operating_conditions for p in parts]),
        numerical_settings=_union([p.numerical_settings for p in parts]),
        target_results=targets,
        ambiguities=ambiguities,
    )
