"""Stage C assembly: acquire PDF → parse → extract SimulationSpec → persist.

Two entries:
- Mode A: the user supplied a local PDF (``pdf_path``).
- Mode B: the user picked a shortlist candidate (``candidate``) — its
  open-access PDF is downloaded (sha256-deduplicated in the DB).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

from korani.agents.spec_extractor import SpecExtractor
from korani.fulltext import (
    FulltextError,
    download_pdf,
    extract_pdf_text,
    sha256_of_file,
    trim_for_llm,
)
from korani.llm import LLMClient, OpenAICompatClient
from korani.models import PaperCandidate, SimulationSpec, TaskSpec
from korani.storage import Storage


class StageCError(RuntimeError):
    pass


def run_stage_c(
    task: TaskSpec,
    config: Dict,
    candidate: Optional[PaperCandidate] = None,
    pdf_path: Optional[str] = None,
    client: Optional[LLMClient] = None,
) -> Tuple[SimulationSpec, str, str]:
    """Return (spec, work_id, spec_file_path). ``client`` injectable for tests."""
    if pdf_path is None and candidate is None:
        raise StageCError("stage C needs either a picked candidate or a local PDF.")

    data_dir = config.get("data_dir", "data")
    verify_ssl = config["search"].get("verify_ssl", True)

    # ── Acquire the PDF ──
    if pdf_path is not None:
        if not Path(pdf_path).exists():
            raise StageCError(f"PDF not found: {pdf_path}")
        sha = sha256_of_file(pdf_path)
    else:
        if not candidate.pdf_url:
            raise StageCError(
                "No open-access PDF for this paper"
                + (f" (doi: {candidate.doi})" if candidate.doi else "")
                + ". Obtain it manually and rerun with --paper <file>."
            )
        pdf_path, sha = download_pdf(
            candidate.pdf_url, dest_dir=str(Path(data_dir) / "papers"),
            verify_ssl=verify_ssl,
        )

    # ── Parse + trim ──
    text = extract_pdf_text(pdf_path)
    ext_cfg = config.get("extraction", {})
    max_chars = ext_cfg.get("max_chars", 24000)
    chunk_overlap = ext_cfg.get("chunk_overlap", 2000)
    max_chunks = ext_cfg.get("max_chunks", 6)
    # Overall budget = per-call budget × chunk cap; the extractor chunks
    # anything longer than one call's worth (two-pass extraction).
    text = trim_for_llm(text, max_chars * max_chunks)

    # ── Extract the SimulationSpec (LLM, risk stage) ──
    if client is None:
        client = OpenAICompatClient(
            base_url=config["llm"]["base_url"],
            api_key=config["llm"].get("api_key", "not-needed"),
        )
    extractor = SpecExtractor(client=client, model=config["models"]["spec_extractor"])
    spec = extractor.extract(
        task, text, max_chars=max_chars, chunk_overlap=chunk_overlap, max_chunks=max_chunks
    )

    # ── Persist: work (metadata) + asset (sha-deduped) + spec ──
    with Storage(data_dir) as storage:
        if candidate is None:
            candidate = PaperCandidate(title=spec.title or Path(pdf_path).name, sources=["local"])
        work_id = storage.upsert_work(candidate)
        storage.add_asset(
            work_id, sha, str(pdf_path), format="pdf",
            size_bytes=Path(pdf_path).stat().st_size,
        )
        spec.work_id = work_id
        storage.save_spec(work_id, spec)

    specs_dir = Path(data_dir) / "specs"
    specs_dir.mkdir(parents=True, exist_ok=True)
    spec_file = specs_dir / f"{work_id}.json"
    spec_file.write_text(
        json.dumps(spec.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return spec, work_id, str(spec_file)
