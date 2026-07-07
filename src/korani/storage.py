"""SQLite persistence — KoCoScientist schema pattern, scaled to KORANI.

Tables:
- ``works``            canonical paper metadata, upserted by DOI
- ``fulltext_assets``  downloaded PDFs, deduplicated by sha256
- ``simulation_specs`` extracted SimulationSpec JSON per work
- ``evaluation_contracts`` stage D contracts (draft/approved history per work)
- ``simulation_runs`` stage E variant outcomes: spec hash, status, scores,
  artifact paths — powers "search past experiments" and the future
  optimization layer's memory (CLAUDE.md Persistence).
- ``analysis_reports`` stage F Result Analyst verdicts; the full report is a
  file referenced by path (Analysis Base pattern).

Plain sqlite3 + WAL — the CLI is synchronous and single-user;
async SQLAlchemy would be dead weight here.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Optional

from korani.models import (
    EvaluationContract,
    PaperCandidate,
    SimulationSpec,
    VariantOutcome,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS works (
    id             TEXT PRIMARY KEY,
    title          TEXT NOT NULL,
    year           INTEGER,
    doi            TEXT UNIQUE,
    authors_json   TEXT,
    url            TEXT,
    pdf_url        TEXT,
    citation_count INTEGER,
    sources_json   TEXT,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS fulltext_assets (
    id         TEXT PRIMARY KEY,
    work_id    TEXT NOT NULL REFERENCES works(id),
    sha256     TEXT UNIQUE NOT NULL,
    path       TEXT NOT NULL,
    format     TEXT NOT NULL,
    size_bytes INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS simulation_specs (
    id         TEXT PRIMARY KEY,
    work_id    TEXT NOT NULL REFERENCES works(id),
    spec_json  TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS evaluation_contracts (
    id            TEXT PRIMARY KEY,
    work_id       TEXT NOT NULL REFERENCES works(id),
    contract_json TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'draft',
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS simulation_runs (
    id            TEXT PRIMARY KEY,
    work_id       TEXT NOT NULL REFERENCES works(id),
    variant       TEXT,
    solver        TEXT,
    spec_sha256   TEXT,
    status        TEXT NOT NULL,
    attempts      INTEGER,
    eval_exit     INTEGER,
    eval_passed   INTEGER,
    eval_failed   INTEGER,
    eval_deferred INTEGER,
    code_path     TEXT,
    results_path  TEXT,
    error_tail    TEXT,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS analysis_reports (
    id          TEXT PRIMARY KEY,
    work_id     TEXT NOT NULL REFERENCES works(id),
    variant     TEXT,
    verdict     TEXT,
    report_path TEXT,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


class Storage:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "korani.db"
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Storage":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ── works ──────────────────────────────────────────────────────

    def upsert_work(self, candidate: PaperCandidate) -> str:
        """Insert the paper or return the existing work id (DOI match)."""
        if candidate.doi:
            row = self._conn.execute(
                "SELECT id FROM works WHERE doi = ?", (candidate.doi,)
            ).fetchone()
            if row:
                return row[0]
        work_id = str(uuid.uuid4())
        self._conn.execute(
            "INSERT INTO works (id, title, year, doi, authors_json, url, pdf_url,"
            " citation_count, sources_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                work_id,
                candidate.title,
                candidate.year,
                candidate.doi,
                json.dumps(candidate.authors, ensure_ascii=False),
                candidate.url,
                candidate.pdf_url,
                candidate.citation_count,
                json.dumps(candidate.sources),
            ),
        )
        self._conn.commit()
        return work_id

    def get_work(self, work_id: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT id, title, year, doi, authors_json, url, pdf_url,"
            " citation_count, sources_json FROM works WHERE id = ?",
            (work_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "title": row[1],
            "year": row[2],
            "doi": row[3],
            "authors": json.loads(row[4] or "[]"),
            "url": row[5],
            "pdf_url": row[6],
            "citation_count": row[7],
            "sources": json.loads(row[8] or "[]"),
        }

    # ── fulltext assets ────────────────────────────────────────────

    def add_asset(
        self, work_id: str, sha256: str, path: str, format: str = "pdf",
        size_bytes: Optional[int] = None,
    ) -> str:
        """Register a downloaded file; same content (sha256) is never duplicated."""
        row = self._conn.execute(
            "SELECT id FROM fulltext_assets WHERE sha256 = ?", (sha256,)
        ).fetchone()
        if row:
            return row[0]
        asset_id = str(uuid.uuid4())
        self._conn.execute(
            "INSERT INTO fulltext_assets (id, work_id, sha256, path, format,"
            " size_bytes) VALUES (?, ?, ?, ?, ?, ?)",
            (asset_id, work_id, sha256, path, format, size_bytes),
        )
        self._conn.commit()
        return asset_id

    def get_asset_by_sha256(self, sha256: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT id, work_id, path, format FROM fulltext_assets WHERE sha256 = ?",
            (sha256,),
        ).fetchone()
        if row is None:
            return None
        return {"id": row[0], "work_id": row[1], "path": row[2], "format": row[3]}

    def get_assets_for_work(self, work_id: str) -> list:
        rows = self._conn.execute(
            "SELECT id, path, format FROM fulltext_assets WHERE work_id = ?"
            " ORDER BY created_at DESC, rowid DESC",
            (work_id,),
        ).fetchall()
        return [{"id": r[0], "path": r[1], "format": r[2]} for r in rows]

    # ── simulation specs ───────────────────────────────────────────

    def save_spec(self, work_id: str, spec: SimulationSpec) -> str:
        spec_id = str(uuid.uuid4())
        self._conn.execute(
            "INSERT INTO simulation_specs (id, work_id, spec_json) VALUES (?, ?, ?)",
            (spec_id, work_id, spec.model_dump_json()),
        )
        self._conn.commit()
        return spec_id

    def get_latest_spec(self, work_id: str) -> Optional[SimulationSpec]:
        row = self._conn.execute(
            "SELECT spec_json FROM simulation_specs WHERE work_id = ?"
            " ORDER BY created_at DESC, rowid DESC LIMIT 1",
            (work_id,),
        ).fetchone()
        if row is None:
            return None
        return SimulationSpec.model_validate_json(row[0])

    # ── evaluation contracts (stage D) ─────────────────────────────

    def save_contract(self, work_id: str, contract: EvaluationContract) -> str:
        """Insert a contract snapshot; approval inserts a new approved row,
        so the draft→approved history is preserved (latest row wins)."""
        contract_id = str(uuid.uuid4())
        self._conn.execute(
            "INSERT INTO evaluation_contracts (id, work_id, contract_json, status)"
            " VALUES (?, ?, ?, ?)",
            (contract_id, work_id, contract.model_dump_json(), contract.status),
        )
        self._conn.commit()
        return contract_id

    def get_latest_contract(self, work_id: str) -> Optional[EvaluationContract]:
        row = self._conn.execute(
            "SELECT contract_json FROM evaluation_contracts WHERE work_id = ?"
            " ORDER BY created_at DESC, rowid DESC LIMIT 1",
            (work_id,),
        ).fetchone()
        if row is None:
            return None
        return EvaluationContract.model_validate_json(row[0])

    # ── simulation runs (stage E) ──────────────────────────────────

    def save_run(
        self, work_id: str, spec_sha256: str, solver: str, outcome: VariantOutcome
    ) -> str:
        run_id = str(uuid.uuid4())
        self._conn.execute(
            "INSERT INTO simulation_runs (id, work_id, variant, solver,"
            " spec_sha256, status, attempts, eval_exit, eval_passed,"
            " eval_failed, eval_deferred, code_path, results_path, error_tail)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                work_id,
                outcome.name,
                solver,
                spec_sha256,
                outcome.status,
                outcome.attempts,
                outcome.eval_exit,
                outcome.eval_passed,
                outcome.eval_failed,
                outcome.eval_deferred,
                outcome.code_path,
                outcome.results_path,
                outcome.error_tail[-2000:],
            ),
        )
        self._conn.commit()
        return run_id

    def list_runs(self, work_id: str) -> list:
        rows = self._conn.execute(
            "SELECT variant, solver, status, attempts, eval_passed, eval_failed,"
            " eval_deferred, code_path, results_path, created_at"
            " FROM simulation_runs WHERE work_id = ? ORDER BY created_at",
            (work_id,),
        ).fetchall()
        keys = (
            "variant", "solver", "status", "attempts", "eval_passed",
            "eval_failed", "eval_deferred", "code_path", "results_path",
            "created_at",
        )
        return [dict(zip(keys, row)) for row in rows]

    # ── analysis reports (stage F) ─────────────────────────────────

    def save_analysis(
        self, work_id: str, variant: str, verdict: str, report_path: str
    ) -> str:
        analysis_id = str(uuid.uuid4())
        self._conn.execute(
            "INSERT INTO analysis_reports (id, work_id, variant, verdict,"
            " report_path) VALUES (?, ?, ?, ?, ?)",
            (analysis_id, work_id, variant, verdict, report_path),
        )
        self._conn.commit()
        return analysis_id
