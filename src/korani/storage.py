"""SQLite persistence — KoCoScientist schema pattern, scaled to KORANI.

Tables:
- ``works``            canonical paper metadata, upserted by DOI
- ``fulltext_assets``  downloaded PDFs, deduplicated by sha256
- ``simulation_specs`` extracted SimulationSpec JSON per work

A ``simulation_runs`` table joins later with stage F (see CLAUDE.md
Persistence). Plain sqlite3 + WAL — the CLI is synchronous and single-user;
async SQLAlchemy would be dead weight here.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Optional

from korani.models import PaperCandidate, SimulationSpec

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
