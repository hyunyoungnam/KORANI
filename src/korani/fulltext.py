"""PDF acquisition and text extraction for stage C."""

from __future__ import annotations

import hashlib
import re
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

import httpx


class FulltextError(RuntimeError):
    pass


def sha256_of_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download_pdf(
    url: str,
    dest_dir: str,
    verify_ssl: bool = True,
    timeout: float = 60.0,
    client: Optional[httpx.Client] = None,
) -> Tuple[str, str]:
    """Download a PDF and return (path, sha256).

    Content is verified to actually be a PDF (%PDF magic) — OA links often
    serve HTML landing pages instead of the file.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    http = client or httpx.Client(
        timeout=timeout, verify=verify_ssl, follow_redirects=True,
        headers={"User-Agent": "KORANI/0.1 (research; non-commercial)"},
    )
    try:
        response = http.get(url)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise FulltextError(f"PDF download failed: {exc}") from exc

    content = response.content
    if not content.startswith(b"%PDF"):
        raise FulltextError(
            f"URL did not return a PDF (got {response.headers.get('content-type')!r}); "
            "download it manually and rerun with --paper <file>."
        )

    sha = hashlib.sha256(content).hexdigest()
    path = dest / f"{sha[:16]}.pdf"
    if not path.exists():
        tmp = dest / f".{uuid.uuid4().hex}.part"
        tmp.write_bytes(content)
        tmp.replace(path)
    return str(path), sha


def extract_pdf_text(path: str) -> str:
    """Extract plain text from a PDF using PyMuPDF."""
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise FulltextError(
            "PyMuPDF is not installed — `pip install pymupdf`."
        ) from exc
    try:
        pages = []
        with fitz.open(path) as doc:
            for page in doc:
                pages.append(page.get_text())
    except Exception as exc:
        raise FulltextError(f"PDF parsing failed for {path}: {exc}") from exc
    text = "\n".join(pages).strip()
    if not text:
        raise FulltextError(
            f"No extractable text in {path} (scanned/image-only PDF?)."
        )
    return text


def trim_for_llm(text: str, max_chars: int) -> str:
    """Cut the references section, then cap length for the LLM context."""
    match = re.search(r"\n\s*(References|REFERENCES|Bibliography)\s*\n", text)
    if match and match.start() > len(text) * 0.4:
        text = text[: match.start()]
    return text[:max_chars]


def chunk_text(text: str, max_chars: int, overlap: int = 2000) -> List[str]:
    """Split text into <=max_chars pieces with overlap, for chunked extraction.

    Prefers to cut at a newline in the last fifth of the window so parameter
    tables and sentences are less likely to be split mid-line; the overlap
    lets whichever chunk holds the full line recover what the cut mangled.
    """
    if len(text) <= max_chars:
        return [text]
    overlap = min(overlap, max_chars // 2)
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            cut = text.rfind("\n", start + int(max_chars * 0.8), end)
            if cut > start:
                end = cut
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks
