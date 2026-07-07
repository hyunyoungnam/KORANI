"""Fulltext download/parse tests — download mocked; parse uses a real tiny PDF."""

import httpx
import pytest

from korani.fulltext import FulltextError, download_pdf, trim_for_llm


def _client(content, content_type="application/pdf"):
    return httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, content=content, headers={"content-type": content_type}
            )
        )
    )


def test_download_pdf_saves_and_hashes(tmp_path):
    content = b"%PDF-1.4 fake pdf body"
    path, sha = download_pdf("https://x/p.pdf", str(tmp_path), client=_client(content))
    assert open(path, "rb").read() == content
    assert len(sha) == 64
    # same content again → same file, no duplicate
    path2, sha2 = download_pdf("https://y/other.pdf", str(tmp_path), client=_client(content))
    assert (path, sha) == (path2, sha2)


def test_download_rejects_html_landing_page(tmp_path):
    with pytest.raises(FulltextError, match="did not return a PDF"):
        download_pdf(
            "https://x/landing", str(tmp_path),
            client=_client(b"<html>paywall</html>", "text/html"),
        )


def test_trim_cuts_references_and_caps():
    body = "intro " * 2000
    text = body + "\nReferences\n[1] Someone et al."
    trimmed = trim_for_llm(text, max_chars=50000)
    assert "[1] Someone" not in trimmed
    assert len(trim_for_llm(text, max_chars=100)) == 100


def test_trim_ignores_early_references_mention():
    # 'References' in the first 40% is not the bibliography heading
    text = "\nReferences\n are discussed later. " + "body " * 1000
    assert "body" in trim_for_llm(text, max_chars=50000)


def test_extract_pdf_text_roundtrip(tmp_path):
    fitz = pytest.importorskip("fitz")
    from korani.fulltext import extract_pdf_text

    pdf_path = str(tmp_path / "tiny.pdf")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "DFN model with SEI growth, D_s = 1e-14 m2/s")
    doc.save(pdf_path)
    doc.close()

    text = extract_pdf_text(pdf_path)
    assert "SEI growth" in text
