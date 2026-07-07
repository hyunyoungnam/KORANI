"""Provider parsing tests — offline via httpx.MockTransport."""

import httpx

from korani.providers.openalex import OpenAlexProvider, normalize_doi, reconstruct_abstract
from korani.providers.semanticscholar import SemanticScholarProvider

OPENALEX_RESPONSE = {
    "results": [
        {
            "id": "https://openalex.org/W123",
            "display_name": "Low-temperature degradation modeling of Li-ion cells",
            "publication_year": 2022,
            "authorships": [
                {"author": {"display_name": "Kim, H."}},
                {"author": {"display_name": "Lee, J."}},
            ],
            "abstract_inverted_index": {"Low": [0], "temperature": [1], "aging": [2]},
            "doi": "https://doi.org/10.1000/XYZ123",
            "cited_by_count": 87,
            "primary_location": {"landing_page_url": "https://journal.example/paper"},
            "open_access": {"oa_url": "https://oa.example/paper.pdf"},
            "best_oa_location": {"pdf_url": "https://best.example/paper.pdf"},
        },
        {"display_name": None},  # dropped: no title
    ]
}

S2_RESPONSE = {
    "data": [
        {
            "title": "P2D model validation at subzero temperatures",
            "abstract": "We validate a P2D model...",
            "year": 2023,
            "authors": [{"name": "Park, S."}],
            "externalIds": {"DOI": "10.1000/xyz123"},
            "citationCount": 12,
            "openAccessPdf": {"url": "https://s2.example/p.pdf"},
            "url": "https://semanticscholar.org/paper/abc",
        }
    ]
}


def _client(payload):
    return httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    )


def test_openalex_parsing():
    provider = OpenAlexProvider(client=_client(OPENALEX_RESPONSE))
    results = provider.search("low temperature battery degradation")
    assert len(results) == 1  # titleless entry dropped
    paper = results[0]
    assert paper.abstract == "Low temperature aging"  # inverted index rebuilt
    assert paper.doi == "10.1000/xyz123"  # prefix stripped + lowercased
    assert paper.pdf_url == "https://best.example/paper.pdf"  # best_oa preferred
    assert paper.citation_count == 87
    assert paper.sources == ["openalex"]


def test_semanticscholar_parsing():
    provider = SemanticScholarProvider(client=_client(S2_RESPONSE))
    results = provider.search("P2D subzero")
    paper = results[0]
    assert paper.doi == "10.1000/xyz123"
    assert paper.pdf_url == "https://s2.example/p.pdf"
    assert paper.sources == ["semanticscholar"]


def test_reconstruct_abstract_orders_by_position():
    assert reconstruct_abstract({"world": [1], "hello": [0]}) == "hello world"
    assert reconstruct_abstract(None) is None
    assert reconstruct_abstract({}) is None


def test_normalize_doi():
    assert normalize_doi("https://doi.org/10.1/A.B") == "10.1/a.b"
    assert normalize_doi(None) is None
    assert normalize_doi("") is None
