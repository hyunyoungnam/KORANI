"""SearchCoordinator dedup/merge/pre-rank tests — offline with fake providers."""

from korani.models import PaperCandidate
from korani.search import SearchCoordinator, prerank_score


class FakeProvider:
    def __init__(self, name, papers):
        self.name = name
        self.papers = papers

    def search(self, query, limit=10):
        return [p.model_copy(deep=True) for p in self.papers]


class BrokenProvider:
    name = "broken"

    def search(self, query, limit=10):
        raise RuntimeError("provider down")


def paper(**kwargs):
    defaults = dict(title="T", sources=["x"])
    defaults.update(kwargs)
    return PaperCandidate(**defaults)


def test_dedup_by_doi_merges_sources_and_fields():
    a = paper(title="Same paper", doi="10.1/a", sources=["openalex"], citation_count=10)
    b = paper(
        title="Same Paper (S2 casing)",
        doi="10.1/a",
        sources=["semanticscholar"],
        abstract="abs",
        pdf_url="http://pdf",
        citation_count=12,
    )
    coordinator = SearchCoordinator(
        providers=[FakeProvider("p1", [a]), FakeProvider("p2", [b])]
    )
    results = coordinator.search(["q"])
    assert len(results) == 1
    merged = results[0]
    assert merged.sources == ["openalex", "semanticscholar"]
    assert merged.abstract == "abs"           # filled from richer record
    assert merged.pdf_url == "http://pdf"
    assert merged.citation_count == 12        # max wins


def test_dedup_by_normalized_title_when_no_doi():
    a = paper(title="Drift-Diffusion Simulation of GaN HEMTs!")
    b = paper(title="drift diffusion simulation of gan hemts")
    coordinator = SearchCoordinator(providers=[FakeProvider("p", [a, b])])
    assert len(coordinator.search(["q"])) == 1


def test_max_candidates_cap_and_ranking():
    papers = [
        paper(title=f"p{i}", doi=f"10.1/{i}", citation_count=i * 100)
        for i in range(1, 6)
    ]
    coordinator = SearchCoordinator(
        providers=[FakeProvider("p", papers)], max_candidates=3
    )
    results = coordinator.search(["q"])
    assert len(results) == 3
    assert results[0].title == "p5"  # most cited first


def test_broken_provider_does_not_sink_search():
    ok = FakeProvider("ok", [paper(title="survivor", doi="10.1/s")])
    coordinator = SearchCoordinator(providers=[BrokenProvider(), ok])
    results = coordinator.search(["q"])
    assert [p.title for p in results] == ["survivor"]


def test_prerank_prefers_pdf_and_abstract():
    bare = paper(title="bare")
    rich = paper(title="rich", pdf_url="http://pdf", abstract="a")
    assert prerank_score(rich) > prerank_score(bare)
