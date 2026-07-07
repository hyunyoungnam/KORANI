"""Storage roundtrip tests — real SQLite in tmp_path."""

from korani.models import PaperCandidate, ParameterEntry, SimulationSpec
from korani.storage import Storage


def candidate(**kwargs):
    defaults = dict(title="A paper", doi="10.1/x", sources=["openalex"])
    defaults.update(kwargs)
    return PaperCandidate(**defaults)


def test_upsert_work_dedups_by_doi(tmp_path):
    with Storage(str(tmp_path)) as storage:
        first = storage.upsert_work(candidate())
        second = storage.upsert_work(candidate(title="Same paper, other title"))
        assert first == second
        assert storage.get_work(first)["title"] == "A paper"


def test_works_without_doi_are_distinct(tmp_path):
    with Storage(str(tmp_path)) as storage:
        a = storage.upsert_work(candidate(doi=None, title="local upload 1"))
        b = storage.upsert_work(candidate(doi=None, title="local upload 2"))
        assert a != b


def test_asset_dedup_by_sha256(tmp_path):
    with Storage(str(tmp_path)) as storage:
        work_id = storage.upsert_work(candidate())
        first = storage.add_asset(work_id, "abc123", "/papers/x.pdf")
        second = storage.add_asset(work_id, "abc123", "/papers/duplicate.pdf")
        assert first == second
        assert storage.get_asset_by_sha256("abc123")["path"] == "/papers/x.pdf"


def test_spec_roundtrip_latest_wins(tmp_path):
    with Storage(str(tmp_path)) as storage:
        work_id = storage.upsert_work(candidate())
        v1 = SimulationSpec(title="T", solver="pybamm", model_summary="v1")
        v2 = SimulationSpec(
            title="T", solver="pybamm", model_summary="v2",
            parameters=[ParameterEntry(name="D_s", value="1e-14", units="m2/s")],
        )
        storage.save_spec(work_id, v1)
        storage.save_spec(work_id, v2)
        loaded = storage.get_latest_spec(work_id)
        assert loaded.model_summary == "v2"
        assert loaded.parameters[0].name == "D_s"


def test_get_latest_spec_none_when_absent(tmp_path):
    with Storage(str(tmp_path)) as storage:
        work_id = storage.upsert_work(candidate())
        assert storage.get_latest_spec(work_id) is None
