"""Stage C assembly test — real PDF + real SQLite, stub LLM."""

import json

import pytest

from korani.models import PaperCandidate, TaskSpec
from korani.stage_c import StageCError, run_stage_c
from korani.storage import Storage

TASK = TaskSpec(mode="A", domain="battery", task_en="reproduce it")

SPEC_JSON = json.dumps(
    {
        "title": "Test paper",
        "domain": "battery",
        "solver": "pybamm",
        "model_summary": "SPM with thermal submodel",
        "parameters": [],
        "target_results": [],
        "ambiguities": [],
    }
)


class StubClient:
    def chat(self, model, messages, temperature=0.2, max_tokens=1024):
        return SPEC_JSON


def _config(tmp_path):
    return {
        "llm": {"base_url": "http://unused", "api_key": "x"},
        "models": {"spec_extractor": "stub"},
        "search": {"verify_ssl": True},
        "extraction": {"max_chars": 24000},
        "data_dir": str(tmp_path / "data"),
    }


def _tiny_pdf(tmp_path):
    fitz = pytest.importorskip("fitz")
    path = str(tmp_path / "paper.pdf")
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "SPM with thermal submodel. C-rate 1C.")
    doc.save(path)
    doc.close()
    return path


def test_mode_a_local_pdf_end_to_end(tmp_path):
    pdf = _tiny_pdf(tmp_path)
    config = _config(tmp_path)
    spec, work_id, spec_file = run_stage_c(
        TASK, config, pdf_path=pdf, client=StubClient()
    )
    assert spec.solver == "pybamm"
    assert spec.work_id == work_id

    # persisted: work + sha-deduped asset + latest spec + json file
    with Storage(config["data_dir"]) as storage:
        assert storage.get_work(work_id)["title"] == "Test paper"
        assert storage.get_latest_spec(work_id).model_summary == "SPM with thermal submodel"
    assert json.loads(open(spec_file, encoding="utf-8").read())["title"] == "Test paper"

    # re-running the same PDF must not duplicate the asset row
    run_stage_c(TASK, config, pdf_path=pdf, client=StubClient())
    with Storage(config["data_dir"]) as storage:
        n = storage._conn.execute("SELECT COUNT(*) FROM fulltext_assets").fetchone()[0]
        assert n == 1


def test_mode_b_candidate_without_pdf_url_fails_clearly(tmp_path):
    candidate = PaperCandidate(title="paywalled", doi="10.1/p")
    with pytest.raises(StageCError, match="--paper"):
        run_stage_c(TASK, _config(tmp_path), candidate=candidate, client=StubClient())


def test_missing_local_pdf_fails_clearly(tmp_path):
    with pytest.raises(StageCError, match="not found"):
        run_stage_c(TASK, _config(tmp_path), pdf_path=str(tmp_path / "nope.pdf"), client=StubClient())
