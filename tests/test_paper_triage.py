"""Paper Triage tests — offline with stub LLM."""

import json

import pytest

from korani.agents.paper_triage import PaperTriage, TriageError
from korani.models import PaperCandidate, TaskSpec


class StubClient:
    def __init__(self, response):
        self.response = response
        self.last_messages = None

    def chat(self, model, messages, temperature=0.2, max_tokens=1024):
        self.last_messages = messages
        return self.response


SPEC = TaskSpec(mode="B", domain="battery", task_en="reproduce low-T degradation model")

CANDIDATES = [
    PaperCandidate(title="Paper one", doi="10.1/1"),
    PaperCandidate(title="Paper two", doi="10.1/2"),
    PaperCandidate(title="Paper three", doi="10.1/3"),
]


def triage_with(response):
    return PaperTriage(StubClient(response), model="stub")


def test_ranked_by_score_and_capped():
    response = json.dumps(
        [
            {"index": 1, "score": 4, "solver_fit": "pybamm", "rationale": "partial params"},
            {"index": 2, "score": 9, "solver_fit": "pybamm", "rationale": "full P2D params"},
            {"index": 3, "score": 6, "solver_fit": "none", "rationale": "experimental only"},
        ]
    )
    shortlist = triage_with(response).triage(SPEC, CANDIDATES, queries_used=["q"], shortlist_size=2)
    assert [e.candidate.title for e in shortlist.entries] == ["Paper two", "Paper three"]
    assert shortlist.entries[0].score == 9
    assert shortlist.queries_used == ["q"]


def test_hallucinated_duplicate_and_out_of_range_indices_ignored():
    response = json.dumps(
        [
            {"index": 99, "score": 10, "solver_fit": "pybamm"},   # out of range
            {"index": 1, "score": 5, "solver_fit": "pybamm"},
            {"index": 1, "score": 8, "solver_fit": "pybamm"},     # duplicate
            {"index": "x", "score": 8},                            # non-numeric
        ]
    )
    shortlist = triage_with(response).triage(SPEC, CANDIDATES, queries_used=[])
    assert len(shortlist.entries) == 1
    assert shortlist.entries[0].score == 5


def test_score_clamped_and_bad_solver_fit_normalized():
    response = json.dumps([{"index": 1, "score": 42, "solver_fit": "matlab"}])
    entry = triage_with(response).triage(SPEC, CANDIDATES, queries_used=[]).entries[0]
    assert entry.score == 10
    assert entry.solver_fit == "none"


def test_markdown_fenced_array_accepted():
    response = "```json\n" + json.dumps([{"index": 2, "score": 7, "solver_fit": "pybamm"}]) + "\n```"
    shortlist = triage_with(response).triage(SPEC, CANDIDATES, queries_used=[])
    assert shortlist.entries[0].candidate.title == "Paper two"


def test_empty_candidates_short_circuits_without_llm():
    class ExplodingClient:
        def chat(self, *a, **k):
            raise AssertionError("LLM must not be called for empty candidates")

    shortlist = PaperTriage(ExplodingClient(), model="stub").triage(SPEC, [], queries_used=[])
    assert shortlist.entries == []


def test_unusable_output_raises_with_raw():
    with pytest.raises(TriageError) as excinfo:
        triage_with("no array here").triage(SPEC, CANDIDATES, queries_used=[])
    assert excinfo.value.raw_output
