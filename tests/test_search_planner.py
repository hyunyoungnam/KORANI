"""Search Planner tests — offline with stub LLM."""

import json

from korani.agents.search_planner import SearchPlanner
from korani.models import TaskSpec


class StubClient:
    def __init__(self, response):
        self.response = response

    def chat(self, model, messages, temperature=0.2, max_tokens=1024):
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def spec(queries):
    return TaskSpec(mode="B", domain="battery", task_en="task", search_queries_en=queries)


def test_merges_interpreter_and_llm_queries_dedup():
    response = json.dumps({"queries": ["low temperature aging model", "P2D model cold climate"]})
    planner = SearchPlanner(StubClient(response), model="stub")
    queries = planner.plan(spec(["low temperature aging model", "battery degradation"]))
    # interpreter queries first, LLM extras appended, case-insensitive dedup
    assert queries == [
        "low temperature aging model",
        "battery degradation",
        "P2D model cold climate",
    ]


def test_cap_at_max_queries():
    response = json.dumps({"queries": [f"q{i}" for i in range(10)]})
    planner = SearchPlanner(StubClient(response), model="stub")
    assert len(planner.plan(spec(["a", "b"]), max_queries=4)) == 4


def test_llm_failure_falls_back_to_interpreter_queries():
    planner = SearchPlanner(StubClient(RuntimeError("down")), model="stub")
    assert planner.plan(spec(["only query"])) == ["only query"]


def test_garbage_output_falls_back():
    planner = SearchPlanner(StubClient("not json at all"), model="stub")
    assert planner.plan(spec(["q1"])) == ["q1"]


def test_no_queries_anywhere_falls_back_to_task():
    planner = SearchPlanner(StubClient("{}"), model="stub")
    assert planner.plan(spec([])) == ["task"]
