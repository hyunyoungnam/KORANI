"""Interpreter unit tests — run offline with a stub LLM client."""

import json

import pytest

from korani.agents.interpreter import Interpreter, InterpreterError, _extract_json
from korani.models import TaskSpec

CANNED = {
    "domain": "battery",
    "task_en": "Design and simulate strategies to reduce battery degradation at low temperature",
    "search_queries_en": [
        "low temperature lithium-ion battery degradation model",
        "battery aging simulation cold climate PyBaMM",
    ],
    "ambiguous_terms": [
        {"term_ko": "열화", "candidates_en": ["degradation", "aging"], "reason": "context-dependent"}
    ],
    "clarifying_questions_ko": ["대상 셀 화학조성이 정해져 있나요?"],
    "reply_ko": "저온 환경에서의 배터리 열화 저감 설계 시뮬레이션 과제로 이해했습니다.",
}


class StubClient:
    """LLMClient stand-in returning a fixed response."""

    def __init__(self, response: str):
        self.response = response
        self.last_messages = None

    def chat(self, model, messages, temperature=0.2, max_tokens=1024):
        self.last_messages = messages
        return self.response


def make_interpreter(response: str) -> Interpreter:
    return Interpreter(client=StubClient(response), model="stub")


def test_mode_b_without_paper():
    spec = make_interpreter(json.dumps(CANNED)).interpret("저온에서 배터리 열화를 줄이는 설계를 시뮬레이션하고 싶은데...")
    assert spec.mode == "B"
    assert spec.domain == "battery"
    assert spec.search_queries_en  # Mode B keeps queries
    assert spec.paper_path is None
    assert spec.ambiguous_terms[0].term_ko == "열화"


def test_mode_a_with_paper_clears_queries():
    spec = make_interpreter(json.dumps(CANNED)).interpret(
        "이 논문을 재현해줘", paper_path="paper.pdf"
    )
    assert spec.mode == "A"
    assert spec.paper_path == "paper.pdf"
    assert spec.search_queries_en == []  # Mode A: no search


def test_mode_is_never_taken_from_llm():
    payload = dict(CANNED)
    payload["mode"] = "A"  # LLM tries to claim Mode A
    spec = make_interpreter(json.dumps(payload)).interpret("질문")
    assert spec.mode == "B"  # attachment presence wins


def test_json_wrapped_in_markdown_fence():
    fenced = "Here you go:\n```json\n" + json.dumps(CANNED) + "\n```"
    spec = make_interpreter(fenced).interpret("질문")
    assert isinstance(spec, TaskSpec)
    assert spec.domain == "battery"


def test_json_with_prose_preamble():
    noisy = "Sure! The result is: " + json.dumps(CANNED) + " Hope this helps."
    spec = make_interpreter(noisy).interpret("질문")
    assert spec.reply_ko.startswith("저온")


def test_garbage_output_raises_with_raw():
    with pytest.raises(InterpreterError) as excinfo:
        make_interpreter("죄송합니다, JSON을 만들 수 없습니다.").interpret("질문")
    assert excinfo.value.raw_output


def test_extract_json_direct():
    assert _extract_json('{"a": 1}') == {"a": 1}
