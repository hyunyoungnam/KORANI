"""Interpreter agent — stage A (Understand).

The only agent that touches Korean user-facing text. Powered by KONI
(KISTI-AI, Korean science/tech-tuned) via the model registry in config.yaml.

Responsibilities:
- Parse the user's Korean request into a ``TaskSpec`` for downstream
  English-reasoning agents.
- Mode routing is NOT delegated to the LLM: mode "A" iff a paper is attached.
- Flag Korean technical terms it cannot confidently translate (non-negotiable:
  never silently guess a translation).
"""

from __future__ import annotations

from typing import Optional

from korani.jsonutil import extract_json_object
from korani.llm import LLMClient, LLMError
from korani.models import TaskSpec

SYSTEM_PROMPT = """\
You are the Interpreter agent of KORANI, a co-scientist system that helps \
Korean semiconductor researchers reproduce the Python code behind TCAD \
research papers using DEVSIM.

You receive a research request written in Korean. Convert it into a task \
specification for downstream English-speaking agents.

Respond with ONE JSON object and NOTHING else. Fields:
- "domain": "semiconductor" | "unknown"
- "task_en": faithful English description of what the user wants
- "search_queries_en": if no paper is attached, 3-5 English literature-search \
queries that would find a reproducible paper for this request; if a paper is \
attached, []
- "ambiguous_terms": Korean technical terms you cannot confidently translate. \
NEVER silently guess a translation - list each as \
{"term_ko": ..., "candidates_en": [...], "reason": ...}. Use [] if none.
- "clarifying_questions_ko": up to 3 questions in Korean, ONLY if the answer \
would materially change the task; otherwise []
- "reply_ko": 1-2 sentences in Korean confirming what you understood

Example output:
{"domain": "semiconductor",
 "task_en": "Reproduce the transfer characteristics of a silicon MOSFET",
 "search_queries_en": ["silicon MOSFET DEVSIM transfer characteristics",
                       "open access TCAD MOSFET drift diffusion simulation"],
 "ambiguous_terms": [],
 "clarifying_questions_ko": ["대상 소자 구조가 정해져 있나요?"],
 "reply_ko": "실리콘 MOSFET의 전달 특성을 DEVSIM으로 재현하는 과제로 이해했습니다."}
"""


class InterpreterError(RuntimeError):
    """Raised when the model output cannot be parsed into a TaskSpec."""

    def __init__(self, message: str, raw_output: str = ""):
        super().__init__(message)
        self.raw_output = raw_output


def _extract_json(text: str) -> dict:
    """Parse the model output's JSON object (shared logic in korani.jsonutil)."""
    try:
        return extract_json_object(text)
    except ValueError as exc:
        raise InterpreterError(str(exc), text) from exc


class Interpreter:
    def __init__(
        self,
        client: LLMClient,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def interpret(
        self, question_ko: str, paper_path: Optional[str] = None
    ) -> TaskSpec:
        """Turn a Korean request into a TaskSpec.

        ``paper_path`` present → Mode A (skip search); absent → Mode B.
        """
        mode = "A" if paper_path else "B"
        attachment_note = (
            f"A paper PDF is attached ({paper_path}). Mode A: no search needed."
            if paper_path
            else "No paper is attached. Mode B: literature search will be needed."
        )
        raw = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"{attachment_note}\n\nUser request (Korean):\n{question_ko}",
                },
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        data = _extract_json(raw)

        # The LLM never decides the mode; enforce it here.
        data["mode"] = mode
        data["paper_path"] = paper_path
        if mode == "A":
            data["search_queries_en"] = []

        try:
            return TaskSpec(**data)
        except Exception as exc:
            raise InterpreterError(f"Model JSON does not fit TaskSpec: {exc}", raw) from exc


def build_interpreter(config: dict) -> Interpreter:
    """Wire an Interpreter from a loaded config dict (see korani.config)."""
    from korani.llm import client_for_role

    client, model = client_for_role(config, "interpreter")
    return Interpreter(
        client=client,
        model=model,
        temperature=config["generation"]["temperature"],
        max_tokens=config["generation"]["max_tokens"],
    )
