"""Debugger agent — stage E.

Fixes solver errors in the Engineer's script within bounded retries (the
retry loop and the solver budget live in engineer_debugger.py, not here).
May receive a failure-playbook entry harvested from earlier runs of the
same error class.
"""

from __future__ import annotations

from typing import Optional

from korani.codeutil import extract_python_code
from korani.llm import LLMClient

SYSTEM_PROMPT = """\
You are the Debugger agent of KORANI. A generated simulation script failed; \
fix it.

RULES:
1. Return the FULL corrected script in ONE ```python code fence — never a \
snippet or a diff.
2. Make the smallest change that plausibly resolves the reported error; do \
not restructure parts that work.
3. NEVER change a physical parameter value taken from the paper just to \
make an error go away. If you believe a parameter value itself is the \
problem, keep it, add a `# SUSPECT PARAMETER:` comment explaining why, and \
fix what you can.
4. Keep the output contract intact: the script must still write \
results.json with the same keys (values computed from the simulation, \
never hardcoded) and the same curve CSV files.
"""


class DebuggerError(RuntimeError):
    def __init__(self, message: str, raw_output: str = ""):
        super().__init__(message)
        self.raw_output = raw_output


class Debugger:
    def __init__(self, client: LLMClient, model: str, temperature: float = 0.2, max_tokens: int = 6144):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def debug(
        self, code: str, error_tail: str, solver: str, hint: Optional[str] = None
    ) -> str:
        hint_block = ""
        if hint:
            hint_block = (
                "\n\nPlaybook entry for this error class (from an earlier "
                "resolved failure — may or may not apply):\n%s" % hint
            )
        raw = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Solver: %s\n\nError output:\n%s%s\n\nCurrent script:\n"
                        "```python\n%s\n```" % (solver, error_tail, hint_block, code)
                    ),
                },
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        try:
            return extract_python_code(raw)
        except ValueError as exc:
            raise DebuggerError(str(exc), raw) from exc
