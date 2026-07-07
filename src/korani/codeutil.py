"""Defensive Python-code extraction from LLM output (Engineer/Debugger).

Mirror of jsonutil: coding models sometimes wrap code in prose or answer
with several fences despite instructions. Raises ``ValueError`` — agents map
it to their own error types.
"""

from __future__ import annotations

import re

_FENCE = re.compile(r"```(?:python|py)?[ \t]*\r?\n(.*?)```", re.DOTALL)


def extract_python_code(text: str) -> str:
    """Return the largest fenced code block, or the whole text if it already
    looks like bare Python."""
    fences = _FENCE.findall(text)
    if fences:
        return max(fences, key=len).strip() + "\n"
    if re.search(r"^\s*(import|from)\s+\w+", text, re.MULTILINE):
        return text.strip() + "\n"
    raise ValueError("No Python code found in model output.")
