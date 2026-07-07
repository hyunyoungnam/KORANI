"""Defensive JSON extraction from LLM output.

Small open models sometimes wrap JSON in prose or markdown fences despite
instructions; every agent parses through here instead of trusting raw output.
Raises ``ValueError`` — agents map it to their own error types.
"""

from __future__ import annotations

import json
import re


def extract_json_object(text: str) -> dict:
    return _extract(text, "{", "}", dict)


def extract_json_array(text: str) -> list:
    return _extract(text, "[", "]", list)


def _extract(text: str, opener: str, closer: str, expected_type: type):
    fenced = re.search(
        r"```(?:json)?\s*(\%s.*?\%s)\s*```" % (opener, closer), text, re.DOTALL
    )
    if fenced:
        text = fenced.group(1)
    start = text.find(opener)
    end = text.rfind(closer)
    if start == -1 or end <= start:
        raise ValueError(f"No JSON {expected_type.__name__} found in model output.")
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model output is not valid JSON: {exc}") from exc
    if not isinstance(value, expected_type):
        raise ValueError(f"Expected a JSON {expected_type.__name__}.")
    return value
