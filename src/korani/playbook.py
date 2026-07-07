"""Failure playbook — automatically harvested Debugger knowledge.

CLAUDE.md settled decision: no curated knowledge base; instead, when the
Debugger turns a failing run into a working one, the fix is distilled (one
small LLM call) into a markdown entry in the AgenticSciML KB format
(Problem / Issue / Fix / Critical parameters), keyed by solver + exception
type. Future Debugger calls on the same error class retrieve the entry as a
hint. Strictly best-effort: playbook failures never sink a run.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from korani.llm import LLMClient

DISTILL_PROMPT = """\
You are harvesting a debugging knowledge-base entry for KORANI. A generated \
simulation script failed with the error below and was then fixed. Write a \
SHORT markdown entry (max 15 lines) that will help debug the SAME ERROR \
CLASS in FUTURE papers — generalize, do not repeat this paper's specifics.

Use exactly these sections:
## Problem
## Issue
## Fix
## Critical parameters
"""

_EXC_RE = re.compile(r"^([A-Za-z_][\w.]*(?:Error|Exception))\b")


def error_signature(stderr_tail: str) -> str:
    """Exception class of the last raised error, e.g. 'ValueError'."""
    signature = "unknown"
    for line in stderr_tail.splitlines():
        match = _EXC_RE.match(line.strip())
        if match:
            signature = match.group(1).rsplit(".", 1)[-1]
    return signature


def _entry_path(data_dir: str, solver: str, signature: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", "%s_%s" % (solver, signature))
    return Path(data_dir) / "playbook" / ("%s.md" % safe)


def lookup(data_dir: str, solver: str, signature: str) -> Optional[str]:
    path = _entry_path(data_dir, solver, signature)
    try:
        return path.read_text(encoding="utf-8") if path.exists() else None
    except OSError:
        return None


def record(
    client: LLMClient,
    model: str,
    data_dir: str,
    solver: str,
    signature: str,
    error_tail: str,
    fixed_code: str,
) -> None:
    """Distill a resolved failure into a playbook entry (best-effort)."""
    try:
        entry = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": DISTILL_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Solver: %s\nError class: %s\n\nError output:\n%s\n\n"
                        "Fixed script (beginning):\n%s"
                        % (solver, signature, error_tail[-2000:], fixed_code[:3000])
                    ),
                },
            ],
            temperature=0.1,
            max_tokens=512,
        )
        path = _entry_path(data_dir, solver, signature)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(entry.strip() + "\n", encoding="utf-8")
    except Exception:
        pass  # the playbook must never sink the run that fed it
