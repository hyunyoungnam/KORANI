"""LLM client seam.

Agents depend on the ``LLMClient`` protocol only — never on a concrete
provider. ``OpenAICompatClient`` talks to any OpenAI-compatible endpoint
(vLLM, Ollama, or a commercial API later), so swapping models/providers is a
config change, not a code change.
"""

from __future__ import annotations

from typing import Dict, List

try:
    from typing import Protocol
except ImportError:  # pragma: no cover
    from typing_extensions import Protocol


class LLMError(RuntimeError):
    """Raised when the LLM endpoint is unreachable or returns garbage."""


class LLMClient(Protocol):
    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        """Return the assistant message text for a chat completion."""
        ...


class OpenAICompatClient:
    """Client for any OpenAI-compatible endpoint (vLLM / Ollama / etc.)."""

    def __init__(self, base_url: str, api_key: str = "not-needed"):
        # Lazy import so unit tests with stub clients don't need `openai`.
        from openai import OpenAI

        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self.base_url = base_url

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            raise LLMError(
                f"LLM call failed (model={model}, endpoint={self.base_url}): {exc}\n"
                "Is your local model server running? "
                "Start Ollama (`ollama serve`) or vLLM, and check config.yaml."
            ) from exc

        content = response.choices[0].message.content
        if not content:
            raise LLMError(f"LLM returned an empty response (model={model}).")
        return content
