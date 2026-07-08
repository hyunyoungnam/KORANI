"""LLM client seam.

Agents depend on the ``LLMClient`` protocol only — never on a concrete
provider. Two implementations exist:

- ``OpenAICompatClient`` — any OpenAI-compatible endpoint (Ollama locally,
  or OpenAI's API on the API profile).
- ``AnthropicClient`` — the Anthropic API via the official ``anthropic``
  SDK (their recommended integration; no OpenAI-compat shim).

``client_for_role`` picks the implementation from the config's per-role
binding (see ``korani.config.resolve_role_llm``), so swapping a role's
model or provider stays a config change, not a code change.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

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


class AnthropicClient:
    """Client for the Anthropic API via the official ``anthropic`` SDK.

    Protocol notes:
    - Anthropic takes the system prompt as a separate ``system`` parameter,
      so system-role messages are split out of the chat list here.
    - Current Claude models (Sonnet 5 / Opus 4.8) reject sampling
      parameters — ``temperature`` is accepted for LLMClient protocol
      compatibility but never sent.
    - Image content blocks use Anthropic's format, not OpenAI's ``image_url``
      — keep vision roles (result_analyst) on an OpenAI-compatible binding
      unless that conversion is added.
    """

    def __init__(self, api_key: str = ""):
        # Lazy import: the anthropic package is only needed on the API profile.
        from anthropic import Anthropic

        self._client = Anthropic(api_key=api_key) if api_key else Anthropic()

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        system = "\n\n".join(
            m["content"] for m in messages if m.get("role") == "system"
        )
        chat_messages = [m for m in messages if m.get("role") != "system"]
        kwargs: Dict = {"model": model, "max_tokens": max_tokens, "messages": chat_messages}
        if system:
            kwargs["system"] = system
        try:
            response = self._client.messages.create(**kwargs)
        except Exception as exc:
            raise LLMError(
                f"LLM call failed (model={model}, endpoint=anthropic): {exc}\n"
                "Is ANTHROPIC_API_KEY set (in .env or the environment)?"
            ) from exc

        if response.stop_reason == "refusal":
            raise LLMError(f"Anthropic model {model} declined the request (refusal).")
        text = "".join(b.text for b in response.content if b.type == "text")
        if not text:
            raise LLMError(f"LLM returned an empty response (model={model}).")
        return text


def client_for_role(
    config: Dict, role: str, client: Optional[LLMClient] = None
) -> Tuple[LLMClient, str]:
    """Return (client, model_name) for a pipeline role.

    ``client`` (test injection) short-circuits construction but still
    resolves the model name, so stub-based tests keep working unchanged.
    """
    from korani.config import resolve_role_llm

    binding = resolve_role_llm(config, role)
    if client is not None:
        return client, binding["model"]
    if binding["provider"] == "anthropic":
        return AnthropicClient(api_key=binding["api_key"]), binding["model"]
    return (
        OpenAICompatClient(base_url=binding["base_url"], api_key=binding["api_key"]),
        binding["model"],
    )
