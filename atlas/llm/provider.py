"""LLM provider protocol — the abstraction layer for LLM calls.

Any LLM backend (Anthropic, OpenAI, local, mock) implements this protocol.
The mediation engine depends only on this interface, never on a concrete SDK.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class LLMResponse:
    """Structured response from an LLM call."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers.

    Implementations must be callable with a string prompt and return an LLMResponse.
    """

    async def complete(self, prompt: str) -> LLMResponse:
        """Send a prompt and get a structured response."""
        ...
