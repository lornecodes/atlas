"""Anthropic LLM provider — uses the Anthropic Python SDK.

Requires: pip install anthropic>=0.40
"""

from __future__ import annotations

import os

from atlas.llm.provider import LLMProvider, LLMResponse

# Map contract model.preference to concrete model IDs.
# Users can override via ATLAS_MODEL_<TIER> env vars.
MODEL_TIERS: dict[str, str] = {
    "fast": os.environ.get("ATLAS_MODEL_FAST", "claude-haiku-4-5-20251001"),
    "balanced": os.environ.get("ATLAS_MODEL_BALANCED", "claude-sonnet-4-20250514"),
    "powerful": os.environ.get("ATLAS_MODEL_POWERFUL", "claude-sonnet-4-20250514"),
}


def model_for_preference(preference: str) -> str:
    """Resolve a contract model preference to a concrete model ID."""
    return MODEL_TIERS.get(preference, MODEL_TIERS["balanced"])


class AnthropicProvider:
    """LLM provider backed by Anthropic's Claude API.

    API key resolution order:
        1. Explicit `api_key` parameter
        2. ATLAS_API_KEY env var
        3. ANTHROPIC_API_KEY env var (SDK default)

    Usage:
        provider = AnthropicProvider(model="claude-haiku-4-5-20251001")
        response = await provider.complete("Transform this data...")
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1024,
    ) -> None:
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "anthropic package required: pip install 'atlas[llm]'"
            ) from e

        resolved_key = api_key or os.environ.get("ATLAS_API_KEY")
        self._client = anthropic.AsyncAnthropic(api_key=resolved_key)
        self._model = model
        self._max_tokens = max_tokens

    async def complete(self, prompt: str) -> LLMResponse:
        """Send a prompt to Claude and return structured response."""
        message = await self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )

        text = ""
        for block in message.content:
            if block.type == "text":
                text += block.text

        return LLMResponse(
            text=text,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            model=self._model,
        )
