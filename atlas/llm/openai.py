"""OpenAI LLM provider — uses the OpenAI Python SDK.

Requires: pip install openai>=1.0
"""

from __future__ import annotations

import os

from atlas.llm.provider import LLMProvider, LLMResponse

MODEL_TIERS: dict[str, str] = {
    "fast": os.environ.get("ATLAS_OPENAI_MODEL_FAST", "gpt-4o-mini"),
    "balanced": os.environ.get("ATLAS_OPENAI_MODEL_BALANCED", "gpt-4o-mini"),
    "powerful": os.environ.get("ATLAS_OPENAI_MODEL_POWERFUL", "gpt-4o"),
}


def model_for_preference(preference: str) -> str:
    """Resolve a contract model preference to a concrete OpenAI model ID."""
    return MODEL_TIERS.get(preference, MODEL_TIERS["balanced"])


class OpenAIProvider:
    """LLM provider backed by OpenAI's Chat Completions API.

    API key resolution order:
        1. Explicit `api_key` parameter
        2. ATLAS_OPENAI_API_KEY env var
        3. OPENAI_API_KEY env var (SDK default)
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        max_tokens: int = 1024,
    ) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ImportError(
                "openai package required: pip install 'atlas[openai]'"
            ) from e

        resolved_key = api_key or os.environ.get("ATLAS_OPENAI_API_KEY")
        self._client = AsyncOpenAI(api_key=resolved_key)
        self._model = model
        self._max_tokens = max_tokens

    async def complete(self, prompt: str) -> LLMResponse:
        """Send a prompt to OpenAI and return structured response."""
        response = await self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )

        choice = response.choices[0]
        text = choice.message.content or ""
        usage = response.usage

        return LLMResponse(
            text=text,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            model=self._model,
        )
