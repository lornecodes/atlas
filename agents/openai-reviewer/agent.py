"""Content reviewer agent — powered by OpenAI GPT."""

from __future__ import annotations

import json
from typing import Any

from atlas.llm.provider import LLMResponse
from atlas.runtime.llm_agent import LLMAgent


class OpenAIReviewerAgent(LLMAgent):
    """Reviews content using OpenAI GPT.

    Demonstrates using a non-Anthropic LLM provider within the Atlas
    contract system. Overrides _create_provider() to use OpenAIProvider.
    """

    def _create_provider(self):
        from atlas.llm.openai import OpenAIProvider, model_for_preference

        preference = self.contract.model.preference
        return OpenAIProvider(model=model_for_preference(preference))

    def build_prompt(self, input_data: dict[str, Any]) -> str:
        content = input_data["content"]
        criteria = input_data.get("criteria", "clarity and accuracy")
        return (
            f"Review the following content based on these criteria: {criteria}\n\n"
            f"Content:\n{content}\n\n"
            f"Respond in this exact JSON format (no markdown, no code fences):\n"
            f'{{"review": "your review text", "rating": 7, "suggestions": ["suggestion 1", "suggestion 2"]}}\n'
            f"Rating must be an integer from 1 to 10."
        )

    def parse_response(
        self, response: LLMResponse, input_data: dict[str, Any]
    ) -> dict[str, Any]:
        text = response.text.strip()
        # Try to parse structured JSON response
        try:
            # Strip markdown fences if present
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            data = json.loads(text)
            return {
                "review": str(data.get("review", text)),
                "rating": int(data.get("rating", 5)),
                "suggestions": list(data.get("suggestions", [])),
                "model": response.model,
            }
        except (json.JSONDecodeError, ValueError, KeyError):
            # Fallback: use raw text as review
            return {
                "review": text,
                "rating": 5,
                "suggestions": [],
                "model": response.model,
            }
