"""Creative writing agent — powered by Claude via the Anthropic SDK."""

from __future__ import annotations

from typing import Any

from atlas.llm.provider import LLMResponse
from atlas.runtime.llm_agent import LLMAgent


class ClaudeWriterAgent(LLMAgent):
    """Writes content on a given topic using Claude."""

    def build_prompt(self, input_data: dict[str, Any]) -> str:
        topic = input_data["topic"]
        style = input_data.get("style", "concise")
        return (
            f"Write a short piece about the following topic.\n"
            f"Style: {style}\n"
            f"Topic: {topic}\n\n"
            f"Return ONLY the content, no preamble or labels."
        )

    def parse_response(
        self, response: LLMResponse, input_data: dict[str, Any]
    ) -> dict[str, Any]:
        content = response.text.strip()
        return {
            "content": content,
            "word_count": len(content.split()),
            "model": response.model,
        }
