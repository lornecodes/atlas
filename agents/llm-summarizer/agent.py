"""LLM-backed summarizer agent."""

from __future__ import annotations

from typing import Any

from atlas.llm.provider import LLMResponse
from atlas.runtime.llm_agent import LLMAgent


class LLMSummarizerAgent(LLMAgent):
    """Summarizes text using an LLM provider."""

    def build_prompt(self, input_data: dict[str, Any]) -> str:
        text = input_data["text"]
        max_sentences = input_data.get("max_sentences", 3)
        return (
            f"Summarize the following text in at most {max_sentences} sentences. "
            f"Return ONLY the summary, no preamble.\n\n{text}"
        )

    def parse_response(
        self, response: LLMResponse, input_data: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "summary": response.text.strip(),
            "model": response.model,
        }
