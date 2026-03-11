"""LLM-backed text classifier agent."""

from __future__ import annotations

import json
from typing import Any

from atlas.llm.provider import LLMResponse
from atlas.runtime.llm_agent import LLMAgent


class LLMClassifierAgent(LLMAgent):
    """Classifies text into one of the provided categories."""

    def build_prompt(self, input_data: dict[str, Any]) -> str:
        text = input_data["text"]
        categories = input_data["categories"]
        cats = ", ".join(f'"{c}"' for c in categories)
        return (
            f"Classify the following text into exactly one of these categories: {cats}\n\n"
            f"Text: {text}\n\n"
            f'Respond with ONLY a JSON object: {{"category": "<chosen>", "confidence": "high|medium|low"}}'
        )

    def parse_response(
        self, response: LLMResponse, input_data: dict[str, Any]
    ) -> dict[str, Any]:
        text = response.text.strip()
        # Try to parse JSON from the response
        try:
            data = json.loads(text)
            category = data.get("category", "")
            confidence = data.get("confidence", "medium")
        except (json.JSONDecodeError, KeyError):
            # Fallback: use the raw text as category
            category = text.split('"')[1] if '"' in text else text
            confidence = "low"

        # Validate category is in the allowed list
        categories = input_data["categories"]
        if category not in categories:
            # Find closest match
            lower_cats = {c.lower(): c for c in categories}
            category = lower_cats.get(category.lower(), categories[0])

        return {
            "category": category,
            "confidence": confidence,
            "model": response.model,
        }
