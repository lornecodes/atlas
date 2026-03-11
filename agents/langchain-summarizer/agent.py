"""Summarizer agent using LangChain LCEL (LangChain Expression Language).

Demonstrates a third framework pattern: instead of using the Anthropic or
OpenAI SDKs directly, this agent builds an LCEL chain internally
(prompt | llm | parser) while still conforming to the Atlas AgentBase contract.

The chain system has no idea LangChain is running inside — it just sees
{text: string} → {summary, key_points, model}.
"""

from __future__ import annotations

import json
from typing import Any

from atlas.runtime.base import AgentBase


class LangChainSummarizerAgent(AgentBase):
    """Summarizer that uses LangChain LCEL internally.

    On startup, builds an LCEL chain:
        ChatPromptTemplate | ChatModel | StrOutputParser

    The LLM backend is configurable via env vars:
        LANGCHAIN_SUMMARIZER_PROVIDER = "anthropic" (default) | "openai"
    """

    def __init__(self, contract: Any, context: Any) -> None:
        super().__init__(contract, context)
        self._chain = None
        self._model_name = ""

    async def on_startup(self) -> None:
        """Build the LCEL chain.

        If ``context.providers["langchain_chain"]`` is set, uses that
        directly (dependency injection). Expects a tuple of
        ``(chain, model_name)``.  Otherwise builds the real chain from
        LangChain imports.
        """
        injected = self.context.providers.get("langchain_chain")
        if injected is not None:
            self._chain, self._model_name = injected
            return

        import os

        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        provider = os.environ.get("LANGCHAIN_SUMMARIZER_PROVIDER", "anthropic")

        if provider == "openai":
            from langchain_openai import ChatOpenAI

            model_name = os.environ.get("LANGCHAIN_SUMMARIZER_MODEL", "gpt-4o-mini")
            llm = ChatOpenAI(model=model_name, temperature=0)
        else:
            from langchain_anthropic import ChatAnthropic

            model_name = os.environ.get(
                "LANGCHAIN_SUMMARIZER_MODEL", "claude-haiku-4-5-20251001"
            )
            llm = ChatAnthropic(model=model_name, temperature=0)

        self._model_name = model_name

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a precise summarizer. Extract key points from the input text.\n"
                    "Respond in this exact JSON format (no markdown, no code fences):\n"
                    '{{"summary": "one paragraph summary", "key_points": ["point 1", "point 2"]}}\n'
                    "Return at most {max_points} key points.",
                ),
                ("human", "{text}"),
            ]
        )

        self._chain = prompt | llm | StrOutputParser()

    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        if not self._chain:
            raise RuntimeError("LangChain agent not started — on_startup() not called")

        text = input_data["text"]
        max_points = input_data.get("max_points", 5)

        raw = await self._chain.ainvoke({"text": text, "max_points": max_points})
        return self._parse(raw.strip())

    def _parse(self, text: str) -> dict[str, Any]:
        """Parse LLM JSON output into the contract output schema."""
        try:
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            data = json.loads(text)
            return {
                "summary": str(data.get("summary", text)),
                "key_points": list(data.get("key_points", [])),
                "model": self._model_name,
            }
        except (json.JSONDecodeError, ValueError):
            return {
                "summary": text,
                "key_points": [],
                "model": self._model_name,
            }
