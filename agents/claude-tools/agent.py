"""Research agent using Claude with tool use — direct Anthropic SDK.

This agent bypasses the LLMAgent abstraction entirely to demonstrate
that AgentBase.execute() is the only real contract. Inside execute(),
it uses the Anthropic Messages API tool-use loop directly.
"""

from __future__ import annotations

import json
import math
from typing import Any

from atlas.runtime.base import AgentBase

_MAX_ITERATIONS = 5

# Built-in tools — no external dependencies needed
_TOOLS = [
    {
        "name": "calculate",
        "description": "Evaluate a mathematical expression. Supports basic arithmetic, powers, sqrt, etc.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression to evaluate (e.g. '42 * 17', 'sqrt(144)')",
                },
            },
            "required": ["expression"],
        },
    },
    {
        "name": "lookup",
        "description": "Look up a topic in the knowledge base. Returns a short factual summary.",
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Topic to look up",
                },
            },
            "required": ["topic"],
        },
    },
]

# Small built-in knowledge base for the lookup tool
_KNOWLEDGE: dict[str, str] = {
    "recursion": "Recursion is a method where the solution depends on solutions to smaller instances of the same problem. Found in mathematics, computer science, and nature (fractals, ferns, Romanesco broccoli).",
    "emergence": "Emergence is when complex systems exhibit properties that their individual parts do not have on their own. Examples: consciousness from neurons, flocking from simple bird rules, market behavior from individual trades.",
    "entropy": "Entropy is a measure of disorder or information content. In thermodynamics, it quantifies energy unavailable for work. In information theory (Shannon), it measures uncertainty or surprise in a message.",
    "fibonacci": "The Fibonacci sequence (0, 1, 1, 2, 3, 5, 8, 13, ...) appears throughout nature: sunflower spirals, pinecone scales, shell curves. Each number is the sum of the two preceding ones.",
    "atlas": "Atlas is a framework-agnostic agent runtime that composes agents from different SDKs via declarative contracts and automatic I/O mediation.",
}


def _execute_tool(name: str, input_data: dict[str, Any]) -> str:
    """Execute a built-in tool and return the result as a string."""
    if name == "calculate":
        expr = input_data["expression"]
        # Safe math evaluation — only allow math operations
        allowed = {
            "sqrt": math.sqrt,
            "abs": abs,
            "round": round,
            "pow": pow,
            "pi": math.pi,
            "e": math.e,
        }
        try:
            result = eval(expr, {"__builtins__": {}}, allowed)  # noqa: S307
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    if name == "lookup":
        topic = input_data["topic"].lower().strip()
        # Try exact match first, then substring match
        if topic in _KNOWLEDGE:
            return _KNOWLEDGE[topic]
        for key, value in _KNOWLEDGE.items():
            if topic in key or key in topic:
                return value
        return f"No information found for '{input_data['topic']}'. Available topics: {', '.join(_KNOWLEDGE.keys())}"

    return f"Unknown tool: {name}"


class ClaudeToolsAgent(AgentBase):
    """Research agent that uses Claude with tool calling.

    Demonstrates using the Anthropic Messages API tool-use loop directly
    inside an Atlas agent. The contract system (agent.yaml) handles
    I/O validation; internally this agent is free to use any SDK pattern.
    """

    def __init__(self, contract, context) -> None:
        super().__init__(contract, context)
        self._client = None
        self._model = "claude-haiku-4-5-20251001"

    async def on_startup(self) -> None:
        """Create the Anthropic client.

        If ``context.providers["anthropic_client"]`` is set, uses that
        directly (dependency injection for testing).
        """
        injected = self.context.providers.get("anthropic_client")
        if injected is not None:
            self._client = injected
            return

        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "anthropic package required: pip install 'atlas[llm]'"
            ) from e
        self._client = anthropic.AsyncAnthropic()

    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        question = input_data["question"]
        messages: list[dict] = [{"role": "user", "content": question}]
        tools_used: list[str] = []
        iterations = 0

        for _ in range(_MAX_ITERATIONS):
            iterations += 1
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=1024,
                tools=_TOOLS,
                messages=messages,
            )

            # Check if we got tool use blocks
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

            if not tool_use_blocks:
                # No tool calls — extract text answer
                text = "".join(b.text for b in response.content if b.type == "text")
                return {
                    "answer": text.strip(),
                    "tools_used": tools_used,
                    "steps": iterations,
                    "model": self._model,
                }

            # Process tool calls
            assistant_content = []
            for block in response.content:
                if block.type == "text":
                    assistant_content.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    assistant_content.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })

            messages.append({"role": "assistant", "content": assistant_content})

            # Execute tools and build results
            tool_results = []
            for block in tool_use_blocks:
                tools_used.append(block.name)
                result = _execute_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

            messages.append({"role": "user", "content": tool_results})

        # Hit max iterations — return what we have
        return {
            "answer": "Reached maximum reasoning steps without a final answer.",
            "tools_used": tools_used,
            "steps": iterations,
            "model": self._model,
        }
