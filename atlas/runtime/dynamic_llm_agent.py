"""DynamicLLMAgent — LLM agent defined entirely in YAML, no Python needed.

Skills declared in ``requires.skills`` are exposed as tools to the LLM.
The tool-use loop dispatches calls via ``context.skill()``.
This is the GRIM pattern: one LLM, different configs.
"""

from __future__ import annotations

import json
from typing import Any

from atlas.runtime.base import AgentBase


class DynamicLLMAgent(AgentBase):
    """LLM agent defined entirely in YAML — no Python code needed.

    The contract declares ``provider: {type: llm, system_prompt: "...", ...}``.
    At runtime, skills from ``requires.skills`` become LLM tools.
    The LLM can call tools iteratively until it produces a final answer.

    If ``requires.memory: true``, shared memory is prepended to the system
    prompt and a ``memory_append`` tool is exposed for the LLM to write back.
    """

    def __init__(self, contract: Any, context: Any) -> None:
        super().__init__(contract, context)
        self._client: Any = None

    async def on_startup(self) -> None:
        """Create the Anthropic client (or use injected mock)."""
        injected = self.context.providers.get("anthropic_client")
        if injected is not None:
            self._client = injected
            return

        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "anthropic package required for llm provider: pip install 'atlas[llm]'"
            ) from e

        import os

        kwargs: dict[str, Any] = {}
        base_url = os.environ.get("ANTHROPIC_BASE_URL")
        if base_url:
            kwargs["base_url"] = base_url
        api_key = os.environ.get("ATLAS_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            kwargs["api_key"] = api_key
        self._client = anthropic.AsyncAnthropic(**kwargs)

    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        if not self._client:
            raise RuntimeError("DynamicLLMAgent not started — on_startup() not called")

        system = self.contract.provider.system_prompt or self.contract.provider.focus or ""
        tools = self._build_tool_definitions()
        max_iter = self.contract.provider.max_iterations
        model = _model_for_preference(self.contract.model.preference)

        # Inject memory into system prompt
        if self.context._memory_provider:
            memory_content = await self.context.memory_read()
            if memory_content:
                system = f"{system}\n\n## Shared Memory\n{memory_content}"
            # Add memory_append tool
            tools.append({
                "name": "memory_append",
                "description": "Append a learning or note to shared memory for future agents to read.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "entry": {
                            "type": "string",
                            "description": "The text to append to shared memory.",
                        },
                    },
                    "required": ["entry"],
                },
            })

        # Format input as user message
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": json.dumps(input_data)},
        ]
        tools_used: list[str] = []

        for _ in range(max_iter):
            kwargs: dict[str, Any] = {
                "model": model,
                "max_tokens": 4096,
                "messages": messages,
            }
            if system:
                kwargs["system"] = system
            if tools:
                kwargs["tools"] = tools

            response = await self._client.messages.create(**kwargs)

            # Track token usage
            if hasattr(response, "usage"):
                self.context.execution_metadata.update({
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "model": model,
                })

            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

            if not tool_use_blocks:
                # No tool calls — extract text answer
                text = "".join(
                    b.text for b in response.content if b.type == "text"
                )
                if self.contract.provider.output_format == "json":
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError:
                        return {"result": text.strip()}
                return {"result": text.strip()}

            # Build assistant message with tool_use blocks
            assistant_content: list[dict[str, Any]] = []
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

            # Execute tool calls
            tool_results: list[dict[str, Any]] = []
            for block in tool_use_blocks:
                tools_used.append(block.name)

                if block.name == "memory_append" and self.context._memory_provider:
                    # Handle memory_append directly
                    entry = block.input.get("entry", "")
                    await self.context.memory_append(entry)
                    result_text = json.dumps({"status": "appended"})
                else:
                    # Dispatch to skill
                    result = await self.context.skill(block.name, block.input)
                    result_text = json.dumps(result)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                })
            messages.append({"role": "user", "content": tool_results})

        # Hit max iterations
        return {"error": "max iterations reached", "tools_used": tools_used}

    def _build_tool_definitions(self) -> list[dict[str, Any]]:
        """Convert injected skill specs to Anthropic tool format."""
        tools: list[dict[str, Any]] = []
        for name, spec in self.context._skill_specs.items():
            tools.append({
                "name": name,
                "description": spec.description or f"Tool: {name}",
                "input_schema": spec.input_schema.to_json_schema(),
            })
        return tools


def _model_for_preference(preference: str) -> str:
    """Resolve contract model preference to a concrete model ID."""
    from atlas.llm.anthropic import model_for_preference

    return model_for_preference(preference)
