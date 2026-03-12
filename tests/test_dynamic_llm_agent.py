"""Tests for DynamicLLMAgent — YAML-only LLM agent with tool-use loop."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from atlas.contract.types import AgentContract, ModelSpec, ProviderSpec, RequiresSpec
from atlas.runtime.context import AgentContext
from atlas.runtime.dynamic_llm_agent import DynamicLLMAgent
from atlas.skills.types import SkillSpec


# --- Mock helpers ---

@dataclass
class MockTextBlock:
    type: str = "text"
    text: str = ""


@dataclass
class MockToolUseBlock:
    type: str = "tool_use"
    id: str = "tool-1"
    name: str = ""
    input: dict = field(default_factory=dict)


@dataclass
class MockUsage:
    input_tokens: int = 100
    output_tokens: int = 50


@dataclass
class MockResponse:
    content: list = field(default_factory=list)
    usage: MockUsage = field(default_factory=MockUsage)


def _make_contract(
    system_prompt: str = "You are helpful.",
    focus: str = "",
    output_format: str = "json",
    max_iterations: int = 5,
    skills: list[str] | None = None,
    memory: bool = False,
) -> AgentContract:
    return AgentContract(
        name="test-llm",
        version="1.0.0",
        provider=ProviderSpec(
            type="llm",
            system_prompt=system_prompt,
            focus=focus,
            output_format=output_format,
            max_iterations=max_iterations,
        ),
        model=ModelSpec(preference="fast"),
        requires=RequiresSpec(skills=skills or [], memory=memory),
    )


def _make_context(
    skills: dict[str, Any] | None = None,
    skill_specs: dict[str, Any] | None = None,
    memory_provider: Any = None,
) -> AgentContext:
    ctx = AgentContext(
        job_id="job-test",
        providers={},
    )
    ctx._skills = skills or {}
    ctx._skill_specs = skill_specs or {}
    ctx._memory_provider = memory_provider
    return ctx


def _mock_client_simple_response(text: str) -> AsyncMock:
    """Create a mock client that returns a simple text response."""
    client = AsyncMock()
    response = MockResponse(content=[MockTextBlock(text=text)])
    client.messages.create = AsyncMock(return_value=response)
    return client


class TestDynamicLLMAgentBasic:
    """Basic LLM agent functionality."""

    @pytest.mark.asyncio
    async def test_simple_json_response(self):
        client = _mock_client_simple_response('{"summary": "Brief text"}')
        contract = _make_contract()
        ctx = _make_context()
        ctx.providers["anthropic_client"] = client

        agent = DynamicLLMAgent(contract, ctx)
        await agent.on_startup()
        result = await agent.execute({"text": "Long text here"})

        assert result == {"summary": "Brief text"}
        client.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_text_output_format(self):
        client = _mock_client_simple_response("Hello world")
        contract = _make_contract(output_format="text")
        ctx = _make_context()
        ctx.providers["anthropic_client"] = client

        agent = DynamicLLMAgent(contract, ctx)
        await agent.on_startup()
        result = await agent.execute({"input": "test"})

        assert result == {"result": "Hello world"}

    @pytest.mark.asyncio
    async def test_system_prompt_passed(self):
        client = _mock_client_simple_response('{"out": 1}')
        contract = _make_contract(system_prompt="Be concise.")
        ctx = _make_context()
        ctx.providers["anthropic_client"] = client

        agent = DynamicLLMAgent(contract, ctx)
        await agent.on_startup()
        await agent.execute({"x": 1})

        call_kwargs = client.messages.create.call_args[1]
        assert call_kwargs["system"] == "Be concise."

    @pytest.mark.asyncio
    async def test_focus_fallback(self):
        client = _mock_client_simple_response('{"out": 1}')
        contract = _make_contract(system_prompt="", focus="Summarize text.")
        ctx = _make_context()
        ctx.providers["anthropic_client"] = client

        agent = DynamicLLMAgent(contract, ctx)
        await agent.on_startup()
        await agent.execute({"x": 1})

        call_kwargs = client.messages.create.call_args[1]
        assert call_kwargs["system"] == "Summarize text."

    @pytest.mark.asyncio
    async def test_token_metadata_captured(self):
        client = _mock_client_simple_response('{"out": 1}')
        contract = _make_contract()
        ctx = _make_context()
        ctx.providers["anthropic_client"] = client

        agent = DynamicLLMAgent(contract, ctx)
        await agent.on_startup()
        await agent.execute({"x": 1})

        assert ctx.execution_metadata["input_tokens"] == 100
        assert ctx.execution_metadata["output_tokens"] == 50

    @pytest.mark.asyncio
    async def test_invalid_json_returns_result_wrapper(self):
        client = _mock_client_simple_response("not valid json")
        contract = _make_contract(output_format="json")
        ctx = _make_context()
        ctx.providers["anthropic_client"] = client

        agent = DynamicLLMAgent(contract, ctx)
        await agent.on_startup()
        result = await agent.execute({"x": 1})

        assert result == {"result": "not valid json"}

    @pytest.mark.asyncio
    async def test_not_started_raises(self):
        contract = _make_contract()
        ctx = _make_context()
        agent = DynamicLLMAgent(contract, ctx)
        with pytest.raises(RuntimeError, match="not started"):
            await agent.execute({"x": 1})


class TestDynamicLLMAgentToolUse:
    """Tool-use loop functionality."""

    @pytest.mark.asyncio
    async def test_tool_call_dispatched_to_skill(self):
        """LLM calls a tool, skill is invoked, then LLM returns final text."""
        # First call returns tool_use, second call returns text
        tool_response = MockResponse(content=[
            MockToolUseBlock(id="t1", name="search", input={"query": "atlas"}),
        ])
        final_response = MockResponse(content=[
            MockTextBlock(text='{"answer": "Atlas is a runtime"}'),
        ])
        client = AsyncMock()
        client.messages.create = AsyncMock(side_effect=[tool_response, final_response])

        search_skill = AsyncMock(return_value={"results": ["Atlas docs"]})
        search_spec = SkillSpec(name="search", version="1.0.0", description="Search")

        contract = _make_contract(skills=["search"])
        ctx = _make_context(
            skills={"search": search_skill},
            skill_specs={"search": search_spec},
        )
        ctx.providers["anthropic_client"] = client

        agent = DynamicLLMAgent(contract, ctx)
        await agent.on_startup()
        result = await agent.execute({"question": "What is Atlas?"})

        assert result == {"answer": "Atlas is a runtime"}
        search_skill.assert_called_once_with({"query": "atlas"})
        assert client.messages.create.call_count == 2

    @pytest.mark.asyncio
    async def test_tool_definitions_from_skill_specs(self):
        client = _mock_client_simple_response('{"out": 1}')
        spec = SkillSpec(
            name="calculator",
            version="1.0.0",
            description="Do math",
        )

        contract = _make_contract()
        ctx = _make_context(
            skills={"calculator": AsyncMock()},
            skill_specs={"calculator": spec},
        )
        ctx.providers["anthropic_client"] = client

        agent = DynamicLLMAgent(contract, ctx)
        await agent.on_startup()
        await agent.execute({"x": 1})

        call_kwargs = client.messages.create.call_args[1]
        tools = call_kwargs.get("tools", [])
        assert len(tools) == 1
        assert tools[0]["name"] == "calculator"
        assert tools[0]["description"] == "Do math"

    @pytest.mark.asyncio
    async def test_max_iterations_reached(self):
        """If LLM keeps calling tools past max_iterations, return error."""
        tool_response = MockResponse(content=[
            MockToolUseBlock(id="t1", name="search", input={"q": "x"}),
        ])
        client = AsyncMock()
        client.messages.create = AsyncMock(return_value=tool_response)

        search_skill = AsyncMock(return_value={"results": []})

        contract = _make_contract(max_iterations=2, skills=["search"])
        ctx = _make_context(
            skills={"search": search_skill},
            skill_specs={"search": SkillSpec(name="search", version="1.0.0")},
        )
        ctx.providers["anthropic_client"] = client

        agent = DynamicLLMAgent(contract, ctx)
        await agent.on_startup()
        result = await agent.execute({"q": "test"})

        assert "error" in result
        assert "max iterations" in result["error"]
        assert client.messages.create.call_count == 2

    @pytest.mark.asyncio
    async def test_no_tools_when_no_skills(self):
        """Without skills, no tools are passed to the LLM."""
        client = _mock_client_simple_response('{"out": 1}')
        contract = _make_contract()
        ctx = _make_context()
        ctx.providers["anthropic_client"] = client

        agent = DynamicLLMAgent(contract, ctx)
        await agent.on_startup()
        await agent.execute({"x": 1})

        call_kwargs = client.messages.create.call_args[1]
        # No tools key or empty tools
        assert not call_kwargs.get("tools")


class TestDynamicLLMAgentMemory:
    """Memory integration for dynamic LLM agents."""

    @pytest.mark.asyncio
    async def test_memory_injected_into_system_prompt(self):
        client = _mock_client_simple_response('{"out": 1}')
        mock_mem = AsyncMock()
        mock_mem.read = AsyncMock(return_value="Previous learning: rate limit is 100/min")

        contract = _make_contract(
            system_prompt="You are helpful.",
            memory=True,
        )
        ctx = _make_context(memory_provider=mock_mem)
        ctx.providers["anthropic_client"] = client

        agent = DynamicLLMAgent(contract, ctx)
        await agent.on_startup()
        await agent.execute({"x": 1})

        call_kwargs = client.messages.create.call_args[1]
        system = call_kwargs["system"]
        assert "## Shared Memory" in system
        assert "rate limit is 100/min" in system

    @pytest.mark.asyncio
    async def test_memory_append_tool_available(self):
        client = _mock_client_simple_response('{"out": 1}')
        mock_mem = AsyncMock()
        mock_mem.read = AsyncMock(return_value="")

        contract = _make_contract(memory=True)
        ctx = _make_context(memory_provider=mock_mem)
        ctx.providers["anthropic_client"] = client

        agent = DynamicLLMAgent(contract, ctx)
        await agent.on_startup()
        await agent.execute({"x": 1})

        call_kwargs = client.messages.create.call_args[1]
        tools = call_kwargs.get("tools", [])
        tool_names = [t["name"] for t in tools]
        assert "memory_append" in tool_names

    @pytest.mark.asyncio
    async def test_memory_append_tool_writes(self):
        """When LLM calls memory_append, it writes to memory provider."""
        tool_response = MockResponse(content=[
            MockToolUseBlock(
                id="t1", name="memory_append",
                input={"entry": "Learned: use batch requests"},
            ),
        ])
        final_response = MockResponse(content=[
            MockTextBlock(text='{"done": true}'),
        ])
        client = AsyncMock()
        client.messages.create = AsyncMock(side_effect=[tool_response, final_response])

        mock_mem = AsyncMock()
        mock_mem.read = AsyncMock(return_value="")

        contract = _make_contract(memory=True)
        ctx = _make_context(memory_provider=mock_mem)
        ctx.providers["anthropic_client"] = client

        agent = DynamicLLMAgent(contract, ctx)
        await agent.on_startup()
        result = await agent.execute({"x": 1})

        mock_mem.append.assert_called_once_with("Learned: use batch requests")
        assert result == {"done": True}

    @pytest.mark.asyncio
    async def test_no_memory_no_tool(self):
        """Without memory provider, no memory_append tool is added."""
        client = _mock_client_simple_response('{"out": 1}')
        contract = _make_contract()
        ctx = _make_context()  # No memory provider
        ctx.providers["anthropic_client"] = client

        agent = DynamicLLMAgent(contract, ctx)
        await agent.on_startup()
        await agent.execute({"x": 1})

        call_kwargs = client.messages.create.call_args[1]
        tools = call_kwargs.get("tools", [])
        assert not any(t["name"] == "memory_append" for t in tools)


class TestDynamicLLMAgentStartup:
    """Agent startup and provider injection."""

    @pytest.mark.asyncio
    async def test_injected_client_used(self):
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(
            return_value=MockResponse(content=[MockTextBlock(text='{"x": 1}')])
        )

        contract = _make_contract()
        ctx = _make_context()
        ctx.providers["anthropic_client"] = mock_client

        agent = DynamicLLMAgent(contract, ctx)
        await agent.on_startup()
        assert agent._client is mock_client

    @pytest.mark.asyncio
    async def test_missing_anthropic_raises(self, monkeypatch):
        """If anthropic not installed and no client injected, raise ImportError."""
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "anthropic":
                raise ImportError("No module named 'anthropic'")
            return original_import(name, *args, **kwargs)

        contract = _make_contract()
        ctx = _make_context()
        agent = DynamicLLMAgent(contract, ctx)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="anthropic"):
            await agent.on_startup()
