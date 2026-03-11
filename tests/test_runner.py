"""Tests for the agent runner — E2E execution."""

from __future__ import annotations

import asyncio

import pytest
import yaml

from atlas.contract.registry import AgentRegistry
from atlas.runtime.runner import run_agent, RunError
from atlas.runtime.context import AgentContext


class TestRunAgent:
    async def test_run_echo(self, registry: AgentRegistry):
        result = await run_agent(registry, "echo", {"message": "hello world"})
        assert result.success
        assert result.data == {"message": "hello world"}
        assert result.agent_name == "echo"

    async def test_run_summarizer(self, registry: AgentRegistry):
        long_text = "This is a test sentence. " * 20
        result = await run_agent(registry, "summarizer", {
            "text": long_text,
            "max_length": 50,
        })
        assert result.success
        assert len(result.data["summary"]) <= 50
        assert result.data["token_count"] > 0

    async def test_run_translator(self, registry: AgentRegistry):
        result = await run_agent(registry, "translator", {
            "text": "hello",
            "target_lang": "fr",
        })
        assert result.success
        assert "fr" in result.data["translated_text"]
        assert result.data["target_lang"] == "fr"

    async def test_run_formatter(self, registry: AgentRegistry):
        result = await run_agent(registry, "formatter", {
            "content": "hello world",
            "style": "uppercase",
        })
        assert result.success
        assert result.data["formatted"] == "HELLO WORLD"

    async def test_missing_agent_raises(self, registry: AgentRegistry):
        with pytest.raises(RunError, match="not found"):
            await run_agent(registry, "nonexistent", {})

    async def test_no_implementation_raises(self, tmp_path):
        """Agent with YAML but no agent.py raises RunError."""
        agent_dir = tmp_path / "no_impl"
        agent_dir.mkdir()
        (agent_dir / "agent.yaml").write_text(yaml.dump({
            "agent": {"name": "no-impl", "version": "1.0.0"}
        }))
        # No agent.py!

        reg = AgentRegistry(search_paths=[tmp_path])
        reg.discover()

        with pytest.raises(RunError, match="No agent implementation"):
            await run_agent(reg, "no-impl", {})

    async def test_invalid_input_rejected(self, registry: AgentRegistry):
        result = await run_agent(registry, "echo", {})  # Missing 'message'
        assert not result.success
        assert "Input validation" in result.error
        assert len(result.validation_errors) > 0

    async def test_context_passed_through(self, registry: AgentRegistry):
        ctx = AgentContext(job_id="test-123", chain_name="test-chain")
        result = await run_agent(
            registry, "echo", {"message": "hi"}, context=ctx
        )
        assert result.success

    async def test_agent_exception_handled(self, tmp_path):
        """Agent that raises during execute() returns clean error."""
        agent_dir = tmp_path / "crasher"
        agent_dir.mkdir()
        (agent_dir / "agent.yaml").write_text(yaml.dump({
            "agent": {
                "name": "crasher",
                "version": "1.0.0",
            }
        }))
        (agent_dir / "agent.py").write_text(
            "from atlas.runtime.base import AgentBase\n"
            "class CrasherAgent(AgentBase):\n"
            "    async def execute(self, input):\n"
            "        raise RuntimeError('boom')\n"
        )

        reg = AgentRegistry(search_paths=[tmp_path])
        reg.discover()

        result = await run_agent(reg, "crasher", {})
        assert not result.success
        assert "boom" in result.error


class TestExecutionTimeout:
    async def test_timeout_from_contract(self, tmp_path):
        """Agent with short timeout gets cancelled."""
        agent_dir = tmp_path / "sleeper"
        agent_dir.mkdir()
        (agent_dir / "agent.yaml").write_text(yaml.dump({
            "agent": {
                "name": "sleeper",
                "version": "1.0.0",
                "execution_timeout": 0.5,
            }
        }))
        (agent_dir / "agent.py").write_text(
            "import asyncio\n"
            "from atlas.runtime.base import AgentBase\n"
            "class SleeperAgent(AgentBase):\n"
            "    async def execute(self, input):\n"
            "        await asyncio.sleep(10)\n"
            "        return {}\n"
        )

        reg = AgentRegistry(search_paths=[tmp_path])
        reg.discover()

        result = await run_agent(reg, "sleeper", {})
        assert not result.success
        assert "timed out" in result.error

    async def test_fast_agent_completes(self, registry: AgentRegistry):
        """Agent within timeout completes normally."""
        result = await run_agent(registry, "echo", {"message": "fast"})
        assert result.success

    async def test_default_timeout_is_60(self, registry: AgentRegistry):
        """Default contract timeout is 60s."""
        entry = registry.get("echo")
        assert entry.contract.execution_timeout == 60.0
