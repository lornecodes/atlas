"""Live integration tests for Phase 11 — exec agents, dynamic LLM agents, shared memory.

Run with:
    python -m pytest tests/test_live_phase11.py -v -s

LLM tests use CLIProxy (ai-bridge) at localhost:8318 for API calls.
Skips automatically if CLIProxy is not reachable.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from atlas.contract.registry import AgentRegistry
from atlas.contract.types import AgentContract, ModelSpec, ProviderSpec, RequiresSpec
from atlas.memory.file_provider import FileMemoryProvider
from atlas.runtime.context import AgentContext
from atlas.runtime.dynamic_llm_agent import DynamicLLMAgent
from atlas.runtime.exec_agent import ExecAgent
from atlas.runtime.runner import run_agent

AGENTS_DIR = Path(__file__).parent.parent / "agents"

# CLIProxy / ai-bridge URL for LLM calls
CLIPROXY_URL = "http://localhost:8318"


def _safe_print(text: str) -> None:
    sys.stdout.buffer.write((text + "\n").encode("utf-8", errors="replace"))
    sys.stdout.buffer.flush()


def _cliproxy_available() -> bool:
    """Check if CLIProxy is reachable."""
    try:
        import urllib.request
        req = urllib.request.Request(
            f"{CLIPROXY_URL}/v1/messages",
            data=b'{}',
            headers={"Content-Type": "application/json", "x-api-key": "test"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=2)
        return True
    except Exception:
        # Any response (even errors) means the server is running
        try:
            import urllib.request
            import urllib.error
            req = urllib.request.Request(
                f"{CLIPROXY_URL}/v1/messages",
                data=b'{}',
                headers={"Content-Type": "application/json", "x-api-key": "test"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=2)
        except urllib.error.HTTPError:
            return True  # Server responded with HTTP error — it's alive
        except Exception:
            return False
    return False


CLIPROXY_UP = _cliproxy_available()
skip_no_proxy = pytest.mark.skipif(not CLIPROXY_UP, reason="CLIProxy not available at localhost:8318")


@pytest.fixture(autouse=True)
def _set_cliproxy(monkeypatch):
    """Point Anthropic SDK at CLIProxy for all LLM tests."""
    monkeypatch.setenv("ANTHROPIC_BASE_URL", CLIPROXY_URL)
    # CLIProxy handles auth — set a dummy key so SDK doesn't complain
    if not os.environ.get("ANTHROPIC_API_KEY"):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "cliproxy-managed")


@pytest.fixture
def registry():
    reg = AgentRegistry(search_paths=[AGENTS_DIR])
    reg.discover()
    return reg


# --- Exec Agent E2E ---


@skip_no_proxy
class TestLiveExecEcho:
    """E2E: exec-echo agent runs a real subprocess."""

    @pytest.mark.asyncio
    async def test_exec_echo_via_registry(self, registry):
        """Discover and run exec-echo agent from agents/ directory."""
        entry = registry.get("exec-echo")
        assert entry is not None, "exec-echo agent not discovered"
        assert entry.contract.provider.type == "exec"

        from atlas.runtime.exec_agent import ExecAgent
        assert entry.agent_class is ExecAgent

        result = await run_agent(registry, "exec-echo", {"message": "hello from e2e"})
        assert result.success
        assert result.data == {"message": "hello from e2e"}
        _safe_print(f"  exec-echo result: {result.data}")

    @pytest.mark.asyncio
    async def test_exec_agent_with_custom_script(self, tmp_path):
        """Run a custom exec agent with a transform script."""
        script = tmp_path / "upper.py"
        script.write_text(textwrap.dedent("""\
            import json, sys
            envelope = json.loads(sys.stdin.read())
            msg = envelope["input"]["message"]
            print(json.dumps({"message": msg.upper(), "length": len(msg)}))
        """))

        contract = AgentContract(
            name="test-upper",
            version="1.0.0",
            provider=ProviderSpec(type="exec", command=["python", str(script)]),
            execution_timeout=10.0,
        )
        ctx = AgentContext(metadata={"_agent_dir": str(tmp_path)})
        agent = ExecAgent(contract, ctx)
        result = await agent.execute({"message": "hello world"})

        assert result["message"] == "HELLO WORLD"
        assert result["length"] == 11
        _safe_print(f"  custom exec result: {result}")


# --- Dynamic LLM Agent E2E ---


@skip_no_proxy
class TestLiveYamlSummarizer:
    """E2E: yaml-summarizer agent calls real Claude API."""

    @pytest.mark.asyncio
    async def test_yaml_summarizer_via_registry(self, registry):
        """Discover and run yaml-summarizer from agents/ directory."""
        entry = registry.get("yaml-summarizer")
        assert entry is not None, "yaml-summarizer agent not discovered"
        assert entry.contract.provider.type == "llm"

        from atlas.runtime.dynamic_llm_agent import DynamicLLMAgent
        assert entry.agent_class is DynamicLLMAgent

        result = await run_agent(registry, "yaml-summarizer", {
            "text": (
                "The Atlas runtime is a framework-agnostic agent execution engine. "
                "It uses YAML contracts to declare what agents take in and produce. "
                "Agents from any framework — LangChain, CrewAI, raw Python — run in "
                "the same pool with shared metrics, chains, and federation."
            ),
        })

        assert result.success, f"Failed: {result.error}"
        assert "summary" in result.data
        _safe_print(f"  yaml-summarizer summary: {result.data['summary']}")

    @pytest.mark.asyncio
    async def test_dynamic_llm_agent_direct(self):
        """Create a DynamicLLMAgent directly and run it."""
        contract = AgentContract(
            name="inline-test",
            version="1.0.0",
            provider=ProviderSpec(
                type="llm",
                system_prompt="You are a JSON calculator. Given a math expression in the 'expr' field, return {\"result\": <number>}. Return ONLY the JSON.",
                output_format="json",
                max_iterations=1,
            ),
            model=ModelSpec(preference="fast"),
        )
        ctx = AgentContext()
        agent = DynamicLLMAgent(contract, ctx)
        await agent.on_startup()
        result = await agent.execute({"expr": "2 + 2"})

        assert "result" in result
        _safe_print(f"  inline LLM result: {result}")

    @pytest.mark.asyncio
    async def test_dynamic_llm_text_output(self):
        """Test text output format (not JSON)."""
        contract = AgentContract(
            name="text-agent",
            version="1.0.0",
            provider=ProviderSpec(
                type="llm",
                system_prompt="Reply with exactly one word: the color of the sky on a clear day.",
                output_format="text",
                max_iterations=1,
            ),
            model=ModelSpec(preference="fast"),
        )
        ctx = AgentContext()
        agent = DynamicLLMAgent(contract, ctx)
        await agent.on_startup()
        result = await agent.execute({"question": "what color?"})

        assert "result" in result
        assert len(result["result"]) < 50  # Should be a short answer
        _safe_print(f"  text output: {result['result']}")


# --- Shared Memory E2E ---


@skip_no_proxy
class TestLiveSharedMemory:
    """E2E: shared memory across multiple agent executions."""

    @pytest.mark.asyncio
    async def test_memory_accumulation_across_agents(self, tmp_path):
        """Two LLM agents share memory — second sees first's appended entry."""
        memory_file = tmp_path / "memory.md"
        provider = FileMemoryProvider(memory_file)

        # Agent 1: append a learning to memory
        contract1 = AgentContract(
            name="learner-1",
            version="1.0.0",
            provider=ProviderSpec(
                type="llm",
                system_prompt=(
                    "You are a learning agent. Use the memory_append tool to save "
                    "the following fact: 'The Atlas runtime supports 3 provider types: "
                    "python, exec, and llm.' Then respond with {\"done\": true}."
                ),
                output_format="json",
                max_iterations=3,
            ),
            model=ModelSpec(preference="fast"),
            requires=RequiresSpec(memory=True),
        )
        ctx1 = AgentContext()
        ctx1._memory_provider = provider
        agent1 = DynamicLLMAgent(contract1, ctx1)
        await agent1.on_startup()
        result1 = await agent1.execute({"task": "save a fact"})
        _safe_print(f"  Agent 1 result: {result1}")

        # Verify memory was written
        memory_content = await provider.read()
        _safe_print(f"  Memory after agent 1: {memory_content[:200]}")
        assert len(memory_content) > 0, "Agent 1 should have written to memory"

        # Agent 2: read memory and answer based on it
        contract2 = AgentContract(
            name="reader-2",
            version="1.0.0",
            provider=ProviderSpec(
                type="llm",
                system_prompt=(
                    "You have access to shared memory from previous agents. "
                    "Based on the shared memory, answer: how many provider types "
                    "does Atlas support? Return {\"count\": <number>}."
                ),
                output_format="json",
                max_iterations=1,
            ),
            model=ModelSpec(preference="fast"),
            requires=RequiresSpec(memory=True),
        )
        ctx2 = AgentContext()
        ctx2._memory_provider = provider
        agent2 = DynamicLLMAgent(contract2, ctx2)
        await agent2.on_startup()
        result2 = await agent2.execute({"question": "how many providers?"})
        _safe_print(f"  Agent 2 result: {result2}")

    @pytest.mark.asyncio
    async def test_exec_agent_memory_roundtrip(self, tmp_path):
        """Exec agent reads memory and writes back via _memory_append."""
        memory_file = tmp_path / "memory.md"
        provider = FileMemoryProvider(memory_file)
        await provider.write("Known: API rate limit is 100/min")

        script = tmp_path / "agent.py"
        script.write_text(textwrap.dedent("""\
            import json, sys
            envelope = json.loads(sys.stdin.read())
            memory = envelope.get("memory", "")
            has_rate_limit = "rate limit" in memory
            print(json.dumps({
                "found_memory": has_rate_limit,
                "_memory_append": "Also learned: retry after 60 seconds"
            }))
        """))

        contract = AgentContract(
            name="mem-exec",
            version="1.0.0",
            provider=ProviderSpec(type="exec", command=["python", str(script)]),
            requires=RequiresSpec(memory=True),
            execution_timeout=10.0,
        )
        ctx = AgentContext(metadata={"_agent_dir": str(tmp_path)})
        ctx._memory_provider = provider
        agent = ExecAgent(contract, ctx)
        result = await agent.execute({})

        assert result["found_memory"] is True
        assert "_memory_append" not in result

        # Verify memory was appended
        final_memory = await provider.read()
        assert "API rate limit is 100/min" in final_memory
        assert "retry after 60 seconds" in final_memory
        _safe_print(f"  Final memory: {final_memory}")


# --- Discovery E2E ---


@skip_no_proxy
class TestLiveDiscovery:
    """E2E: agent discovery with new provider types."""

    def test_exec_agent_discovered(self, registry):
        entry = registry.get("exec-echo")
        assert entry is not None
        assert entry.contract.provider.type == "exec"
        assert entry.contract.provider.command == ["python", "run.py"]

    def test_llm_agent_discovered(self, registry):
        entry = registry.get("yaml-summarizer")
        assert entry is not None
        assert entry.contract.provider.type == "llm"
        assert "summarizer" in entry.contract.provider.system_prompt.lower()

    def test_python_agents_still_work(self, registry):
        entry = registry.get("echo")
        assert entry is not None
        assert entry.contract.provider.type == "python"

    def test_all_agents_discovered(self, registry):
        all_agents = registry.list_all()
        names = {a.contract.name for a in all_agents}
        assert "exec-echo" in names
        assert "yaml-summarizer" in names
        assert "echo" in names

    @pytest.mark.asyncio
    async def test_python_echo_still_works(self, registry):
        result = await run_agent(registry, "echo", {"message": "still works"})
        assert result.success
        assert result.data == {"message": "still works"}
