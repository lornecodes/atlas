"""Tests for MCP remote agents — federated chain support (Phase 10C)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from atlas.contract.registry import AgentRegistry, RegisteredAgent
from atlas.contract.types import AgentContract, RequiresSpec, SchemaSpec
from atlas.mcp.remote_agents import (
    RemoteAgent,
    RemoteAgentProvider,
    _build_contract,
    _make_remote_agent_class,
)
from atlas.runtime.base import AgentBase
from atlas.runtime.context import AgentContext
from atlas.skills.platform import PlatformToolProvider
from atlas.skills.registry import SkillRegistry
from atlas.skills.resolver import SkillResolver
from atlas.skills.types import SkillSpec

from conftest import AGENTS_DIR


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def agent_registry() -> AgentRegistry:
    reg = AgentRegistry(search_paths=[AGENTS_DIR])
    reg.discover()
    return reg


@pytest.fixture
def skill_registry() -> SkillRegistry:
    return SkillRegistry()


# ---------------------------------------------------------------------------
# TestAgentRegistryVirtual
# ---------------------------------------------------------------------------

class TestAgentRegistryVirtual:
    def test_register_virtual(self):
        reg = AgentRegistry()
        contract = AgentContract(name="test-virtual", version="1.0.0")

        class DummyAgent(AgentBase):
            async def execute(self, input_data):
                return {"done": True}

        reg.register_virtual(contract, DummyAgent)
        entry = reg.get("test-virtual")
        assert entry is not None
        assert entry.contract.name == "test-virtual"
        assert entry.agent_class is DummyAgent

    def test_register_virtual_source_path_is_virtual(self):
        reg = AgentRegistry()
        contract = AgentContract(name="virt", version="1.0.0")
        reg.register_virtual(contract, AgentBase)
        entry = reg.get("virt")
        assert entry.source_path == Path("<virtual>")

    def test_unregister_returns_true(self):
        reg = AgentRegistry()
        contract = AgentContract(name="to-remove", version="1.0.0")
        reg.register_virtual(contract, AgentBase)
        assert reg.unregister("to-remove") is True
        assert reg.get("to-remove") is None

    def test_unregister_missing_returns_false(self):
        reg = AgentRegistry()
        assert reg.unregister("nonexistent") is False

    def test_virtual_agent_in_list_all(self):
        reg = AgentRegistry()
        contract = AgentContract(name="listed", version="1.0.0")
        reg.register_virtual(contract, AgentBase)
        names = [a.contract.name for a in reg.list_all()]
        assert "listed" in names

    def test_virtual_agent_in_contains(self):
        reg = AgentRegistry()
        contract = AgentContract(name="contained", version="1.0.0")
        reg.register_virtual(contract, AgentBase)
        assert "contained" in reg

    def test_unregister_removes_from_contains(self):
        reg = AgentRegistry()
        contract = AgentContract(name="removable", version="1.0.0")
        reg.register_virtual(contract, AgentBase)
        reg.unregister("removable")
        assert "removable" not in reg


# ---------------------------------------------------------------------------
# TestExecRunTool
# ---------------------------------------------------------------------------

class TestExecRunTool:
    @pytest.fixture
    def platform_provider(self, agent_registry):
        from atlas.pool.queue import JobQueue
        from atlas.pool.executor import ExecutionPool
        queue = JobQueue()
        pool = ExecutionPool(agent_registry, queue, max_concurrent=1, warm_pool_size=0)
        return PlatformToolProvider(agent_registry, queue, pool)

    def test_exec_run_registered(self, platform_provider, skill_registry):
        platform_provider.register_all(skill_registry)
        entry = skill_registry.get("atlas.exec.run")
        assert entry is not None
        assert entry.callable is not None

    def test_platform_tool_count_is_12(self, platform_provider, skill_registry):
        count = platform_provider.register_all(skill_registry)
        assert count == 12

    async def test_exec_run_executes_agent(self, platform_provider, skill_registry, agent_registry):
        platform_provider.register_all(skill_registry)
        fn = skill_registry.get("atlas.exec.run").callable
        result = await fn({"agent": "echo", "input": {"message": "hello"}})
        assert result["success"] is True
        assert result["data"]["message"] == "hello"
        assert result["agent_name"] == "echo"

    async def test_exec_run_unknown_agent(self, platform_provider, skill_registry):
        platform_provider.register_all(skill_registry)
        fn = skill_registry.get("atlas.exec.run").callable
        result = await fn({"agent": "nonexistent", "input": {}})
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    async def test_exec_run_missing_agent_param(self, platform_provider, skill_registry):
        platform_provider.register_all(skill_registry)
        fn = skill_registry.get("atlas.exec.run").callable
        result = await fn({"input": {}})
        assert result["success"] is False
        assert "Missing" in result["error"]

    async def test_exec_run_default_empty_input(self, platform_provider, skill_registry):
        platform_provider.register_all(skill_registry)
        fn = skill_registry.get("atlas.exec.run").callable
        # No "input" key — should default to {}; echo requires "message" so it fails validation
        result = await fn({"agent": "echo"})
        assert result["success"] is False
        assert "validation" in result["error"].lower()


# ---------------------------------------------------------------------------
# TestRemoteAgent
# ---------------------------------------------------------------------------

class TestRemoteAgent:
    async def test_execute_calls_skill(self):
        mock_skill = AsyncMock(return_value={
            "success": True,
            "data": {"translated": "bonjour"},
            "error": "",
        })
        cls = _make_remote_agent_class(mock_skill, "translator")
        contract = AgentContract(name="lab.translator", version="1.0.0")
        agent = cls(contract, AgentContext())
        result = await agent.execute({"text": "hello"})
        assert result == {"translated": "bonjour"}
        mock_skill.assert_called_once_with({
            "agent": "translator",
            "input": {"text": "hello"},
        })

    async def test_execute_raises_on_failure(self):
        mock_skill = AsyncMock(return_value={
            "success": False,
            "data": {},
            "error": "Agent crashed",
        })
        cls = _make_remote_agent_class(mock_skill, "bad-agent")
        contract = AgentContract(name="lab.bad-agent", version="1.0.0")
        agent = cls(contract, AgentContext())
        with pytest.raises(RuntimeError, match="Agent crashed"):
            await agent.execute({})

    async def test_execute_returns_empty_dict_on_missing_data(self):
        mock_skill = AsyncMock(return_value={
            "success": True,
            "error": "",
        })
        cls = _make_remote_agent_class(mock_skill, "minimal")
        contract = AgentContract(name="lab.minimal", version="1.0.0")
        agent = cls(contract, AgentContext())
        result = await agent.execute({})
        assert result == {}

    def test_is_agent_base_subclass(self):
        cls = _make_remote_agent_class(AsyncMock(), "x")
        assert issubclass(cls, AgentBase)
        assert issubclass(cls, RemoteAgent)

    def test_different_agents_get_different_classes(self):
        cls1 = _make_remote_agent_class(AsyncMock(), "agent-a")
        cls2 = _make_remote_agent_class(AsyncMock(), "agent-b")
        assert cls1 is not cls2
        assert cls1._remote_agent_name == "agent-a"
        assert cls2._remote_agent_name == "agent-b"


# ---------------------------------------------------------------------------
# TestBuildContract
# ---------------------------------------------------------------------------

class TestBuildContract:
    def test_builds_from_describe_response(self):
        details = {
            "name": "translator",
            "version": "2.0.0",
            "type": "agent",
            "description": "Translates text",
            "input_schema": {"type": "object", "properties": {"text": {"type": "string"}}},
            "output_schema": {"type": "object", "properties": {"result": {"type": "string"}}},
            "capabilities": ["translation"],
            "execution_timeout": 90.0,
        }
        contract = _build_contract("lab.translator", details)
        assert contract.name == "lab.translator"
        assert contract.version == "2.0.0"
        assert contract.description == "Translates text"
        assert "translation" in contract.capabilities
        assert contract.execution_timeout == 90.0
        assert contract.input_schema.properties.get("text") == {"type": "string"}

    def test_defaults_for_missing_fields(self):
        contract = _build_contract("lab.minimal", {})
        assert contract.name == "lab.minimal"
        assert contract.version == "1.0.0"
        assert contract.type == "agent"
        assert contract.execution_timeout == 120.0


# ---------------------------------------------------------------------------
# TestRemoteAgentProvider
# ---------------------------------------------------------------------------

class TestRemoteAgentProvider:
    def _setup_skill_registry_with_platform_tools(
        self, skill_registry, agents_list, describe_map=None,
    ):
        """Register mock platform tools as if they came from a remote server."""
        prefix = "lab"

        # Mock atlas.registry.list
        async def mock_list(input_data):
            return {"agents": agents_list}

        skill_registry.register_callable(
            SkillSpec(name=f"{prefix}.atlas.registry.list", version="1.0.0"),
            mock_list,
        )

        # Mock atlas.registry.describe
        async def mock_describe(input_data):
            name = input_data.get("name", "")
            if describe_map and name in describe_map:
                return describe_map[name]
            return {"name": name, "version": "1.0.0"}

        skill_registry.register_callable(
            SkillSpec(name=f"{prefix}.atlas.registry.describe", version="1.0.0"),
            mock_describe,
        )

        # Mock atlas.exec.run
        async def mock_exec_run(input_data):
            return {
                "success": True,
                "data": {"echo": input_data.get("input", {})},
                "error": "",
                "agent_name": input_data.get("agent", ""),
            }

        skill_registry.register_callable(
            SkillSpec(name=f"{prefix}.atlas.exec.run", version="1.0.0"),
            mock_exec_run,
        )

    def _make_server(self, name="lab"):
        from atlas.mcp.client import RemoteServer
        return RemoteServer(name=name, url="http://localhost:8401/mcp")

    async def test_connect_discovers_agents(self, skill_registry):
        agents_list = [
            {"name": "translator", "version": "1.0.0", "type": "agent", "description": "Translates"},
            {"name": "embedder", "version": "1.0.0", "type": "agent", "description": "Embeds"},
        ]
        self._setup_skill_registry_with_platform_tools(skill_registry, agents_list)
        reg = AgentRegistry()
        provider = RemoteAgentProvider()
        count = await provider.connect(self._make_server(), reg, skill_registry)
        assert count == 2

    async def test_agents_namespaced(self, skill_registry):
        agents_list = [{"name": "translator", "version": "1.0.0"}]
        self._setup_skill_registry_with_platform_tools(skill_registry, agents_list)
        reg = AgentRegistry()
        provider = RemoteAgentProvider()
        await provider.connect(self._make_server(), reg, skill_registry)
        assert reg.get("lab.translator") is not None
        assert reg.get("translator") is None

    async def test_virtual_agents_have_contracts(self, skill_registry):
        agents_list = [{"name": "translator", "version": "1.0.0"}]
        describe_map = {
            "translator": {
                "name": "translator",
                "version": "2.0.0",
                "description": "Translates text",
                "capabilities": ["translation"],
            },
        }
        self._setup_skill_registry_with_platform_tools(
            skill_registry, agents_list, describe_map,
        )
        reg = AgentRegistry()
        provider = RemoteAgentProvider()
        await provider.connect(self._make_server(), reg, skill_registry)
        entry = reg.get("lab.translator")
        assert entry.contract.version == "2.0.0"
        assert entry.contract.description == "Translates text"
        assert "translation" in entry.contract.capabilities

    async def test_virtual_agent_is_executable(self, skill_registry):
        agents_list = [{"name": "echo-remote", "version": "1.0.0"}]
        self._setup_skill_registry_with_platform_tools(skill_registry, agents_list)
        reg = AgentRegistry()
        provider = RemoteAgentProvider()
        await provider.connect(self._make_server(), reg, skill_registry)
        entry = reg.get("lab.echo-remote")
        # Instantiate and execute the virtual agent
        agent = entry.agent_class(entry.contract, AgentContext())
        result = await agent.execute({"message": "test"})
        assert result == {"echo": {"message": "test"}}

    async def test_disconnect_removes_agents(self, skill_registry):
        agents_list = [{"name": "agent-a", "version": "1.0.0"}]
        self._setup_skill_registry_with_platform_tools(skill_registry, agents_list)
        reg = AgentRegistry()
        provider = RemoteAgentProvider()
        await provider.connect(self._make_server(), reg, skill_registry)
        assert "lab.agent-a" in reg
        provider.disconnect("lab", reg)
        assert "lab.agent-a" not in reg

    async def test_disconnect_all(self, skill_registry):
        agents_list = [{"name": "agent-a", "version": "1.0.0"}]
        self._setup_skill_registry_with_platform_tools(skill_registry, agents_list)
        reg = AgentRegistry()
        provider = RemoteAgentProvider()
        await provider.connect(self._make_server(), reg, skill_registry)
        provider.disconnect_all(reg)
        assert len(reg) == 0
        assert provider.connected_servers == []

    async def test_duplicate_server_name_errors(self, skill_registry):
        agents_list = [{"name": "a", "version": "1.0.0"}]
        self._setup_skill_registry_with_platform_tools(skill_registry, agents_list)
        reg = AgentRegistry()
        provider = RemoteAgentProvider()
        await provider.connect(self._make_server(), reg, skill_registry)
        with pytest.raises(ValueError, match="Already connected"):
            await provider.connect(self._make_server(), reg, skill_registry)

    async def test_missing_list_skill_raises(self, skill_registry):
        reg = AgentRegistry()
        provider = RemoteAgentProvider()
        with pytest.raises(ValueError, match="atlas.registry.list"):
            await provider.connect(self._make_server(), reg, skill_registry)

    async def test_missing_exec_run_raises(self, skill_registry):
        # Register list but not exec.run
        async def mock_list(input_data):
            return {"agents": []}

        skill_registry.register_callable(
            SkillSpec(name="lab.atlas.registry.list", version="1.0.0"),
            mock_list,
        )
        reg = AgentRegistry()
        provider = RemoteAgentProvider()
        with pytest.raises(ValueError, match="atlas.exec.run"):
            await provider.connect(self._make_server(), reg, skill_registry)

    async def test_connected_servers_property(self, skill_registry):
        agents_list = [{"name": "a", "version": "1.0.0"}]
        self._setup_skill_registry_with_platform_tools(skill_registry, agents_list)
        reg = AgentRegistry()
        provider = RemoteAgentProvider()
        assert provider.connected_servers == []
        await provider.connect(self._make_server(), reg, skill_registry)
        assert provider.connected_servers == ["lab"]


# ---------------------------------------------------------------------------
# TestChainRunnerSkillInjection
# ---------------------------------------------------------------------------

class TestChainRunnerSkillInjection:
    async def test_chain_without_resolver_works(self, agent_registry):
        """Backward compat: ChainRunner without skill_resolver still works."""
        from atlas.chains.definition import ChainDefinition, ChainStep
        from atlas.chains.runner import ChainRunner
        from atlas.mediation.engine import MediationEngine

        runner = ChainRunner(agent_registry, MediationEngine())
        chain = ChainDefinition(
            name="no-resolver",
            steps=[ChainStep(agent_name="echo")],
        )
        result = await runner.execute(chain, {"message": "hi"})
        assert result.success
        assert result.output["message"] == "hi"

    async def test_chain_with_resolver_injects_skills(self, agent_registry):
        """ChainRunner with skill_resolver resolves agent skills."""
        from atlas.chains.definition import ChainDefinition, ChainStep
        from atlas.chains.runner import ChainRunner
        from atlas.mediation.engine import MediationEngine

        skill_reg = SkillRegistry()
        resolver = SkillResolver(skill_reg)
        runner = ChainRunner(agent_registry, MediationEngine(), skill_resolver=resolver)

        # echo agent doesn't require skills, so this just tests the code path runs
        chain = ChainDefinition(
            name="with-resolver",
            steps=[ChainStep(agent_name="echo")],
        )
        result = await runner.execute(chain, {"message": "hello"})
        assert result.success


# ---------------------------------------------------------------------------
# TestCliIntegration
# ---------------------------------------------------------------------------

class TestCliIntegration:
    def test_serve_command_has_remote_param(self):
        from atlas.cli.app import serve
        import inspect
        sig = inspect.signature(serve)
        assert "remote" in sig.parameters

    def test_mcp_command_has_remote_param(self):
        from atlas.cli.app import mcp_server
        import inspect
        sig = inspect.signature(mcp_server)
        assert "remote" in sig.parameters
