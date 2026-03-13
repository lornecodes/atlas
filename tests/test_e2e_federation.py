"""E2E tests — MCP federation and remote agents.

Tests RemoteAgentProvider: discovery, registration, execution, and
disconnect lifecycle. Uses in-process skill stubs instead of real
MCP connections.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from atlas.contract.registry import AgentRegistry
from atlas.contract.types import AgentContract, SchemaSpec
from atlas.mcp.remote_agents import RemoteAgent, RemoteAgentProvider
from atlas.skills.registry import SkillRegistry
from atlas.skills.types import SkillSpec

AGENTS_DIR = Path(__file__).parent.parent / "agents"


# ---------------------------------------------------------------------------
# Fake remote server + skill stubs
# ---------------------------------------------------------------------------


@dataclass
class FakeRemoteServer:
    """Mimics the RemoteServer interface for testing."""

    name: str


def _make_platform_skills(
    server_name: str,
    agents: list[dict[str, Any]],
) -> dict[str, Any]:
    """Create fake platform skill callables for a remote server.

    Returns skills that should be registered in the SkillRegistry:
    - {server_name}.atlas.registry.list → returns agent list
    - {server_name}.atlas.registry.describe → returns agent details
    - {server_name}.atlas.exec.run → executes agent locally (echo behavior)
    """

    async def list_agents(input_data: dict) -> dict:
        return {"agents": agents}

    async def describe_agent(input_data: dict) -> dict:
        name = input_data.get("name", "")
        for a in agents:
            if a["name"] == name:
                return a
        return {"name": name, "description": f"Remote agent: {name}"}

    async def exec_run(input_data: dict) -> dict:
        agent_name = input_data.get("agent", "")
        agent_input = input_data.get("input", {})
        # Simulated remote execution — echo with server prefix
        return {
            "success": True,
            "data": {
                "result": f"{server_name}:{agent_name}:{agent_input.get('message', '')}",
            },
        }

    return {
        f"{server_name}.atlas.registry.list": (
            SkillSpec(name=f"{server_name}.atlas.registry.list", version="1.0.0"),
            list_agents,
        ),
        f"{server_name}.atlas.registry.describe": (
            SkillSpec(name=f"{server_name}.atlas.registry.describe", version="1.0.0"),
            describe_agent,
        ),
        f"{server_name}.atlas.exec.run": (
            SkillSpec(name=f"{server_name}.atlas.exec.run", version="1.0.0"),
            exec_run,
        ),
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def agent_registry():
    """Local AgentRegistry with test agents."""
    reg = AgentRegistry(search_paths=[AGENTS_DIR])
    reg.discover()
    return reg


@pytest.fixture
def skill_registry():
    """Shared SkillRegistry for platform tool registration."""
    return SkillRegistry()


@pytest.fixture
def remote_provider():
    return RemoteAgentProvider()


@pytest.fixture
def lab_server():
    return FakeRemoteServer(name="lab")


@pytest.fixture
def staging_server():
    return FakeRemoteServer(name="staging")


def _register_platform_skills(
    skill_registry: SkillRegistry,
    server_name: str,
    agents: list[dict[str, Any]],
) -> None:
    """Register fake platform skills for a remote server."""
    skills = _make_platform_skills(server_name, agents)
    for name, (spec, fn) in skills.items():
        skill_registry.register_callable(spec, fn)


# ---------------------------------------------------------------------------
# Tests: Remote agent discovery and registration
# ---------------------------------------------------------------------------


class TestRemoteAgentDiscovery:
    """RemoteAgentProvider discovery and registration lifecycle."""

    @pytest.mark.asyncio
    async def test_discover_registers_remote_agents(
        self, remote_provider, agent_registry, skill_registry, lab_server
    ):
        """connect() discovers and registers remote agents."""
        _register_platform_skills(skill_registry, "lab", [
            {"name": "translator", "version": "1.0.0", "description": "Translates text"},
            {"name": "summarizer", "version": "1.0.0", "description": "Summarizes text"},
        ])

        count = await remote_provider.connect(lab_server, agent_registry, skill_registry)

        assert count == 2
        assert agent_registry.get("lab.translator") is not None
        assert agent_registry.get("lab.summarizer") is not None

    @pytest.mark.asyncio
    async def test_connected_servers_tracking(
        self, remote_provider, agent_registry, skill_registry, lab_server
    ):
        """connected_servers property tracks connected servers."""
        _register_platform_skills(skill_registry, "lab", [
            {"name": "agent1", "version": "1.0.0"},
        ])

        assert remote_provider.connected_servers == []
        await remote_provider.connect(lab_server, agent_registry, skill_registry)
        assert "lab" in remote_provider.connected_servers

    @pytest.mark.asyncio
    async def test_disconnect_removes_agents(
        self, remote_provider, agent_registry, skill_registry, lab_server
    ):
        """disconnect() removes remote agents from registry."""
        _register_platform_skills(skill_registry, "lab", [
            {"name": "worker", "version": "1.0.0"},
        ])
        await remote_provider.connect(lab_server, agent_registry, skill_registry)
        assert agent_registry.get("lab.worker") is not None

        remote_provider.disconnect("lab", agent_registry)

        assert agent_registry.get("lab.worker") is None
        assert "lab" not in remote_provider.connected_servers

    @pytest.mark.asyncio
    async def test_disconnect_all(
        self, remote_provider, agent_registry, skill_registry, lab_server, staging_server
    ):
        """disconnect_all() clears all remote agents."""
        _register_platform_skills(skill_registry, "lab", [
            {"name": "a1", "version": "1.0.0"},
        ])
        _register_platform_skills(skill_registry, "staging", [
            {"name": "a2", "version": "1.0.0"},
        ])
        await remote_provider.connect(lab_server, agent_registry, skill_registry)
        await remote_provider.connect(staging_server, agent_registry, skill_registry)

        assert len(remote_provider.connected_servers) == 2

        remote_provider.disconnect_all(agent_registry)

        assert remote_provider.connected_servers == []
        assert agent_registry.get("lab.a1") is None
        assert agent_registry.get("staging.a2") is None


# ---------------------------------------------------------------------------
# Tests: Remote agent execution
# ---------------------------------------------------------------------------


class TestRemoteAgentExecution:
    """Remote agents execute via platform tool proxy."""

    @pytest.mark.asyncio
    async def test_remote_agent_execution(
        self, remote_provider, agent_registry, skill_registry, lab_server
    ):
        """Remote agent executes via exec.run platform tool."""
        _register_platform_skills(skill_registry, "lab", [
            {"name": "echo-remote", "version": "1.0.0"},
        ])
        await remote_provider.connect(lab_server, agent_registry, skill_registry)

        entry = agent_registry.get("lab.echo-remote")
        assert entry is not None

        # Instantiate and execute the remote agent
        agent_cls = entry.agent_class
        agent = agent_cls(entry.contract, None)
        result = await agent.execute({"message": "test-data"})

        assert result["result"] == "lab:echo-remote:test-data"

    @pytest.mark.asyncio
    async def test_remote_agent_failure(
        self, remote_provider, agent_registry, skill_registry
    ):
        """Remote agent failure is propagated."""
        server = FakeRemoteServer(name="failing")

        # Register platform skills with a failing exec.run
        async def failing_exec(input_data: dict) -> dict:
            return {"success": False, "error": "Connection refused"}

        skill_registry.register_callable(
            SkillSpec(name="failing.atlas.registry.list", version="1.0.0"),
            lambda _: asyncio.coroutine(lambda: {"agents": [{"name": "worker", "version": "1.0.0"}]})(),
        )
        # Register proper list skill
        async def list_fn(input_data):
            return {"agents": [{"name": "worker", "version": "1.0.0"}]}

        skill_registry.register_callable(
            SkillSpec(name="failing.atlas.registry.list", version="1.0.0"),
            list_fn,
        )
        skill_registry.register_callable(
            SkillSpec(name="failing.atlas.exec.run", version="1.0.0"),
            failing_exec,
        )

        await remote_provider.connect(server, agent_registry, skill_registry)

        entry = agent_registry.get("failing.worker")
        agent = entry.agent_class(entry.contract, None)

        with pytest.raises(RuntimeError, match="failed"):
            await agent.execute({"message": "test"})

    @pytest.mark.asyncio
    async def test_multiple_remote_servers(
        self, remote_provider, agent_registry, skill_registry, lab_server, staging_server
    ):
        """Agents from multiple remote servers coexist."""
        _register_platform_skills(skill_registry, "lab", [
            {"name": "worker", "version": "1.0.0"},
        ])
        _register_platform_skills(skill_registry, "staging", [
            {"name": "worker", "version": "2.0.0"},
        ])

        await remote_provider.connect(lab_server, agent_registry, skill_registry)
        await remote_provider.connect(staging_server, agent_registry, skill_registry)

        # Both namespaced agents exist
        lab_entry = agent_registry.get("lab.worker")
        staging_entry = agent_registry.get("staging.worker")
        assert lab_entry is not None
        assert staging_entry is not None

        # Execute both — they route to different "servers"
        lab_agent = lab_entry.agent_class(lab_entry.contract, None)
        lab_result = await lab_agent.execute({"message": "hi"})
        assert "lab:" in lab_result["result"]

        staging_agent = staging_entry.agent_class(staging_entry.contract, None)
        staging_result = await staging_agent.execute({"message": "hi"})
        assert "staging:" in staging_result["result"]

    @pytest.mark.asyncio
    async def test_duplicate_connect_raises(
        self, remote_provider, agent_registry, skill_registry, lab_server
    ):
        """Connecting the same server twice raises ValueError."""
        _register_platform_skills(skill_registry, "lab", [
            {"name": "agent1", "version": "1.0.0"},
        ])
        await remote_provider.connect(lab_server, agent_registry, skill_registry)

        with pytest.raises(ValueError, match="Already connected"):
            await remote_provider.connect(lab_server, agent_registry, skill_registry)
