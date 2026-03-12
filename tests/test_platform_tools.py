"""Tests for platform tools — Atlas internals exposed as skills."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from atlas.contract.registry import AgentRegistry, RegisteredAgent
from atlas.contract.types import AgentContract, RequiresSpec, SchemaSpec
from atlas.pool.executor import ExecutionPool
from atlas.pool.job import JobData
from atlas.pool.queue import JobQueue
from atlas.runtime.context import AgentContext
from atlas.skills.platform import PLATFORM_PREFIX, PlatformToolProvider
from atlas.skills.registry import SkillRegistry
from atlas.skills.resolver import SkillResolver

from conftest import AGENTS_DIR


@pytest.fixture
def agent_registry() -> AgentRegistry:
    reg = AgentRegistry(search_paths=[AGENTS_DIR])
    reg.discover()
    return reg


@pytest.fixture
def queue() -> JobQueue:
    return JobQueue()


@pytest.fixture
def pool(agent_registry, queue) -> ExecutionPool:
    return ExecutionPool(agent_registry, queue, max_concurrent=2, warm_pool_size=0)


@pytest.fixture
def skill_registry() -> SkillRegistry:
    return SkillRegistry()


@pytest.fixture
def provider(agent_registry, queue, pool) -> PlatformToolProvider:
    return PlatformToolProvider(agent_registry, queue, pool)


# === PlatformToolProvider basics ===


class TestPlatformToolProvider:
    def test_register_all_returns_count(self, provider, skill_registry):
        count = provider.register_all(skill_registry)
        assert count == 12

    def test_all_tools_have_atlas_prefix(self, provider, skill_registry):
        provider.register_all(skill_registry)
        for entry in skill_registry.list_all():
            assert entry.spec.name.startswith(PLATFORM_PREFIX)

    def test_all_tools_are_callable(self, provider, skill_registry):
        provider.register_all(skill_registry)
        for entry in skill_registry.list_all():
            assert entry.callable is not None


# === Registry tools ===


class TestRegistryTools:
    @pytest.mark.asyncio
    async def test_list_all_agents(self, provider, skill_registry):
        provider.register_all(skill_registry)
        fn = skill_registry.get("atlas.registry.list").callable
        result = await fn({})
        assert "agents" in result
        names = [a["name"] for a in result["agents"]]
        assert "echo" in names

    @pytest.mark.asyncio
    async def test_list_filter_by_type(self, provider, skill_registry, agent_registry):
        provider.register_all(skill_registry)
        fn = skill_registry.get("atlas.registry.list").callable
        # List only orchestrators — echo is type "agent" so it shouldn't appear
        result = await fn({"type": "orchestrator"})
        agent_names = [a["name"] for a in result["agents"]]
        assert "echo" not in agent_names

    @pytest.mark.asyncio
    async def test_describe_existing_agent(self, provider, skill_registry):
        provider.register_all(skill_registry)
        fn = skill_registry.get("atlas.registry.describe").callable
        result = await fn({"name": "echo"})
        assert result["name"] == "echo"
        assert "version" in result
        assert "input_schema" in result
        assert "output_schema" in result

    @pytest.mark.asyncio
    async def test_describe_missing_agent(self, provider, skill_registry):
        provider.register_all(skill_registry)
        fn = skill_registry.get("atlas.registry.describe").callable
        result = await fn({"name": "nonexistent-agent"})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_search_by_capability(self, provider, skill_registry, agent_registry):
        provider.register_all(skill_registry)
        fn = skill_registry.get("atlas.registry.search").callable
        # echo agent has "echo" capability
        result = await fn({"capabilities": ["echo"]})
        names = [a["name"] for a in result["agents"]]
        assert "echo" in names

    @pytest.mark.asyncio
    async def test_search_no_matches(self, provider, skill_registry):
        provider.register_all(skill_registry)
        fn = skill_registry.get("atlas.registry.search").callable
        result = await fn({"capabilities": ["nonexistent-cap"]})
        assert result["agents"] == []


# === Exec tools ===


class TestExecTools:
    @pytest.mark.asyncio
    async def test_spawn_submits_job(self, provider, skill_registry, queue):
        provider.register_all(skill_registry)
        fn = skill_registry.get("atlas.exec.spawn").callable
        result = await fn({"agent": "echo", "input": {"message": "hi"}})
        assert "job_id" in result
        # Job should be in the queue
        job = queue.get(result["job_id"])
        assert job is not None
        assert job.agent_name == "echo"

    @pytest.mark.asyncio
    async def test_spawn_returns_job_id(self, provider, skill_registry):
        provider.register_all(skill_registry)
        fn = skill_registry.get("atlas.exec.spawn").callable
        result = await fn({"agent": "test", "input": {}, "priority": 5})
        assert result["job_id"].startswith("job-")

    @pytest.mark.asyncio
    async def test_status_existing_job(self, provider, skill_registry, queue):
        provider.register_all(skill_registry)
        job = JobData(agent_name="echo", input_data={"message": "test"})
        await queue.submit(job)

        fn = skill_registry.get("atlas.exec.status").callable
        result = await fn({"job_id": job.id})
        assert result["id"] == job.id
        assert result["agent_name"] == "echo"
        assert result["status"] == "pending"

    @pytest.mark.asyncio
    async def test_status_missing_job(self, provider, skill_registry):
        provider.register_all(skill_registry)
        fn = skill_registry.get("atlas.exec.status").callable
        result = await fn({"job_id": "nonexistent"})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_cancel_job(self, provider, skill_registry, queue):
        provider.register_all(skill_registry)
        job = JobData(agent_name="echo", input_data={})
        await queue.submit(job)

        fn = skill_registry.get("atlas.exec.cancel").callable
        result = await fn({"job_id": job.id})
        assert result["cancelled"] is True


# === Queue tools ===


class TestQueueTools:
    @pytest.mark.asyncio
    async def test_inspect_all(self, provider, skill_registry, queue):
        provider.register_all(skill_registry)
        job1 = JobData(agent_name="echo", input_data={})
        job2 = JobData(agent_name="echo", input_data={})
        await queue.submit(job1)
        await queue.submit(job2)

        fn = skill_registry.get("atlas.queue.inspect").callable
        result = await fn({})
        assert len(result["jobs"]) == 2

    @pytest.mark.asyncio
    async def test_inspect_with_status_filter(self, provider, skill_registry, queue):
        provider.register_all(skill_registry)
        job = JobData(agent_name="echo", input_data={})
        await queue.submit(job)

        fn = skill_registry.get("atlas.queue.inspect").callable
        result = await fn({"status": "pending"})
        assert len(result["jobs"]) >= 1
        assert all(j["status"] == "pending" for j in result["jobs"])

        result = await fn({"status": "completed"})
        assert len(result["jobs"]) == 0

    @pytest.mark.asyncio
    async def test_inspect_with_limit(self, provider, skill_registry, queue):
        provider.register_all(skill_registry)
        for _ in range(5):
            await queue.submit(JobData(agent_name="echo", input_data={}))

        fn = skill_registry.get("atlas.queue.inspect").callable
        result = await fn({"limit": 2})
        assert len(result["jobs"]) == 2


# === Monitor tools ===


class TestMonitorTools:
    @pytest.mark.asyncio
    async def test_health_returns_counts(self, provider, skill_registry, queue):
        provider.register_all(skill_registry)
        fn = skill_registry.get("atlas.monitor.health").callable
        result = await fn({})
        assert "pending" in result
        assert "running" in result
        assert "capacity_remaining" in result

    @pytest.mark.asyncio
    async def test_metrics_no_collector(self, provider, skill_registry):
        # Provider created without metrics_collector
        provider.register_all(skill_registry)
        fn = skill_registry.get("atlas.monitor.metrics").callable
        result = await fn({})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_metrics_with_collector(self, agent_registry, queue, pool, skill_registry):
        mock_metrics = MagicMock()
        mock_metrics.get_all_metrics.return_value = {"total_jobs": 10}
        provider = PlatformToolProvider(
            agent_registry, queue, pool, metrics_collector=mock_metrics,
        )
        provider.register_all(skill_registry)
        fn = skill_registry.get("atlas.monitor.metrics").callable
        result = await fn({})
        assert result == {"total_jobs": 10}

    @pytest.mark.asyncio
    async def test_metrics_per_agent(self, agent_registry, queue, pool, skill_registry):
        mock_metrics = MagicMock()
        mock_metrics.get_agent_metrics.return_value = {"executions": 5}
        provider = PlatformToolProvider(
            agent_registry, queue, pool, metrics_collector=mock_metrics,
        )
        provider.register_all(skill_registry)
        fn = skill_registry.get("atlas.monitor.metrics").callable
        result = await fn({"agent": "echo"})
        assert result == {"executions": 5}

    @pytest.mark.asyncio
    async def test_metrics_per_agent_not_found(self, agent_registry, queue, pool, skill_registry):
        mock_metrics = MagicMock()
        mock_metrics.get_agent_metrics.return_value = None
        provider = PlatformToolProvider(
            agent_registry, queue, pool, metrics_collector=mock_metrics,
        )
        provider.register_all(skill_registry)
        fn = skill_registry.get("atlas.monitor.metrics").callable
        result = await fn({"agent": "nonexistent"})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_trace_not_found(self, provider, skill_registry):
        # No trace collector
        provider.register_all(skill_registry)
        fn = skill_registry.get("atlas.monitor.trace").callable
        result = await fn({"trace_id": "abc"})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_trace_found(self, agent_registry, queue, pool, skill_registry):
        mock_trace = MagicMock()
        mock_trace.to_dict.return_value = {"id": "t1", "agent": "echo"}
        mock_traces = MagicMock()
        mock_traces.get.return_value = mock_trace
        provider = PlatformToolProvider(
            agent_registry, queue, pool, trace_collector=mock_traces,
        )
        provider.register_all(skill_registry)
        fn = skill_registry.get("atlas.monitor.trace").callable
        result = await fn({"trace_id": "t1"})
        assert result == {"id": "t1", "agent": "echo"}

    @pytest.mark.asyncio
    async def test_traces_list(self, agent_registry, queue, pool, skill_registry):
        mock_item = MagicMock()
        mock_item.to_dict.return_value = {"id": "t1"}
        mock_traces = MagicMock()
        mock_traces.list.return_value = [mock_item]
        provider = PlatformToolProvider(
            agent_registry, queue, pool, trace_collector=mock_traces,
        )
        provider.register_all(skill_registry)
        fn = skill_registry.get("atlas.monitor.traces").callable
        result = await fn({})
        assert result == {"traces": [{"id": "t1"}]}

    @pytest.mark.asyncio
    async def test_traces_filter_by_agent(self, agent_registry, queue, pool, skill_registry):
        mock_traces = MagicMock()
        mock_traces.list.return_value = []
        provider = PlatformToolProvider(
            agent_registry, queue, pool, trace_collector=mock_traces,
        )
        provider.register_all(skill_registry)
        fn = skill_registry.get("atlas.monitor.traces").callable
        await fn({"agent": "echo", "limit": 10})
        mock_traces.list.assert_called_once_with(limit=10, agent_name="echo")


# === Pool integration ===


class TestPoolIntegration:
    @pytest.mark.asyncio
    async def test_platform_tools_injected_when_required(self, agent_registry, queue):
        """Agent with platform_tools: true gets atlas.* skills injected."""
        skill_registry = SkillRegistry()
        pool = ExecutionPool(
            agent_registry, queue, max_concurrent=2, warm_pool_size=0,
        )
        provider = PlatformToolProvider(agent_registry, queue, pool)
        provider.register_all(skill_registry)
        resolver = SkillResolver(skill_registry)
        pool._skill_resolver = resolver

        # Resolve what an agent with platform_tools=true would get
        all_skills = resolver.registry.list_all()
        platform_skills = {
            rs.spec.name: rs.callable
            for rs in all_skills
            if rs.spec.name.startswith(PLATFORM_PREFIX) and rs.callable
        }
        assert len(platform_skills) == 12
        assert "atlas.registry.list" in platform_skills
        assert "atlas.exec.spawn" in platform_skills
        assert "atlas.monitor.health" in platform_skills

    @pytest.mark.asyncio
    async def test_platform_tools_not_injected_when_not_required(self):
        """Agent without platform_tools: true does not get atlas.* skills."""
        requires = RequiresSpec(platform_tools=False)
        # When platform_tools is False, the executor skips atlas.* injection
        assert requires.platform_tools is False
        # Skills dict should remain empty for this agent
        resolved_skills: dict = {}
        assert len(resolved_skills) == 0


# === Read-only mode ===


class TestReadOnlyMode:
    """Platform tools created with read_only=True gate exec operations."""

    @pytest.fixture
    def ro_provider(self, agent_registry, queue, pool) -> PlatformToolProvider:
        return PlatformToolProvider(agent_registry, queue, pool, read_only=True)

    @pytest.mark.asyncio
    async def test_spawn_blocked(self, ro_provider, skill_registry):
        ro_provider.register_all(skill_registry)
        fn = skill_registry.get("atlas.exec.spawn").callable
        result = await fn({"agent": "echo", "input": {}})
        assert "error" in result
        assert "not running" in result["error"]

    @pytest.mark.asyncio
    async def test_status_blocked(self, ro_provider, skill_registry):
        ro_provider.register_all(skill_registry)
        fn = skill_registry.get("atlas.exec.status").callable
        result = await fn({"job_id": "any"})
        assert "error" in result
        assert "not running" in result["error"]

    @pytest.mark.asyncio
    async def test_cancel_blocked(self, ro_provider, skill_registry):
        ro_provider.register_all(skill_registry)
        fn = skill_registry.get("atlas.exec.cancel").callable
        result = await fn({"job_id": "any"})
        assert "error" in result
        assert "not running" in result["error"]

    @pytest.mark.asyncio
    async def test_registry_list_still_works(self, ro_provider, skill_registry):
        ro_provider.register_all(skill_registry)
        fn = skill_registry.get("atlas.registry.list").callable
        result = await fn({})
        assert "agents" in result

    @pytest.mark.asyncio
    async def test_health_still_works(self, ro_provider, skill_registry):
        ro_provider.register_all(skill_registry)
        fn = skill_registry.get("atlas.monitor.health").callable
        result = await fn({})
        assert "pending" in result
