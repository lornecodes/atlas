"""Tests for agent spawning — Phase 7."""

from __future__ import annotations

import asyncio

import pytest

from atlas.contract.registry import AgentRegistry
from atlas.pool.executor import ExecutionPool
from atlas.pool.job import JobData
from atlas.pool.queue import JobQueue
from atlas.runtime.context import AgentContext, SpawnError, SpawnResult


# === SpawnResult Tests ===


class TestSpawnResult:
    def test_success_result(self):
        r = SpawnResult(success=True, data={"key": "val"})
        assert r.success is True
        assert r.data == {"key": "val"}
        assert r.error == ""

    def test_failure_result(self):
        r = SpawnResult(success=False, error="boom")
        assert r.success is False
        assert r.error == "boom"

    def test_defaults(self):
        r = SpawnResult(success=True)
        assert r.data == {}
        assert r.error == ""


# === AgentContext.spawn() Guard Tests ===


class TestSpawnGuards:
    async def test_spawn_not_allowed_raises(self):
        ctx = AgentContext(spawn_allowed=False)
        with pytest.raises(SpawnError, match="spawn_agents permission"):
            await ctx.spawn("echo", {"message": "hi"})

    async def test_spawn_depth_exceeded_raises(self):
        ctx = AgentContext(spawn_allowed=True, depth=3, max_depth=3)
        with pytest.raises(SpawnError, match="Max spawn depth"):
            await ctx.spawn("echo", {"message": "hi"})

    async def test_spawn_no_callback_raises(self):
        ctx = AgentContext(spawn_allowed=True, depth=0, max_depth=3)
        with pytest.raises(SpawnError, match="No spawn callback"):
            await ctx.spawn("echo", {"message": "hi"})

    async def test_spawn_with_callback_succeeds(self):
        async def fake_spawn(agent_name, input_data, priority, parent_depth, parent_job_id=""):
            return SpawnResult(success=True, data={"echoed": input_data})

        ctx = AgentContext(
            spawn_allowed=True, depth=0, max_depth=3,
            _spawn_callback=fake_spawn,
        )
        result = await ctx.spawn("echo", {"message": "hi"})
        assert result.success is True
        assert result.data == {"echoed": {"message": "hi"}}

    async def test_spawn_increments_depth_in_callback(self):
        received_depth = None

        async def capture_spawn(agent_name, input_data, priority, parent_depth, parent_job_id=""):
            nonlocal received_depth
            received_depth = parent_depth
            return SpawnResult(success=True)

        ctx = AgentContext(
            spawn_allowed=True, depth=2, max_depth=5,
            _spawn_callback=capture_spawn,
        )
        await ctx.spawn("echo", {"message": "hi"})
        assert received_depth == 2  # parent depth passed to callback

    async def test_spawn_at_boundary_succeeds(self):
        """depth=2, max_depth=3 should succeed (2 < 3)."""
        async def fake_spawn(agent_name, input_data, priority, parent_depth, parent_job_id=""):
            return SpawnResult(success=True)

        ctx = AgentContext(
            spawn_allowed=True, depth=2, max_depth=3,
            _spawn_callback=fake_spawn,
        )
        result = await ctx.spawn("echo", {})
        assert result.success is True

    async def test_spawn_passes_priority(self):
        received_priority = None

        async def capture_spawn(agent_name, input_data, priority, parent_depth, parent_job_id=""):
            nonlocal received_priority
            received_priority = priority
            return SpawnResult(success=True)

        ctx = AgentContext(
            spawn_allowed=True, depth=0, max_depth=3,
            _spawn_callback=capture_spawn,
        )
        await ctx.spawn("echo", {}, priority=42)
        assert received_priority == 42


# === Pool Integration: Decomposer Agent ===


@pytest.fixture
def queue():
    return JobQueue()


class TestDecomposerAgent:
    async def test_decomposer_spawns_echo(self, registry, queue):
        """Decomposer spawns echo for each message and collects results."""
        pool = ExecutionPool(
            registry, queue, max_concurrent=4, warm_pool_size=0,
        )
        await pool.start()
        try:
            job = JobData(
                agent_name="decomposer",
                input_data={"messages": ["hello", "world"]},
            )
            await pool.submit(job)
            # Give enough time for parent + 2 child spawns
            await asyncio.sleep(1.0)

            result = queue.get(job.id)
            assert result is not None
            assert result.status == "completed", f"Job failed: {result.error}"
            assert result.output_data["count"] == 2
            results = result.output_data["results"]
            assert results[0]["success"] is True
            assert results[0]["data"]["message"] == "hello"
            assert results[1]["success"] is True
            assert results[1]["data"]["message"] == "world"
        finally:
            await pool.stop()

    async def test_decomposer_single_message(self, registry, queue):
        pool = ExecutionPool(
            registry, queue, max_concurrent=4, warm_pool_size=0,
        )
        await pool.start()
        try:
            job = JobData(
                agent_name="decomposer",
                input_data={"messages": ["solo"]},
            )
            await pool.submit(job)
            await asyncio.sleep(0.5)

            result = queue.get(job.id)
            assert result.status == "completed"
            assert result.output_data["count"] == 1
        finally:
            await pool.stop()

    async def test_decomposer_empty_list(self, registry, queue):
        pool = ExecutionPool(
            registry, queue, max_concurrent=4, warm_pool_size=0,
        )
        await pool.start()
        try:
            job = JobData(
                agent_name="decomposer",
                input_data={"messages": []},
            )
            await pool.submit(job)
            await asyncio.sleep(0.3)

            result = queue.get(job.id)
            assert result.status == "completed"
            assert result.output_data["count"] == 0
            assert result.output_data["results"] == []
        finally:
            await pool.stop()


# === Spawn Permission Enforcement ===


class TestSpawnPermission:
    async def test_agent_without_spawn_permission_fails(self, registry, queue):
        """Echo agent doesn't have spawn_agents — calling spawn should fail."""
        # We test this via context guard, not pool integration,
        # because echo doesn't call spawn() in its execute().
        entry = registry.get("echo")
        assert entry is not None
        assert entry.contract.requires.spawn_agents is False

    async def test_decomposer_has_spawn_permission(self, registry):
        entry = registry.get("decomposer")
        assert entry is not None
        assert entry.contract.requires.spawn_agents is True


# === Spawn Depth Enforcement in Pool ===


class TestSpawnDepthInPool:
    async def test_child_inherits_incremented_depth(self, registry, queue):
        """When decomposer spawns echo, the child gets depth=1."""
        pool = ExecutionPool(
            registry, queue, max_concurrent=4, warm_pool_size=0,
        )
        await pool.start()
        try:
            job = JobData(
                agent_name="decomposer",
                input_data={"messages": ["test"]},
            )
            await pool.submit(job)
            await asyncio.sleep(0.5)

            # The parent job should complete
            result = queue.get(job.id)
            assert result.status == "completed"

            # Find the child job — it was submitted with _spawn_depth=1
            completed_jobs = queue.list_by_status("completed")
            child_jobs = [
                j for j in completed_jobs
                if j.id != job.id and j.agent_name == "echo"
            ]
            assert len(child_jobs) >= 1
            child = child_jobs[0]
            assert child.metadata.get("_spawn_depth") == 1
        finally:
            await pool.stop()

    async def test_spawn_depth_metadata_at_top_level(self, registry, queue):
        """Top-level job has no _spawn_depth metadata (defaults to 0)."""
        pool = ExecutionPool(
            registry, queue, max_concurrent=4, warm_pool_size=0,
        )
        await pool.start()
        try:
            job = JobData(
                agent_name="echo",
                input_data={"message": "top"},
            )
            await pool.submit(job)
            await asyncio.sleep(0.3)

            result = queue.get(job.id)
            assert result.status == "completed"
            # No _spawn_depth in metadata means depth 0
            assert result.metadata.get("_spawn_depth", 0) == 0
        finally:
            await pool.stop()
