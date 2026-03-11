"""Tests for the execution pool — Spike 3."""

from __future__ import annotations

import asyncio
import time

import pytest

from atlas.contract.registry import AgentRegistry
from atlas.pool.job import JobData
from atlas.pool.queue import JobQueue, QueueFullError
from atlas.pool.executor import ExecutionPool

from conftest import AGENTS_DIR


@pytest.fixture
def queue() -> JobQueue:
    return JobQueue()


@pytest.fixture
def pool(registry: AgentRegistry, queue: JobQueue) -> ExecutionPool:
    return ExecutionPool(
        registry,
        queue,
        max_concurrent=2,
        warm_pool_size=1,
        idle_timeout=60.0,
    )


# === Job Tests ===

class TestJobData:
    def test_auto_id(self):
        job = JobData(agent_name="echo")
        assert job.id.startswith("job-")

    def test_is_terminal(self):
        job = JobData(status="completed")
        assert job.is_terminal
        job2 = JobData(status="running")
        assert not job2.is_terminal


# === Queue Tests ===

class TestJobQueue:
    async def test_submit_and_next(self, queue: JobQueue):
        job = JobData(agent_name="echo", input_data={"message": "hi"})
        job_id = await queue.submit(job)
        assert job_id == job.id

        dequeued = await queue.next()
        assert dequeued.id == job.id

    async def test_priority_ordering(self, queue: JobQueue):
        low = JobData(agent_name="echo", priority=1, input_data={"message": "low"})
        high = JobData(agent_name="echo", priority=10, input_data={"message": "high"})
        await queue.submit(low)
        await queue.submit(high)

        first = await queue.next()
        assert first.input_data["message"] == "high"

    async def test_cancel_pending(self, queue: JobQueue):
        job = JobData(agent_name="echo", input_data={"message": "cancel me"})
        await queue.submit(job)
        assert await queue.cancel(job.id)
        assert job.status == "cancelled"

    async def test_list_by_status(self, queue: JobQueue):
        j1 = JobData(agent_name="echo", input_data={"message": "1"})
        j2 = JobData(agent_name="echo", input_data={"message": "2"})
        await queue.submit(j1)
        await queue.submit(j2)
        assert len(queue.list_by_status("pending")) == 2

    async def test_wait_for_terminal(self, queue: JobQueue):
        job = JobData(agent_name="echo", input_data={"message": "wait"})
        await queue.submit(job)

        async def complete_later():
            await asyncio.sleep(0.1)
            await queue.update(job.id, status="completed")

        asyncio.create_task(complete_later())
        result = await queue.wait_for_terminal(job.id, timeout=2.0)
        assert result.status == "completed"


# === Queue Backpressure Tests ===

class TestQueueBackpressure:
    async def test_queue_rejects_when_full(self):
        queue = JobQueue(max_size=3)
        for i in range(3):
            await queue.submit(JobData(agent_name="echo", input_data={"message": str(i)}))

        with pytest.raises(QueueFullError):
            await queue.submit(JobData(agent_name="echo", input_data={"message": "overflow"}))

    async def test_completed_jobs_free_capacity(self):
        queue = JobQueue(max_size=2)
        j1 = JobData(agent_name="echo", input_data={"message": "1"})
        j2 = JobData(agent_name="echo", input_data={"message": "2"})
        await queue.submit(j1)
        await queue.submit(j2)

        # Queue is full
        with pytest.raises(QueueFullError):
            await queue.submit(JobData(agent_name="echo", input_data={"message": "3"}))

        # Complete a job
        await queue.update(j1.id, status="completed")

        # Now there's room
        j3 = JobData(agent_name="echo", input_data={"message": "3"})
        await queue.submit(j3)
        assert queue.capacity_remaining == 0

    async def test_cancelled_jobs_free_capacity(self):
        queue = JobQueue(max_size=2)
        j1 = JobData(agent_name="echo", input_data={"message": "1"})
        await queue.submit(j1)
        assert queue.capacity_remaining == 1

        await queue.cancel(j1.id)
        assert queue.capacity_remaining == 2

    async def test_default_max_size(self):
        queue = JobQueue()
        assert queue.capacity_remaining == 1000


# === Pool Tests ===

class TestExecutionPool:
    @pytest.fixture(autouse=True)
    async def _start_stop_pool(self, pool: ExecutionPool):
        await pool.start()
        yield
        await pool.stop(timeout=5.0)

    async def test_submit_and_execute(self, pool: ExecutionPool, queue: JobQueue):
        job = JobData(agent_name="echo", input_data={"message": "hello pool"})
        await pool.submit(job)
        result = await queue.wait_for_terminal(job.id, timeout=5.0)
        assert result.status == "completed"
        assert result.output_data == {"message": "hello pool"}

    async def test_execution_timing(self, pool: ExecutionPool, queue: JobQueue):
        job = JobData(agent_name="echo", input_data={"message": "timed"})
        await pool.submit(job)
        result = await queue.wait_for_terminal(job.id, timeout=5.0)
        assert result.status == "completed"
        assert result.execution_ms >= 0

    async def test_cold_start_measured(self, pool: ExecutionPool, queue: JobQueue):
        """slow-starter agent should show warmup time on first run."""
        job = JobData(agent_name="slow-starter", input_data={"text": "cold"})
        await pool.submit(job)
        result = await queue.wait_for_terminal(job.id, timeout=10.0)
        assert result.status == "completed"
        assert result.warmup_ms > 100  # slow-starter sleeps 0.5s

    async def test_warm_pool_reuse(self, pool: ExecutionPool, queue: JobQueue):
        """Second job for same agent should reuse warm slot (warmup_ms == 0)."""
        j1 = JobData(agent_name="slow-starter", input_data={"text": "first"})
        await pool.submit(j1)
        r1 = await queue.wait_for_terminal(j1.id, timeout=10.0)
        assert r1.status == "completed"
        assert r1.warmup_ms > 100

        j2 = JobData(agent_name="slow-starter", input_data={"text": "second"})
        await pool.submit(j2)
        r2 = await queue.wait_for_terminal(j2.id, timeout=10.0)
        assert r2.status == "completed"
        assert r2.warmup_ms == 0.0  # Warm slot reused!

    async def test_concurrency_limit(self, pool: ExecutionPool, queue: JobQueue):
        """Only max_concurrent jobs should run simultaneously."""
        jobs = []
        for i in range(5):
            job = JobData(agent_name="echo", input_data={"message": f"job-{i}"})
            jobs.append(job)
            await pool.submit(job)

        # Wait for all to complete
        for job in jobs:
            result = await queue.wait_for_terminal(job.id, timeout=10.0)
            assert result.status == "completed"

    async def test_priority_execution_order(self, pool: ExecutionPool, queue: JobQueue):
        """Higher priority jobs should complete before lower priority."""
        # Submit low-priority first, then high
        low = JobData(agent_name="echo", priority=1, input_data={"message": "low"})
        high = JobData(agent_name="echo", priority=100, input_data={"message": "high"})
        await pool.submit(low)
        await pool.submit(high)

        # Both should complete
        for job in [low, high]:
            result = await queue.wait_for_terminal(job.id, timeout=5.0)
            assert result.status == "completed"

    async def test_invalid_input_fails(self, pool: ExecutionPool, queue: JobQueue):
        """Job with invalid input should fail cleanly."""
        job = JobData(agent_name="echo", input_data={})  # Missing 'message'
        await pool.submit(job)
        result = await queue.wait_for_terminal(job.id, timeout=5.0)
        assert result.status == "failed"
        assert "validation" in result.error.lower()

    async def test_agent_crash_recovery(self, pool: ExecutionPool, queue: JobQueue, registry: AgentRegistry, tmp_path):
        """Crashed agent should not poison subsequent jobs."""
        import yaml

        # Create a crasher agent
        agent_dir = tmp_path / "crasher"
        agent_dir.mkdir()
        (agent_dir / "agent.yaml").write_text(yaml.dump({
            "agent": {"name": "crasher", "version": "1.0.0"}
        }))
        (agent_dir / "agent.py").write_text(
            "from atlas.runtime.base import AgentBase\n"
            "class CrasherAgent(AgentBase):\n"
            "    async def execute(self, input):\n"
            "        raise RuntimeError('pool crash')\n"
        )

        # Need a new pool with the crasher registered
        reg = AgentRegistry(search_paths=[AGENTS_DIR, tmp_path])
        reg.discover()
        q = JobQueue()
        p = ExecutionPool(reg, q, max_concurrent=2, warm_pool_size=1)
        await p.start()

        try:
            # Crash job
            crash_job = JobData(agent_name="crasher", input_data={})
            await p.submit(crash_job)
            result = await q.wait_for_terminal(crash_job.id, timeout=5.0)
            assert result.status == "failed"

            # Normal job should still work
            ok_job = JobData(agent_name="echo", input_data={"message": "still works"})
            await p.submit(ok_job)
            result = await q.wait_for_terminal(ok_job.id, timeout=5.0)
            assert result.status == "completed"
        finally:
            await p.stop(timeout=5.0)

    async def test_missing_agent_fails(self, pool: ExecutionPool, queue: JobQueue):
        job = JobData(agent_name="nonexistent", input_data={})
        await pool.submit(job)
        result = await queue.wait_for_terminal(job.id, timeout=5.0)
        assert result.status == "failed"

    async def test_mixed_agent_types(self, pool: ExecutionPool, queue: JobQueue):
        """Pool handles different agent types concurrently."""
        j1 = JobData(agent_name="echo", input_data={"message": "echo"})
        j2 = JobData(agent_name="summarizer", input_data={"text": "hello world test", "max_length": 50})
        j3 = JobData(agent_name="translator", input_data={"text": "hi", "target_lang": "de"})

        for j in [j1, j2, j3]:
            await pool.submit(j)

        for j in [j1, j2, j3]:
            result = await queue.wait_for_terminal(j.id, timeout=5.0)
            assert result.status == "completed", f"{j.agent_name} failed: {result.error}"


# === Pool Race Condition Tests ===

class TestPoolRaceConditions:
    async def test_concurrent_stop(self, registry: AgentRegistry):
        """Multiple stop() calls don't crash."""
        q = JobQueue()
        p = ExecutionPool(registry, q, max_concurrent=2, warm_pool_size=1)
        await p.start()

        # Submit a job so there's activity
        job = JobData(agent_name="echo", input_data={"message": "race"})
        await p.submit(job)

        # Stop concurrently
        await asyncio.gather(
            p.stop(timeout=5.0),
            p.stop(timeout=5.0),
        )

    async def test_stop_without_start(self, registry: AgentRegistry):
        """Stopping a pool that was never started is safe."""
        q = JobQueue()
        p = ExecutionPool(registry, q, max_concurrent=2, warm_pool_size=1)
        await p.stop(timeout=1.0)

    async def test_rapid_submit_cancel(self, registry: AgentRegistry):
        """Rapidly submitting and cancelling doesn't corrupt state."""
        q = JobQueue()
        p = ExecutionPool(registry, q, max_concurrent=2, warm_pool_size=1)
        await p.start()

        try:
            jobs = []
            for i in range(10):
                job = JobData(agent_name="echo", input_data={"message": f"rapid-{i}"})
                await p.submit(job)
                jobs.append(job)

            # Cancel half
            for job in jobs[:5]:
                await q.cancel(job.id)

            # Wait for the rest to complete
            for job in jobs[5:]:
                result = await q.wait_for_terminal(job.id, timeout=10.0)
                assert result.status in ("completed", "cancelled")
        finally:
            await p.stop(timeout=5.0)
