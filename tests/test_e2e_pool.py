"""E2E integration tests — pool + store + metrics + traces wired together.

These tests exercise the full stack: submit jobs through the pool, verify
they persist to the store, metrics accumulate correctly, and traces are
created.  No mocking of internal components.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from atlas.contract.registry import AgentRegistry
from atlas.events import EventBus
from atlas.metrics import MetricsCollector
from atlas.pool.executor import ExecutionPool
from atlas.pool.job import JobData
from atlas.pool.queue import JobQueue, QueueFullError
from atlas.store.job_store import JobStore
from atlas.trace import TraceCollector

AGENTS_DIR = Path(__file__).parent.parent / "agents"


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bus():
    return EventBus()


@pytest.fixture
def registry():
    reg = AgentRegistry(search_paths=[AGENTS_DIR])
    reg.discover()
    return reg


@pytest.fixture
async def store(tmp_path):
    s = JobStore(str(tmp_path / "e2e.db"))
    await s.init()
    yield s
    await s.close()


@pytest.fixture
def metrics(bus):
    mc = MetricsCollector(bus)
    yield mc
    mc.close()


@pytest.fixture
def traces(bus):
    tc = TraceCollector(bus)
    yield tc
    tc.close()


@pytest.fixture
def queue(bus, store):
    return JobQueue(max_size=100, store=store, event_bus=bus)


@pytest.fixture
async def pool(registry, queue):
    p = ExecutionPool(registry, queue, max_concurrent=4, warm_pool_size=2)
    await p.start()
    yield p
    await p.stop(timeout=5.0)


# ---------------------------------------------------------------------------
# Pool + Store integration
# ---------------------------------------------------------------------------


class TestPoolStoreIntegration:
    """Jobs submitted through the pool should persist to the store."""

    async def test_completed_job_persisted(self, pool, queue, store):
        job = JobData(agent_name="echo", input_data={"message": "hello"})
        await pool.submit(job)
        result = await queue.wait_for_terminal(job.id, timeout=5.0)

        assert result is not None
        assert result.status == "completed"
        assert result.output_data == {"message": "hello"}

        # Verify persisted in store
        stored = await store.get(job.id)
        assert stored is not None
        assert stored.status == "completed"
        assert stored.output_data == {"message": "hello"}

    async def test_failed_job_persisted(self, pool, queue, store):
        job = JobData(agent_name="echo", input_data={})  # missing "message" → fails
        await pool.submit(job)
        result = await queue.wait_for_terminal(job.id, timeout=5.0)

        assert result is not None
        assert result.status == "failed"

        stored = await store.get(job.id)
        assert stored is not None
        assert stored.status == "failed"
        assert stored.error  # Should have an error message

    async def test_multiple_jobs_persisted(self, pool, queue, store):
        """Submit 10 jobs, all should end up in store."""
        jobs = [
            JobData(agent_name="echo", input_data={"message": f"msg-{i}"})
            for i in range(10)
        ]
        for job in jobs:
            await pool.submit(job)

        # Wait for all to complete
        for job in jobs:
            await queue.wait_for_terminal(job.id, timeout=10.0)

        # All in store
        stored = await store.list(agent_name="echo", limit=100)
        assert len(stored) >= 10

        for job in jobs:
            s = await store.get(job.id)
            assert s is not None
            assert s.status == "completed"

    async def test_store_has_timing_data(self, pool, queue, store):
        job = JobData(agent_name="echo", input_data={"message": "timing"})
        await pool.submit(job)
        await queue.wait_for_terminal(job.id, timeout=5.0)

        stored = await store.get(job.id)
        assert stored is not None
        assert stored.execution_ms >= 0  # Echo is fast, may be 0.0
        assert stored.started_at > 0
        assert stored.completed_at >= stored.started_at


# ---------------------------------------------------------------------------
# Pool + Metrics integration
# ---------------------------------------------------------------------------


class TestPoolMetricsIntegration:
    """Metrics should accumulate correctly from real pool executions."""

    async def test_metrics_track_completed_jobs(self, pool, queue, metrics):
        for i in range(5):
            job = JobData(agent_name="echo", input_data={"message": f"m{i}"})
            await pool.submit(job)
            await queue.wait_for_terminal(job.id, timeout=5.0)

        agent_m = metrics.get_agent_metrics("echo")
        assert agent_m is not None
        assert agent_m["jobs_by_status"]["completed"] == 5

    async def test_metrics_track_failures(self, pool, queue, metrics):
        job = JobData(agent_name="echo", input_data={})  # will fail
        await pool.submit(job)
        await queue.wait_for_terminal(job.id, timeout=5.0)

        agent_m = metrics.get_agent_metrics("echo")
        assert agent_m is not None
        assert agent_m["jobs_by_status"]["failed"] >= 1

    async def test_global_metrics_aggregate(self, pool, queue, metrics):
        # Mix agents
        j1 = JobData(agent_name="echo", input_data={"message": "a"})
        j2 = JobData(agent_name="formatter", input_data={"content": "b", "style": "uppercase"})
        await pool.submit(j1)
        await pool.submit(j2)
        await queue.wait_for_terminal(j1.id, timeout=5.0)
        await queue.wait_for_terminal(j2.id, timeout=5.0)

        g = metrics.get_global_metrics()
        assert g["total_jobs"] >= 2

    async def test_metrics_have_latency(self, pool, queue, metrics):
        job = JobData(agent_name="echo", input_data={"message": "lat"})
        await pool.submit(job)
        await queue.wait_for_terminal(job.id, timeout=5.0)

        agent_m = metrics.get_agent_metrics("echo")
        assert agent_m["latency_p50_ms"] >= 0  # Should have recorded some latency

    async def test_warm_hit_rate(self, pool, queue, metrics):
        """Second job to same agent should be warm hit."""
        j1 = JobData(agent_name="echo", input_data={"message": "first"})
        await pool.submit(j1)
        await queue.wait_for_terminal(j1.id, timeout=5.0)

        j2 = JobData(agent_name="echo", input_data={"message": "second"})
        await pool.submit(j2)
        await queue.wait_for_terminal(j2.id, timeout=5.0)

        agent_m = metrics.get_agent_metrics("echo")
        # At least one warm hit (second job reused slot)
        assert agent_m["warm_hit_rate"] > 0


# ---------------------------------------------------------------------------
# Pool + Traces integration
# ---------------------------------------------------------------------------


class TestPoolTracesIntegration:
    """Traces should be created from real pool executions."""

    async def test_trace_created_on_completion(self, pool, queue, traces):
        job = JobData(agent_name="echo", input_data={"message": "traced"})
        await pool.submit(job)
        await queue.wait_for_terminal(job.id, timeout=5.0)

        trace = traces.get(job.id)
        assert trace is not None
        assert trace.agent_name == "echo"
        assert trace.status == "completed"
        assert trace.execution_ms >= 0  # Echo is fast, may be 0.0

    async def test_trace_created_on_failure(self, pool, queue, traces):
        job = JobData(agent_name="echo", input_data={})
        await pool.submit(job)
        await queue.wait_for_terminal(job.id, timeout=5.0)

        trace = traces.get(job.id)
        assert trace is not None
        assert trace.status == "failed"

    async def test_multiple_traces(self, pool, queue, traces):
        ids = []
        for i in range(5):
            job = JobData(agent_name="echo", input_data={"message": f"t{i}"})
            await pool.submit(job)
            ids.append(job.id)

        for jid in ids:
            await queue.wait_for_terminal(jid, timeout=5.0)

        # Each job should have a trace after wait_for_terminal
        for jid in ids:
            assert traces.get(jid) is not None, f"Missing trace for {jid}"

    async def test_trace_filter_by_agent(self, pool, queue, traces):
        j1 = JobData(agent_name="echo", input_data={"message": "x"})
        j2 = JobData(agent_name="formatter", input_data={"content": "y", "style": "uppercase"})
        await pool.submit(j1)
        await pool.submit(j2)
        await queue.wait_for_terminal(j1.id, timeout=5.0)
        await queue.wait_for_terminal(j2.id, timeout=5.0)

        echo_traces = traces.list(agent_name="echo")
        assert all(t.agent_name == "echo" for t in echo_traces)


# ---------------------------------------------------------------------------
# Pool + Store + Metrics + Traces (full stack)
# ---------------------------------------------------------------------------


class TestFullStackIntegration:
    """All subscribers wired together on the same EventBus."""

    async def test_full_lifecycle(self, pool, queue, store, metrics, traces):
        """Single job flows through pool → store + metrics + traces."""
        job = JobData(agent_name="echo", input_data={"message": "full-stack"})
        await pool.submit(job)
        await queue.wait_for_terminal(job.id, timeout=5.0)

        # Store
        stored = await store.get(job.id)
        assert stored is not None
        assert stored.status == "completed"

        # Metrics
        agent_m = metrics.get_agent_metrics("echo")
        assert agent_m is not None
        assert agent_m["jobs_by_status"]["completed"] >= 1

        # Traces
        trace = traces.get(job.id)
        assert trace is not None
        assert trace.status == "completed"

    async def test_batch_integrity(self, pool, queue, store, metrics, traces):
        """20 jobs — all end up in store, metrics, and traces."""
        jobs = [
            JobData(agent_name="echo", input_data={"message": f"batch-{i}"})
            for i in range(20)
        ]
        for job in jobs:
            await pool.submit(job)

        for job in jobs:
            await queue.wait_for_terminal(job.id, timeout=15.0)

        # Store has all 20
        stored = await store.list(limit=100)
        stored_ids = {s.id for s in stored}
        for job in jobs:
            assert job.id in stored_ids

        # Metrics count is correct
        agent_m = metrics.get_agent_metrics("echo")
        assert agent_m["jobs_by_status"]["completed"] == 20

        # Traces exist for all
        all_traces = traces.list(limit=100)
        trace_ids = {t.job_id for t in all_traces}
        for job in jobs:
            assert job.id in trace_ids


# ---------------------------------------------------------------------------
# Queue edge cases
# ---------------------------------------------------------------------------


class TestQueueEdgeCases:
    async def test_duplicate_job_id_rejected(self, bus):
        queue = JobQueue(event_bus=bus)
        job1 = JobData(agent_name="echo", input_data={"message": "a"})
        await queue.submit(job1)

        # Same ID should raise
        job2 = JobData.__new__(JobData)
        job2.id = job1.id
        job2.agent_name = "echo"
        job2.status = "pending"
        job2.input_data = {"message": "b"}
        job2.output_data = None
        job2.error = ""
        job2.priority = 0
        job2.created_at = time.time()
        job2.started_at = 0.0
        job2.completed_at = 0.0
        job2.warmup_ms = 0.0
        job2.execution_ms = 0.0
        job2.retry_count = 0
        job2.original_job_id = ""
        job2.metadata = {}

        with pytest.raises(ValueError, match="Duplicate job ID"):
            await queue.submit(job2)

    async def test_cancel_frees_capacity(self, bus):
        queue = JobQueue(max_size=5, event_bus=bus)
        jobs = [JobData(agent_name="echo", input_data={"message": str(i)}) for i in range(5)]
        for j in jobs:
            await queue.submit(j)

        assert queue.capacity_remaining == 0

        # Cancel one frees capacity
        cancelled = await queue.cancel(jobs[0].id)
        assert cancelled is True
        assert queue.capacity_remaining == 1

        # Can submit one more
        extra = JobData(agent_name="echo", input_data={"message": "extra"})
        await queue.submit(extra)

    async def test_list_all(self, bus):
        queue = JobQueue(event_bus=bus)
        for i in range(3):
            await queue.submit(JobData(agent_name="echo", input_data={"message": str(i)}))

        all_jobs = queue.list_all()
        assert len(all_jobs) == 3


# ---------------------------------------------------------------------------
# Priority ordering through pool
# ---------------------------------------------------------------------------


class TestPoolPriorityOrdering:
    async def test_high_priority_runs_first(self, registry, bus, store):
        """Submit low then high priority — high should start first."""
        queue = JobQueue(max_size=100, store=store, event_bus=bus)
        pool = ExecutionPool(
            registry, queue, max_concurrent=1, warm_pool_size=0
        )

        # Submit BOTH before pool starts — ensures priority ordering
        low = JobData(agent_name="echo", input_data={"message": "low"}, priority=0)
        high = JobData(agent_name="echo", input_data={"message": "high"}, priority=10)
        await queue.submit(low)
        await queue.submit(high)

        await pool.start()

        try:
            await queue.wait_for_terminal(low.id, timeout=10.0)
            await queue.wait_for_terminal(high.id, timeout=10.0)

            low_job = queue.get(low.id)
            high_job = queue.get(high.id)
            assert low_job.status == "completed"
            assert high_job.status == "completed"

            # High priority should have started first (earlier started_at)
            assert high_job.started_at <= low_job.started_at
        finally:
            await pool.stop(timeout=5.0)
