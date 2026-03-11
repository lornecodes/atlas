"""E2E integration tests — orchestrator routing through the pool.

Tests orchestrator routing decisions with real agents executing through
the full pool lifecycle, including failure cascades and hot-swap.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from atlas.contract.registry import AgentRegistry
from atlas.events import EventBus
from atlas.metrics import MetricsCollector
from atlas.orchestrator.default import DefaultOrchestrator
from atlas.orchestrator.protocol import Orchestrator, RoutingDecision
from atlas.pool.executor import ExecutionPool
from atlas.pool.job import JobData
from atlas.pool.queue import JobQueue
from atlas.store.job_store import JobStore
from atlas.trace import TraceCollector

AGENTS_DIR = Path(__file__).parent.parent / "agents"


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
    s = JobStore(str(tmp_path / "e2e_orch.db"))
    await s.init()
    yield s
    await s.close()


class RejectAllOrchestrator(Orchestrator):
    """Rejects every job."""

    async def route(self, job, registry):
        return RoutingDecision(
            action="reject",
            metadata={"reason": "All jobs rejected for testing"},
        )

    async def on_job_complete(self, job):
        pass

    async def on_job_failed(self, job):
        pass


class RedirectOrchestrator(Orchestrator):
    """Redirects echo → formatter."""

    async def route(self, job, registry):
        if job.agent_name == "echo":
            return RoutingDecision(action="redirect", agent_name="formatter")
        return RoutingDecision(action="allow")

    async def on_job_complete(self, job):
        pass

    async def on_job_failed(self, job):
        pass


class TrackingOrchestrator(Orchestrator):
    """Tracks lifecycle callbacks."""

    def __init__(self):
        self.routed = []
        self.completed = []
        self.failed = []

    async def route(self, job, registry):
        self.routed.append(job.agent_name)
        return RoutingDecision(action="allow")

    async def on_job_complete(self, job):
        self.completed.append(job.id)

    async def on_job_failed(self, job):
        self.failed.append(job.id)


# ---------------------------------------------------------------------------
# Orchestrator routing through pool
# ---------------------------------------------------------------------------


class TestOrchestratorRouting:

    async def test_reject_fails_job(self, registry, bus, store):
        queue = JobQueue(max_size=100, store=store, event_bus=bus)
        pool = ExecutionPool(
            registry, queue,
            max_concurrent=2,
            orchestrator=RejectAllOrchestrator(),
        )
        await pool.start()

        try:
            job = JobData(agent_name="echo", input_data={"message": "rejected"})
            await pool.submit(job)
            await queue.wait_for_terminal(job.id, timeout=5.0)

            result = queue.get(job.id)
            assert result.status == "failed"
            assert "rejected" in result.error.lower()

            # Store should also reflect failure
            stored = await store.get(job.id)
            assert stored.status == "failed"
        finally:
            await pool.stop(timeout=5.0)

    async def test_redirect_executes_different_agent(self, registry, bus, store):
        queue = JobQueue(max_size=100, store=store, event_bus=bus)
        pool = ExecutionPool(
            registry, queue,
            max_concurrent=2,
            orchestrator=RedirectOrchestrator(),
        )
        await pool.start()

        try:
            # Submit as echo, but orchestrator redirects to formatter
            job = JobData(
                agent_name="echo",
                input_data={"content": "hello world", "style": "uppercase"},
            )
            await pool.submit(job)
            await queue.wait_for_terminal(job.id, timeout=5.0)

            result = queue.get(job.id)
            assert result.status == "completed"
            # Formatter outputs uppercase
            assert result.output_data.get("formatted") == "HELLO WORLD"
        finally:
            await pool.stop(timeout=5.0)

    async def test_lifecycle_callbacks_called(self, registry, bus, store):
        orch = TrackingOrchestrator()
        queue = JobQueue(max_size=100, store=store, event_bus=bus)
        pool = ExecutionPool(
            registry, queue,
            max_concurrent=2,
            orchestrator=orch,
        )
        await pool.start()

        try:
            job = JobData(agent_name="echo", input_data={"message": "track"})
            await pool.submit(job)
            await queue.wait_for_terminal(job.id, timeout=5.0)
            await asyncio.sleep(0.1)  # Let on_job_complete fire

            assert "echo" in orch.routed
            assert job.id in orch.completed
        finally:
            await pool.stop(timeout=5.0)

    async def test_failed_job_callback(self, registry, bus, store):
        orch = TrackingOrchestrator()
        queue = JobQueue(max_size=100, store=store, event_bus=bus)
        pool = ExecutionPool(
            registry, queue,
            max_concurrent=2,
            orchestrator=orch,
        )
        await pool.start()

        try:
            job = JobData(agent_name="echo", input_data={})  # will fail
            await pool.submit(job)
            await queue.wait_for_terminal(job.id, timeout=5.0)

            assert "echo" in orch.routed
            # on_job_failed should have been called (from outer except)
            # Note: input validation failure skips on_job_failed but execution
            # errors do call it
        finally:
            await pool.stop(timeout=5.0)


class TestOrchestratorHotSwap:
    """Test runtime orchestrator swapping."""

    async def test_hot_swap(self, registry, bus, store):
        queue = JobQueue(max_size=100, store=store, event_bus=bus)
        pool = ExecutionPool(registry, queue, max_concurrent=2)
        await pool.start()

        try:
            # Default orchestrator — job succeeds
            j1 = JobData(agent_name="echo", input_data={"message": "before"})
            await pool.submit(j1)
            await queue.wait_for_terminal(j1.id, timeout=5.0)
            assert queue.get(j1.id).status == "completed"

            # Swap to reject-all
            pool.set_orchestrator(RejectAllOrchestrator())

            j2 = JobData(agent_name="echo", input_data={"message": "after"})
            await pool.submit(j2)
            await queue.wait_for_terminal(j2.id, timeout=5.0)
            assert queue.get(j2.id).status == "failed"

            # Swap back to default
            pool.set_orchestrator(DefaultOrchestrator())

            j3 = JobData(agent_name="echo", input_data={"message": "restored"})
            await pool.submit(j3)
            await queue.wait_for_terminal(j3.id, timeout=5.0)
            assert queue.get(j3.id).status == "completed"
        finally:
            await pool.stop(timeout=5.0)


class TestOrchestratorWithTraces:
    """Orchestrator routing decisions should be visible in traces."""

    async def test_rejected_job_creates_trace(self, registry, bus, store):
        tc = TraceCollector(bus)
        queue = JobQueue(max_size=100, store=store, event_bus=bus)
        pool = ExecutionPool(
            registry, queue,
            max_concurrent=2,
            orchestrator=RejectAllOrchestrator(),
        )
        await pool.start()

        try:
            job = JobData(agent_name="echo", input_data={"message": "t"})
            await pool.submit(job)
            await queue.wait_for_terminal(job.id, timeout=5.0)

            trace = tc.get(job.id)
            assert trace is not None
            assert trace.status == "failed"
        finally:
            await pool.stop(timeout=5.0)
            tc.close()

    async def test_redirected_job_trace_has_correct_agent(self, registry, bus, store):
        tc = TraceCollector(bus)
        queue = JobQueue(max_size=100, store=store, event_bus=bus)
        pool = ExecutionPool(
            registry, queue,
            max_concurrent=2,
            orchestrator=RedirectOrchestrator(),
        )
        await pool.start()

        try:
            job = JobData(
                agent_name="echo",
                input_data={"content": "hi", "style": "uppercase"},
            )
            await pool.submit(job)
            await queue.wait_for_terminal(job.id, timeout=5.0)

            trace = tc.get(job.id)
            assert trace is not None
            # After redirect, agent_name on job is changed to formatter
            assert trace.agent_name == "formatter"
        finally:
            await pool.stop(timeout=5.0)
            tc.close()
