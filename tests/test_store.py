"""Tests for SQLite job persistence."""

from __future__ import annotations

import time

import pytest

from atlas.pool.job import JobData
from atlas.store.job_store import JobStore


@pytest.fixture
async def store(tmp_path):
    """Create a temporary JobStore for testing."""
    db_path = str(tmp_path / "test_jobs.db")
    s = JobStore(db_path)
    await s.init()
    yield s
    await s.close()


def _make_job(**kwargs) -> JobData:
    """Create a job with sensible defaults."""
    defaults = {
        "agent_name": "echo",
        "status": "pending",
        "input_data": {"message": "hello"},
        "priority": 0,
    }
    defaults.update(kwargs)
    return JobData(**defaults)


# ============================================================================
# TestJobStoreCRUD
# ============================================================================

class TestJobStoreCRUD:
    """Basic save, get, list, count operations."""

    async def test_save_and_get(self, store):
        """Save a job and retrieve it by ID."""
        job = _make_job()
        await store.save(job)

        loaded = await store.get(job.id)
        assert loaded is not None
        assert loaded.id == job.id
        assert loaded.agent_name == "echo"
        assert loaded.input_data == {"message": "hello"}
        assert loaded.status == "pending"

    async def test_get_nonexistent(self, store):
        """Getting a nonexistent job returns None."""
        result = await store.get("nonexistent-id")
        assert result is None

    async def test_upsert_updates_fields(self, store):
        """Saving the same job ID again updates fields."""
        job = _make_job()
        await store.save(job)

        job.status = "completed"
        job.output_data = {"result": "done"}
        job.completed_at = time.monotonic()
        await store.save(job)

        loaded = await store.get(job.id)
        assert loaded.status == "completed"
        assert loaded.output_data == {"result": "done"}

    async def test_list_all(self, store):
        """List returns all jobs."""
        for i in range(5):
            await store.save(_make_job(agent_name=f"agent-{i}"))

        jobs = await store.list()
        assert len(jobs) == 5

    async def test_list_by_status(self, store):
        """List filters by status."""
        await store.save(_make_job(status="pending"))
        await store.save(_make_job(status="completed"))
        await store.save(_make_job(status="failed"))

        pending = await store.list(status="pending")
        assert len(pending) == 1
        assert pending[0].status == "pending"

    async def test_list_by_agent_name(self, store):
        """List filters by agent name."""
        await store.save(_make_job(agent_name="echo"))
        await store.save(_make_job(agent_name="summarizer"))

        echo_jobs = await store.list(agent_name="echo")
        assert len(echo_jobs) == 1
        assert echo_jobs[0].agent_name == "echo"

    async def test_list_pagination(self, store):
        """List supports limit and offset."""
        for i in range(10):
            job = _make_job()
            job.created_at = float(i)  # Deterministic ordering
            await store.save(job)

        page1 = await store.list(limit=3, offset=0)
        page2 = await store.list(limit=3, offset=3)
        assert len(page1) == 3
        assert len(page2) == 3
        # No overlap
        ids1 = {j.id for j in page1}
        ids2 = {j.id for j in page2}
        assert ids1.isdisjoint(ids2)

    async def test_count_all(self, store):
        """Count returns total jobs."""
        await store.save(_make_job())
        await store.save(_make_job())
        assert await store.count() == 2

    async def test_count_by_status(self, store):
        """Count filters by status."""
        await store.save(_make_job(status="pending"))
        await store.save(_make_job(status="completed"))
        await store.save(_make_job(status="completed"))

        assert await store.count(status="pending") == 1
        assert await store.count(status="completed") == 2

    async def test_json_roundtrip(self, store):
        """Complex input/output data survives JSON roundtrip."""
        job = _make_job(
            input_data={"nested": {"key": [1, 2, 3]}, "flag": True},
        )
        job.output_data = {"items": [{"name": "a"}, {"name": "b"}]}
        await store.save(job)

        loaded = await store.get(job.id)
        assert loaded.input_data == {"nested": {"key": [1, 2, 3]}, "flag": True}
        assert loaded.output_data == {"items": [{"name": "a"}, {"name": "b"}]}

    async def test_null_output(self, store):
        """Job with None output roundtrips correctly."""
        job = _make_job()
        assert job.output_data is None
        await store.save(job)

        loaded = await store.get(job.id)
        assert loaded.output_data is None


# ============================================================================
# TestStoreIntegration
# ============================================================================

class TestStoreIntegration:
    """Integration with JobQueue and ExecutionPool."""

    async def test_queue_persists_on_submit(self, store):
        """Queue with store persists jobs on submit."""
        from atlas.pool.queue import JobQueue
        queue = JobQueue(store=store)

        job = _make_job()
        await queue.submit(job)

        # Job should be in store
        loaded = await store.get(job.id)
        assert loaded is not None
        assert loaded.status == "pending"

    async def test_queue_load_pending(self, store):
        """Queue can reload pending jobs from store on restart."""
        from atlas.pool.queue import JobQueue

        # Submit 3 jobs, complete 1
        q1 = JobQueue(store=store)
        j1 = _make_job(agent_name="echo", priority=1)
        j2 = _make_job(agent_name="echo", priority=2)
        j3 = _make_job(agent_name="echo", priority=0)
        await q1.submit(j1)
        await q1.submit(j2)
        await q1.submit(j3)

        # Mark j1 as completed in store
        j1.status = "completed"
        await store.save(j1)

        # New queue should reload only pending jobs
        q2 = JobQueue(store=store)
        count = await q2.load_pending()
        assert count == 2
        assert q2.pending_count == 2

    async def test_backward_compat_no_store(self):
        """Queue/Pool work without store (backward compat)."""
        from atlas.pool.queue import JobQueue
        queue = JobQueue()  # No store

        job = _make_job()
        await queue.submit(job)
        assert queue.pending_count == 1

        # load_pending returns 0 without store
        count = await queue.load_pending()
        assert count == 0

    async def test_pool_persists_completed_job(self, store):
        """Pool with store persists job after completion via event bus."""
        from atlas.contract.registry import AgentRegistry
        from atlas.events import EventBus
        from atlas.pool.queue import JobQueue
        from atlas.pool.executor import ExecutionPool

        agents_path = str(
            __import__("pathlib").Path(__file__).parent.parent / "agents"
        )
        registry = AgentRegistry(search_paths=[agents_path])
        registry.discover()

        bus = EventBus()

        async def persist_on_event(job, old_status, new_status):
            await store.save(job)

        bus.subscribe(persist_on_event)
        queue = JobQueue(store=store, event_bus=bus)
        pool = ExecutionPool(registry, queue)

        await pool.start()
        try:
            job = JobData(agent_name="echo", input_data={"message": "persist me"})
            await pool.submit(job)
            result = await queue.wait_for_terminal(job.id, timeout=5.0)
            assert result is not None
            assert result.status == "completed"

            # Check store
            stored = await store.get(job.id)
            assert stored is not None
            assert stored.status == "completed"
            assert stored.output_data is not None
        finally:
            await pool.stop()

    async def test_pool_persists_failed_job(self, store):
        """Pool with store persists job on failure via event bus."""
        from atlas.contract.registry import AgentRegistry
        from atlas.events import EventBus
        from atlas.pool.queue import JobQueue
        from atlas.pool.executor import ExecutionPool

        agents_path = str(
            __import__("pathlib").Path(__file__).parent.parent / "agents"
        )
        registry = AgentRegistry(search_paths=[agents_path])
        registry.discover()

        bus = EventBus()

        async def persist_on_event(job, old_status, new_status):
            await store.save(job)

        bus.subscribe(persist_on_event)
        queue = JobQueue(store=store, event_bus=bus)
        pool = ExecutionPool(registry, queue)

        await pool.start()
        try:
            # Invalid input should fail validation
            job = JobData(agent_name="echo", input_data={})
            await pool.submit(job)
            result = await queue.wait_for_terminal(job.id, timeout=5.0)
            assert result is not None
            assert result.status == "failed"

            stored = await store.get(job.id)
            assert stored is not None
            assert stored.status == "failed"
            assert stored.error
        finally:
            await pool.stop()

    async def test_store_survives_reconnect(self, tmp_path):
        """Data survives store close and reopen."""
        db_path = str(tmp_path / "survive.db")

        s1 = JobStore(db_path)
        await s1.init()
        job = _make_job()
        await s1.save(job)
        await s1.close()

        s2 = JobStore(db_path)
        await s2.init()
        loaded = await s2.get(job.id)
        assert loaded is not None
        assert loaded.agent_name == "echo"
        await s2.close()

    async def test_not_initialized_raises(self):
        """Operations before init() raise RuntimeError."""
        store = JobStore(":memory:")
        with pytest.raises(RuntimeError, match="not initialized"):
            await store.save(_make_job())
        with pytest.raises(RuntimeError, match="not initialized"):
            await store.get("id")
        with pytest.raises(RuntimeError, match="not initialized"):
            await store.list()
        with pytest.raises(RuntimeError, match="not initialized"):
            await store.count()
