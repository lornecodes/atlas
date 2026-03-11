"""Tests for the EventBus — lifecycle events, error isolation, integration."""

from __future__ import annotations

import asyncio

import pytest

from atlas.events import EventBus
from atlas.pool.job import JobData
from atlas.pool.queue import JobQueue


# === EventBus Unit Tests ===


class TestEventBusBasics:
    def test_subscribe(self):
        bus = EventBus()
        async def cb(job, old, new): pass
        bus.subscribe(cb)
        assert bus.subscriber_count == 1

    def test_subscribe_idempotent(self):
        bus = EventBus()
        async def cb(job, old, new): pass
        bus.subscribe(cb)
        bus.subscribe(cb)
        assert bus.subscriber_count == 1

    def test_unsubscribe(self):
        bus = EventBus()
        async def cb(job, old, new): pass
        bus.subscribe(cb)
        bus.unsubscribe(cb)
        assert bus.subscriber_count == 0

    def test_unsubscribe_noop(self):
        bus = EventBus()
        async def cb(job, old, new): pass
        bus.unsubscribe(cb)  # Should not raise
        assert bus.subscriber_count == 0

    async def test_emit_calls_subscriber(self):
        bus = EventBus()
        events = []

        async def cb(job, old, new):
            events.append((job.id, old, new))

        bus.subscribe(cb)
        job = JobData(agent_name="echo", input_data={"msg": "hi"})
        await bus.emit(job, "pending", "running")

        assert len(events) == 1
        assert events[0] == (job.id, "pending", "running")

    async def test_emit_multiple_subscribers(self):
        bus = EventBus()
        results = []

        async def cb1(job, old, new):
            results.append("cb1")

        async def cb2(job, old, new):
            results.append("cb2")

        bus.subscribe(cb1)
        bus.subscribe(cb2)
        job = JobData(agent_name="echo")
        await bus.emit(job, "pending", "running")

        assert results == ["cb1", "cb2"]

    async def test_emit_ordering_preserved(self):
        """Subscribers fire in subscription order."""
        bus = EventBus()
        order = []

        for i in range(5):
            async def cb(job, old, new, idx=i):
                order.append(idx)
            bus.subscribe(cb)

        job = JobData(agent_name="echo")
        await bus.emit(job, "a", "b")
        assert order == [0, 1, 2, 3, 4]


class TestEventBusErrorIsolation:
    async def test_bad_subscriber_doesnt_crash_bus(self):
        bus = EventBus()
        results = []

        async def bad_cb(job, old, new):
            raise ValueError("boom")

        async def good_cb(job, old, new):
            results.append("ok")

        bus.subscribe(bad_cb)
        bus.subscribe(good_cb)

        job = JobData(agent_name="echo")
        await bus.emit(job, "pending", "failed")

        # good_cb still ran despite bad_cb raising
        assert results == ["ok"]

    async def test_multiple_failures_all_isolated(self):
        bus = EventBus()
        results = []

        async def fail1(job, old, new):
            raise RuntimeError("fail1")

        async def success(job, old, new):
            results.append("success")

        async def fail2(job, old, new):
            raise RuntimeError("fail2")

        bus.subscribe(fail1)
        bus.subscribe(success)
        bus.subscribe(fail2)

        job = JobData(agent_name="echo")
        await bus.emit(job, "a", "b")
        assert results == ["success"]

    async def test_unsubscribe_during_emit(self):
        """Unsubscribing during emit doesn't crash (uses list copy)."""
        bus = EventBus()
        results = []

        async def self_removing_cb(job, old, new):
            bus.unsubscribe(self_removing_cb)
            results.append("removed")

        async def stable_cb(job, old, new):
            results.append("stable")

        bus.subscribe(self_removing_cb)
        bus.subscribe(stable_cb)

        job = JobData(agent_name="echo")
        await bus.emit(job, "a", "b")
        assert "removed" in results
        assert "stable" in results
        assert bus.subscriber_count == 1


# === Queue Integration Tests ===


class TestQueueEvents:
    async def test_update_emits_on_status_change(self):
        events = []
        bus = EventBus()

        async def cb(job, old, new):
            events.append((old, new))

        bus.subscribe(cb)
        queue = JobQueue(event_bus=bus)

        job = JobData(agent_name="echo", input_data={"msg": "hi"})
        await queue.submit(job)

        await queue.update(job.id, status="running")
        assert events == [("pending", "running")]

        await queue.update(job.id, status="completed")
        assert events == [("pending", "running"), ("running", "completed")]

    async def test_update_no_emit_when_status_unchanged(self):
        events = []
        bus = EventBus()

        async def cb(job, old, new):
            events.append((old, new))

        bus.subscribe(cb)
        queue = JobQueue(event_bus=bus)

        job = JobData(agent_name="echo", input_data={"msg": "hi"})
        await queue.submit(job)

        # Update a non-status field
        await queue.update(job.id, error="some note")
        assert events == []

    async def test_cancel_emits_event(self):
        events = []
        bus = EventBus()

        async def cb(job, old, new):
            events.append((old, new))

        bus.subscribe(cb)
        queue = JobQueue(event_bus=bus)

        job = JobData(agent_name="echo", input_data={"msg": "hi"})
        await queue.submit(job)
        await queue.cancel(job.id)

        assert events == [("pending", "cancelled")]

    async def test_no_bus_no_crash(self):
        """Queue works fine without an event bus."""
        queue = JobQueue()
        job = JobData(agent_name="echo", input_data={"msg": "hi"})
        await queue.submit(job)
        await queue.update(job.id, status="running")
        await queue.update(job.id, status="completed")
        assert job.status == "completed"

    async def test_all_transitions_emit(self):
        """Every status transition path emits correctly."""
        events = []
        bus = EventBus()

        async def cb(job, old, new):
            events.append(new)

        bus.subscribe(cb)
        queue = JobQueue(event_bus=bus)

        # Path: pending → running → completed
        j1 = JobData(agent_name="echo", input_data={"msg": "1"})
        await queue.submit(j1)
        await queue.update(j1.id, status="running")
        await queue.update(j1.id, status="completed")

        # Path: pending → running → failed
        j2 = JobData(agent_name="echo", input_data={"msg": "2"})
        await queue.submit(j2)
        await queue.update(j2.id, status="running")
        await queue.update(j2.id, status="failed")

        # Path: pending → cancelled
        j3 = JobData(agent_name="echo", input_data={"msg": "3"})
        await queue.submit(j3)
        await queue.cancel(j3.id)

        assert events == [
            "running", "completed",  # j1
            "running", "failed",     # j2
            "cancelled",             # j3
        ]

    async def test_event_receives_correct_job_data(self):
        """Event callback gets the actual job with updated fields."""
        captured_jobs = []
        bus = EventBus()

        async def cb(job, old, new):
            captured_jobs.append((job.id, job.status, job.error))

        bus.subscribe(cb)
        queue = JobQueue(event_bus=bus)

        job = JobData(agent_name="echo", input_data={"msg": "hi"})
        await queue.submit(job)
        await queue.update(job.id, status="failed", error="something broke")

        assert len(captured_jobs) == 1
        assert captured_jobs[0] == (job.id, "failed", "something broke")

    async def test_persistence_as_subscriber(self):
        """Store integration: persistence works as an event subscriber."""
        from atlas.store.job_store import JobStore
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "test.db")
            store = JobStore(db_path)
            await store.init()

            bus = EventBus()

            async def persist(job, old, new):
                await store.save(job)

            bus.subscribe(persist)
            queue = JobQueue(event_bus=bus)

            job = JobData(agent_name="echo", input_data={"msg": "hi"})
            await queue.submit(job)
            await queue.update(job.id, status="running")
            await queue.update(job.id, status="completed", output_data={"result": "ok"})

            stored = await store.get(job.id)
            assert stored is not None
            assert stored.status == "completed"
            assert stored.output_data == {"result": "ok"}

            await store.close()
