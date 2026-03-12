"""Tests for TriggerScheduler — async background loop that fires triggers."""

from __future__ import annotations

import asyncio
import time

import pytest

from atlas.contract.registry import AgentRegistry
from atlas.events import EventBus
from atlas.pool.executor import ExecutionPool
from atlas.pool.job import JobData
from atlas.pool.queue import JobQueue
from atlas.store.trigger_store import TriggerStore
from atlas.triggers.models import TriggerDefinition
from atlas.triggers.scheduler import TriggerScheduler

from conftest import AGENTS_DIR


# --- Fixtures ---

@pytest.fixture
async def trigger_store(tmp_path):
    s = TriggerStore(str(tmp_path / "test.db"))
    await s.init()
    yield s
    await s.close()


@pytest.fixture
def bus():
    return EventBus()


@pytest.fixture
def queue(bus):
    return JobQueue(event_bus=bus)


@pytest.fixture
def registry():
    reg = AgentRegistry(search_paths=[AGENTS_DIR])
    reg.discover()
    return reg


@pytest.fixture
async def pool(registry, queue):
    p = ExecutionPool(registry, queue, max_concurrent=4, warm_pool_size=2)
    await p.start()
    yield p
    await p.stop()


@pytest.fixture
def scheduler(trigger_store, pool, bus):
    return TriggerScheduler(
        store=trigger_store,
        pool=pool,
        poll_interval=999,  # don't auto-tick in tests
        event_bus=bus,
    )


def _make_trigger(**kwargs) -> TriggerDefinition:
    defaults = dict(
        trigger_type="interval",
        interval_seconds=60,
        agent_name="echo",
        input_data={"message": "scheduled"},
    )
    defaults.update(kwargs)
    return TriggerDefinition(**defaults)


# --- Tick Tests ---

class TestSchedulerTick:
    async def test_fires_due_trigger(self, scheduler, trigger_store, queue):
        """A due trigger should produce a job in the queue."""
        t = _make_trigger()
        t.next_fire = time.time() - 10  # already due
        await trigger_store.save(t)

        await scheduler._tick()

        updated = await trigger_store.get(t.id)
        assert updated.fire_count == 1
        assert updated.last_job_id.startswith("job-")
        assert updated.last_fired > 0

    async def test_skips_future_trigger(self, scheduler, trigger_store):
        t = _make_trigger()
        t.next_fire = time.time() + 10000  # far future
        await trigger_store.save(t)

        await scheduler._tick()

        updated = await trigger_store.get(t.id)
        assert updated.fire_count == 0

    async def test_skips_disabled_trigger(self, scheduler, trigger_store):
        t = _make_trigger(enabled=False)
        t.next_fire = time.time() - 10
        await trigger_store.save(t)

        await scheduler._tick()

        updated = await trigger_store.get(t.id)
        assert updated.fire_count == 0

    async def test_one_shot_disables_after_fire(self, scheduler, trigger_store):
        t = _make_trigger(
            trigger_type="one_shot",
            fire_at=time.time() - 10,
        )
        t.next_fire = t.fire_at
        await trigger_store.save(t)

        await scheduler._tick()

        updated = await trigger_store.get(t.id)
        assert updated.fire_count == 1
        assert updated.enabled is False
        assert updated.next_fire == 0.0

    async def test_interval_computes_next_fire(self, scheduler, trigger_store):
        t = _make_trigger(interval_seconds=300)
        t.next_fire = time.time() - 5
        await trigger_store.save(t)

        await scheduler._tick()

        updated = await trigger_store.get(t.id)
        assert updated.fire_count == 1
        assert updated.next_fire > time.time()
        # next_fire should be approximately last_fired + 300
        assert abs(updated.next_fire - (updated.last_fired + 300)) < 2.0

    async def test_multiple_due_triggers(self, scheduler, trigger_store):
        for i in range(3):
            t = _make_trigger(name=f"t{i}")
            t.next_fire = time.time() - (10 + i)
            await trigger_store.save(t)

        await scheduler._tick()

        triggers = await trigger_store.list()
        fired = [t for t in triggers if t.fire_count > 0]
        assert len(fired) == 3

    async def test_job_has_trigger_metadata(self, scheduler, trigger_store, queue):
        t = _make_trigger(name="meta-test")
        t.next_fire = time.time() - 10
        await trigger_store.save(t)

        await scheduler._tick()

        # Wait briefly for the job to be submitted
        await asyncio.sleep(0.1)
        updated = await trigger_store.get(t.id)
        job = queue.get(updated.last_job_id)
        assert job is not None
        assert job.metadata["_trigger_id"] == t.id
        assert job.metadata["_trigger_type"] == "interval"

    async def test_job_input_matches_trigger(self, scheduler, trigger_store, queue):
        t = _make_trigger(input_data={"message": "hello from trigger"})
        t.next_fire = time.time() - 10
        await trigger_store.save(t)

        await scheduler._tick()

        updated = await trigger_store.get(t.id)
        job = queue.get(updated.last_job_id)
        assert job.input_data == {"message": "hello from trigger"}


# --- Webhook Tests ---

class TestWebhookFire:
    async def test_fire_webhook(self, scheduler, trigger_store, queue):
        t = _make_trigger(trigger_type="webhook", input_data={"default": "val"})
        await trigger_store.save(t)

        job_id = await scheduler.fire_webhook(t.id, payload={"extra": "data"})

        assert job_id.startswith("job-")
        job = queue.get(job_id)
        assert job is not None
        assert job.input_data == {"default": "val", "extra": "data"}

        updated = await trigger_store.get(t.id)
        assert updated.fire_count == 1
        assert updated.last_job_id == job_id

    async def test_fire_webhook_payload_overrides(self, scheduler, trigger_store, queue):
        t = _make_trigger(trigger_type="webhook", input_data={"key": "original"})
        await trigger_store.save(t)

        job_id = await scheduler.fire_webhook(t.id, payload={"key": "overridden"})
        job = queue.get(job_id)
        assert job.input_data["key"] == "overridden"

    async def test_fire_webhook_no_payload(self, scheduler, trigger_store, queue):
        t = _make_trigger(trigger_type="webhook", input_data={"message": "default"})
        await trigger_store.save(t)

        job_id = await scheduler.fire_webhook(t.id)
        job = queue.get(job_id)
        assert job.input_data == {"message": "default"}

    async def test_fire_webhook_not_found(self, scheduler):
        with pytest.raises(ValueError, match="not found"):
            await scheduler.fire_webhook("nonexistent")

    async def test_fire_webhook_disabled(self, scheduler, trigger_store):
        t = _make_trigger(trigger_type="webhook", enabled=False)
        await trigger_store.save(t)
        with pytest.raises(ValueError, match="disabled"):
            await scheduler.fire_webhook(t.id)

    async def test_fire_webhook_wrong_type(self, scheduler, trigger_store):
        t = _make_trigger(trigger_type="interval", interval_seconds=60)
        await trigger_store.save(t)
        with pytest.raises(ValueError, match="not webhook"):
            await scheduler.fire_webhook(t.id)


# --- Manual Fire Tests ---

class TestManualFire:
    async def test_fire_manual(self, scheduler, trigger_store, queue):
        t = _make_trigger()
        await trigger_store.save(t)

        job_id = await scheduler.fire_manual(t.id)
        assert job_id.startswith("job-")

        updated = await trigger_store.get(t.id)
        assert updated.fire_count == 1
        assert updated.last_job_id == job_id

    async def test_fire_manual_not_found(self, scheduler):
        with pytest.raises(ValueError, match="not found"):
            await scheduler.fire_manual("nonexistent")

    async def test_fire_manual_updates_next_fire_for_recurring(self, scheduler, trigger_store):
        t = _make_trigger(trigger_type="interval", interval_seconds=300)
        await trigger_store.save(t)

        await scheduler.fire_manual(t.id)

        updated = await trigger_store.get(t.id)
        assert updated.next_fire > time.time()


# --- Start/Stop Tests ---

class TestSchedulerLifecycle:
    async def test_start_stop(self, scheduler):
        assert scheduler.running is False
        await scheduler.start()
        assert scheduler.running is True
        await scheduler.stop()
        assert scheduler.running is False

    async def test_double_start(self, scheduler):
        await scheduler.start()
        await scheduler.start()  # should not raise
        assert scheduler.running is True
        await scheduler.stop()

    async def test_stop_without_start(self, scheduler):
        await scheduler.stop()  # should not raise
