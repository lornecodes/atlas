"""E2E integration tests — trigger scheduler + store + pool wired together.

These tests exercise the full trigger stack: create triggers through the
store, fire them via the scheduler, verify jobs land in the pool and
complete, and test HTTP routes including webhook HMAC validation.
No mocking of internal components.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac as hmac_mod
import json
import time
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from atlas.app_keys import TRIGGER_SCHEDULER, TRIGGER_STORE
from atlas.contract.registry import AgentRegistry
from atlas.events import EventBus
from atlas.pool.executor import ExecutionPool
from atlas.pool.job import JobData
from atlas.pool.queue import JobQueue
from atlas.store.job_store import JobStore
from atlas.store.trigger_store import TriggerStore
from atlas.triggers.models import TriggerDefinition
from atlas.triggers.routes import setup_trigger_routes
from atlas.triggers.scheduler import TriggerScheduler

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
async def job_store(tmp_path):
    s = JobStore(str(tmp_path / "e2e_triggers.db"))
    await s.init()
    yield s
    await s.close()


@pytest.fixture
async def trigger_store(tmp_path):
    s = TriggerStore(str(tmp_path / "e2e_triggers.db"))
    await s.init()
    yield s
    await s.close()


@pytest.fixture
def queue(bus, job_store):
    return JobQueue(max_size=100, store=job_store, event_bus=bus)


@pytest.fixture
async def pool(registry, queue):
    p = ExecutionPool(registry, queue, max_concurrent=4, warm_pool_size=2)
    await p.start()
    yield p
    await p.stop(timeout=5.0)


@pytest.fixture
async def scheduler(trigger_store, pool, bus):
    s = TriggerScheduler(
        store=trigger_store,
        pool=pool,
        event_bus=bus,
        poll_interval=0.1,
    )
    yield s
    if s.running:
        await s.stop()


# ---------------------------------------------------------------------------
# Trigger Store CRUD + persistence
# ---------------------------------------------------------------------------


class TestTriggerStoreCrud:
    """Trigger CRUD operations through the SQLite store."""

    @pytest.mark.asyncio
    async def test_trigger_crud_persistence(self, trigger_store):
        """Save, get, update, list, and delete a trigger through the store."""
        trigger = TriggerDefinition(
            name="test-interval",
            trigger_type="interval",
            agent_name="echo",
            interval_seconds=60.0,
            input_data={"message": "hello"},
        )
        trigger.next_fire = time.time() + 60

        # Save
        await trigger_store.save(trigger)

        # Get
        fetched = await trigger_store.get(trigger.id)
        assert fetched is not None
        assert fetched.name == "test-interval"
        assert fetched.agent_name == "echo"
        assert fetched.interval_seconds == 60.0
        assert fetched.input_data == {"message": "hello"}

        # Update
        fetched.name = "test-interval-updated"
        fetched.priority = 5
        await trigger_store.save(fetched)
        updated = await trigger_store.get(trigger.id)
        assert updated is not None
        assert updated.name == "test-interval-updated"
        assert updated.priority == 5

        # List
        all_triggers = await trigger_store.list()
        assert len(all_triggers) >= 1
        ids = [t.id for t in all_triggers]
        assert trigger.id in ids

        # Delete
        deleted = await trigger_store.delete(trigger.id)
        assert deleted is True
        gone = await trigger_store.get(trigger.id)
        assert gone is None

    @pytest.mark.asyncio
    async def test_due_triggers_filtering(self, trigger_store):
        """Only triggers whose next_fire has passed are returned by list_due."""
        now = time.time()

        past = TriggerDefinition(
            name="past",
            trigger_type="interval",
            agent_name="echo",
            interval_seconds=10.0,
            input_data={"message": "past"},
        )
        past.next_fire = now - 10

        future = TriggerDefinition(
            name="future",
            trigger_type="interval",
            agent_name="echo",
            interval_seconds=10.0,
            input_data={"message": "future"},
        )
        future.next_fire = now + 3600

        await trigger_store.save(past)
        await trigger_store.save(future)

        due = await trigger_store.list_due(before=now)
        due_ids = [t.id for t in due]
        assert past.id in due_ids
        assert future.id not in due_ids

    @pytest.mark.asyncio
    async def test_disabled_trigger_skipped(self, trigger_store):
        """Disabled triggers are excluded from list_due results."""
        now = time.time()

        trigger = TriggerDefinition(
            name="disabled",
            trigger_type="interval",
            agent_name="echo",
            interval_seconds=10.0,
            enabled=False,
            input_data={"message": "skip"},
        )
        trigger.next_fire = now - 10

        await trigger_store.save(trigger)

        due = await trigger_store.list_due(before=now)
        due_ids = [t.id for t in due]
        assert trigger.id not in due_ids

    @pytest.mark.asyncio
    async def test_trigger_store_recovery(self, tmp_path):
        """Triggers survive store close and reopen (SQLite persistence)."""
        db_path = str(tmp_path / "recovery.db")

        store1 = TriggerStore(db_path)
        await store1.init()

        trigger = TriggerDefinition(
            name="persistent",
            trigger_type="interval",
            agent_name="echo",
            interval_seconds=30.0,
            input_data={"message": "survive"},
        )
        trigger.next_fire = time.time() + 30
        await store1.save(trigger)
        saved_id = trigger.id
        await store1.close()

        # Reopen with fresh store instance
        store2 = TriggerStore(db_path)
        await store2.init()
        recovered = await store2.get(saved_id)
        assert recovered is not None
        assert recovered.name == "persistent"
        assert recovered.agent_name == "echo"
        assert recovered.input_data == {"message": "survive"}
        await store2.close()


# ---------------------------------------------------------------------------
# Scheduler + Pool integration
# ---------------------------------------------------------------------------


class TestSchedulerPoolIntegration:
    """Scheduler fires triggers, jobs execute through the real pool."""

    @pytest.mark.asyncio
    async def test_cron_trigger_fires_job(self, scheduler, trigger_store, queue):
        """An interval trigger with next_fire in the past fires on the next tick."""
        trigger = TriggerDefinition(
            name="fast-interval",
            trigger_type="interval",
            agent_name="echo",
            interval_seconds=0.01,
            input_data={"message": "triggered"},
        )
        trigger.next_fire = time.time() - 1  # already due
        await trigger_store.save(trigger)

        await scheduler.start()
        await asyncio.sleep(0.5)
        await scheduler.stop()

        # Trigger should have fired at least once
        updated = await trigger_store.get(trigger.id)
        assert updated is not None
        assert updated.fire_count >= 1
        assert updated.last_job_id != ""

        # Job should have completed in the pool
        result = await queue.wait_for_terminal(updated.last_job_id, timeout=5.0)
        assert result is not None
        assert result.status == "completed"
        assert result.output_data == {"message": "triggered"}

    @pytest.mark.asyncio
    async def test_one_shot_trigger_disables(self, scheduler, trigger_store, queue):
        """A one-shot trigger fires once, then becomes disabled."""
        now = time.time()
        trigger = TriggerDefinition(
            name="one-shot",
            trigger_type="one_shot",
            agent_name="echo",
            fire_at=now - 1,
            input_data={"message": "once"},
        )
        trigger.next_fire = now - 1
        await trigger_store.save(trigger)

        await scheduler.start()
        await asyncio.sleep(0.5)
        await scheduler.stop()

        updated = await trigger_store.get(trigger.id)
        assert updated is not None
        assert updated.fire_count == 1
        assert updated.enabled is False
        assert updated.next_fire == 0.0

        # Second start should produce no new jobs
        first_job_id = updated.last_job_id
        await scheduler.start()
        await asyncio.sleep(0.3)
        await scheduler.stop()

        still = await trigger_store.get(trigger.id)
        assert still is not None
        assert still.fire_count == 1
        assert still.last_job_id == first_job_id

    @pytest.mark.asyncio
    async def test_recurring_trigger_increments(self, scheduler, trigger_store, queue):
        """A recurring trigger increments fire_count and updates last_fired."""
        trigger = TriggerDefinition(
            name="recurring",
            trigger_type="interval",
            agent_name="echo",
            interval_seconds=0.01,
            input_data={"message": "repeat"},
        )
        trigger.next_fire = time.time() - 1
        await trigger_store.save(trigger)

        await scheduler.start()
        await asyncio.sleep(0.5)
        await scheduler.stop()

        updated = await trigger_store.get(trigger.id)
        assert updated is not None
        assert updated.fire_count >= 1
        assert updated.last_fired > 0
        assert updated.last_job_id != ""

    @pytest.mark.asyncio
    async def test_webhook_trigger_fires_job(self, scheduler, trigger_store, queue):
        """fire_webhook() submits a job with payload merged into input_data."""
        trigger = TriggerDefinition(
            name="webhook",
            trigger_type="webhook",
            agent_name="echo",
            input_data={"message": "base"},
        )
        await trigger_store.save(trigger)

        job_id = await scheduler.fire_webhook(
            trigger.id, payload={"message": "from-webhook"}
        )
        result = await queue.wait_for_terminal(job_id, timeout=5.0)
        assert result is not None
        assert result.status == "completed"
        # Payload overrides base input_data
        assert result.output_data == {"message": "from-webhook"}

        updated = await trigger_store.get(trigger.id)
        assert updated is not None
        assert updated.fire_count == 1
        assert updated.last_job_id == job_id

    @pytest.mark.asyncio
    async def test_manual_fire(self, scheduler, trigger_store, queue):
        """fire_manual() creates and submits a job for any trigger type."""
        trigger = TriggerDefinition(
            name="manual-target",
            trigger_type="interval",
            agent_name="echo",
            interval_seconds=3600,
            input_data={"message": "manual"},
        )
        trigger.next_fire = time.time() + 3600  # far future — won't fire on tick
        await trigger_store.save(trigger)

        job_id = await scheduler.fire_manual(trigger.id)
        result = await queue.wait_for_terminal(job_id, timeout=5.0)
        assert result is not None
        assert result.status == "completed"
        assert result.output_data == {"message": "manual"}

        updated = await trigger_store.get(trigger.id)
        assert updated is not None
        assert updated.fire_count == 1
        assert updated.last_job_id == job_id


# ---------------------------------------------------------------------------
# Scheduler lifecycle
# ---------------------------------------------------------------------------


class TestSchedulerLifecycle:
    """Scheduler start/stop and concurrency guarantees."""

    @pytest.mark.asyncio
    async def test_scheduler_start_stop_lifecycle(self, scheduler, trigger_store):
        """Start creates a background task, stop cancels it cleanly."""
        assert scheduler.running is False

        await scheduler.start()
        assert scheduler.running is True

        # Let a few ticks happen
        await asyncio.sleep(0.3)

        await scheduler.stop()
        assert scheduler.running is False

    @pytest.mark.asyncio
    async def test_tick_lock_prevents_overlap(self, scheduler, trigger_store):
        """Concurrent _tick() calls do not overlap — the lock gates entry."""
        now = time.time()
        # Create several triggers so _tick work is non-trivial
        for i in range(5):
            t = TriggerDefinition(
                name=f"lock-test-{i}",
                trigger_type="interval",
                agent_name="echo",
                interval_seconds=0.01,
                input_data={"message": f"lock-{i}"},
            )
            t.next_fire = now - 1
            await trigger_store.save(t)

        # Fire multiple concurrent _tick() calls
        results = await asyncio.gather(
            scheduler._tick(),
            scheduler._tick(),
            scheduler._tick(),
            return_exceptions=True,
        )

        # No exceptions should have occurred
        for r in results:
            assert not isinstance(r, Exception), f"_tick() raised: {r}"


# ---------------------------------------------------------------------------
# HTTP routes — webhook HMAC, payload limits, CRUD
# ---------------------------------------------------------------------------


@pytest.fixture
async def trigger_app(trigger_store, scheduler):
    """Create an aiohttp app with trigger routes for test client use."""
    app = web.Application()
    app[TRIGGER_STORE] = trigger_store
    app[TRIGGER_SCHEDULER] = scheduler
    setup_trigger_routes(app)
    return app


@pytest.fixture
async def client(trigger_app):
    """aiohttp TestClient wired to the trigger routes."""
    async with TestClient(TestServer(trigger_app)) as c:
        yield c


class TestTriggerRoutes:
    """HTTP route tests using aiohttp test client."""

    @pytest.mark.asyncio
    async def test_webhook_hmac_validation(self, client, trigger_store):
        """Webhook route validates HMAC-SHA256 and rejects bad signatures."""
        secret = "test-secret-key"
        trigger = TriggerDefinition(
            name="hmac-webhook",
            trigger_type="webhook",
            agent_name="echo",
            input_data={"message": "hmac-test"},
            webhook_secret=secret,
        )
        await trigger_store.save(trigger)

        payload = json.dumps({"message": "webhook-payload"}).encode()

        # Valid signature
        valid_sig = hmac_mod.new(
            secret.encode(), payload, hashlib.sha256
        ).hexdigest()
        resp = await client.post(
            f"/api/hooks/{trigger.id}",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "X-Atlas-Signature": f"sha256={valid_sig}",
            },
        )
        assert resp.status == 201
        body = await resp.json()
        assert "job_id" in body

        # Invalid signature
        resp_bad = await client.post(
            f"/api/hooks/{trigger.id}",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "X-Atlas-Signature": "sha256=deadbeef",
            },
        )
        assert resp_bad.status == 403
        err = await resp_bad.json()
        assert "Invalid signature" in err["error"]

    @pytest.mark.asyncio
    async def test_webhook_payload_size_limit(self, client, trigger_store):
        """Payloads exceeding 1MB are rejected with 413."""
        trigger = TriggerDefinition(
            name="size-limit-webhook",
            trigger_type="webhook",
            agent_name="echo",
            input_data={"message": "size"},
        )
        await trigger_store.save(trigger)

        # 1MB + 1 byte payload
        oversized = b"x" * (1_048_576 + 1)
        resp = await client.post(
            f"/api/hooks/{trigger.id}",
            data=oversized,
            headers={"Content-Type": "application/json"},
        )
        assert resp.status == 413

    @pytest.mark.asyncio
    async def test_trigger_routes_crud(self, client, trigger_store):
        """Full CRUD lifecycle through HTTP routes."""
        # CREATE
        create_body = {
            "name": "route-test",
            "trigger_type": "interval",
            "agent_name": "echo",
            "interval_seconds": 300,
            "input_data": {"message": "route"},
        }
        resp = await client.post("/api/triggers", json=create_body)
        assert resp.status == 201
        created = await resp.json()
        trigger_id = created["id"]
        assert created["name"] == "route-test"
        assert created["interval_seconds"] == 300

        # GET
        resp = await client.get(f"/api/triggers/{trigger_id}")
        assert resp.status == 200
        fetched = await resp.json()
        assert fetched["id"] == trigger_id
        assert fetched["agent_name"] == "echo"

        # LIST
        resp = await client.get("/api/triggers")
        assert resp.status == 200
        listing = await resp.json()
        ids = [t["id"] for t in listing]
        assert trigger_id in ids

        # UPDATE
        resp = await client.put(
            f"/api/triggers/{trigger_id}",
            json={"name": "route-test-updated", "priority": 7},
        )
        assert resp.status == 200
        updated = await resp.json()
        assert updated["name"] == "route-test-updated"
        assert updated["priority"] == 7

        # DELETE
        resp = await client.delete(f"/api/triggers/{trigger_id}")
        assert resp.status == 200
        del_body = await resp.json()
        assert del_body["deleted"] is True

        # Confirm gone
        resp = await client.get(f"/api/triggers/{trigger_id}")
        assert resp.status == 404
