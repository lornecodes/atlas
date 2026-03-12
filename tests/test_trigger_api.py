"""Tests for trigger HTTP API routes."""

from __future__ import annotations

import hashlib
import hmac
import json

import pytest
from aiohttp.test_utils import TestClient, TestServer
from pathlib import Path

from atlas.contract.registry import AgentRegistry
from atlas.events import EventBus
from atlas.pool.executor import ExecutionPool
from atlas.pool.queue import JobQueue
from atlas.serve import create_app
from atlas.store.trigger_store import TriggerStore
from atlas.triggers.scheduler import TriggerScheduler

AGENTS_DIR = Path(__file__).parent.parent / "agents"


@pytest.fixture
async def client(tmp_path):
    registry = AgentRegistry(search_paths=[AGENTS_DIR])
    registry.discover()

    bus = EventBus()
    queue = JobQueue(event_bus=bus)
    pool = ExecutionPool(registry, queue, max_concurrent=2, warm_pool_size=0)

    trigger_store = TriggerStore(str(tmp_path / "test.db"))
    await trigger_store.init()

    scheduler = TriggerScheduler(
        store=trigger_store, pool=pool, poll_interval=999, event_bus=bus,
    )

    app = create_app(
        registry, queue, pool, event_bus=bus,
        trigger_store=trigger_store, trigger_scheduler=scheduler,
    )
    await pool.start()

    server = TestServer(app)
    test_client = TestClient(server)
    await test_client.start_server()

    yield test_client

    await test_client.close()
    await pool.stop()
    await trigger_store.close()


# --- CRUD ---

class TestTriggerCreate:
    async def test_create_cron(self, client):
        resp = await client.post("/api/triggers", json={
            "name": "test-cron",
            "trigger_type": "cron",
            "cron_expr": "*/5 * * * *",
            "agent_name": "echo",
            "input_data": {"message": "cron"},
        })
        assert resp.status == 201
        data = await resp.json()
        assert data["id"].startswith("trigger-")
        assert data["name"] == "test-cron"
        assert data["next_fire"] > 0

    async def test_create_interval(self, client):
        resp = await client.post("/api/triggers", json={
            "trigger_type": "interval",
            "interval_seconds": 300,
            "agent_name": "echo",
        })
        assert resp.status == 201
        data = await resp.json()
        assert data["next_fire"] > 0

    async def test_create_webhook(self, client):
        resp = await client.post("/api/triggers", json={
            "trigger_type": "webhook",
            "agent_name": "echo",
            "webhook_secret": "mysecret",
        })
        assert resp.status == 201
        data = await resp.json()
        assert data["trigger_type"] == "webhook"

    async def test_create_invalid_type(self, client):
        resp = await client.post("/api/triggers", json={
            "trigger_type": "bogus",
            "agent_name": "echo",
        })
        assert resp.status == 400

    async def test_create_missing_target(self, client):
        resp = await client.post("/api/triggers", json={
            "trigger_type": "cron",
            "cron_expr": "* * * * *",
        })
        assert resp.status == 400

    async def test_create_invalid_json(self, client):
        resp = await client.post("/api/triggers", data=b"not json",
                                  headers={"Content-Type": "application/json"})
        assert resp.status == 400


class TestTriggerList:
    async def test_list_empty(self, client):
        resp = await client.get("/api/triggers")
        assert resp.status == 200
        assert await resp.json() == []

    async def test_list_with_triggers(self, client):
        await client.post("/api/triggers", json={
            "trigger_type": "webhook", "agent_name": "echo",
        })
        await client.post("/api/triggers", json={
            "trigger_type": "cron", "cron_expr": "* * * * *", "agent_name": "echo",
        })
        resp = await client.get("/api/triggers")
        data = await resp.json()
        assert len(data) == 2

    async def test_list_filter_type(self, client):
        await client.post("/api/triggers", json={
            "trigger_type": "webhook", "agent_name": "echo",
        })
        await client.post("/api/triggers", json={
            "trigger_type": "cron", "cron_expr": "* * * * *", "agent_name": "echo",
        })
        resp = await client.get("/api/triggers?type=webhook")
        data = await resp.json()
        assert len(data) == 1
        assert data[0]["trigger_type"] == "webhook"


class TestTriggerGetUpdateDelete:
    async def test_get(self, client):
        create_resp = await client.post("/api/triggers", json={
            "trigger_type": "webhook", "agent_name": "echo", "name": "test",
        })
        tid = (await create_resp.json())["id"]

        resp = await client.get(f"/api/triggers/{tid}")
        assert resp.status == 200
        assert (await resp.json())["name"] == "test"

    async def test_get_not_found(self, client):
        resp = await client.get("/api/triggers/nonexistent")
        assert resp.status == 404

    async def test_update(self, client):
        create_resp = await client.post("/api/triggers", json={
            "trigger_type": "webhook", "agent_name": "echo", "name": "original",
        })
        tid = (await create_resp.json())["id"]

        resp = await client.put(f"/api/triggers/{tid}", json={"name": "updated"})
        assert resp.status == 200
        assert (await resp.json())["name"] == "updated"

    async def test_update_not_found(self, client):
        resp = await client.put("/api/triggers/nonexistent", json={"name": "x"})
        assert resp.status == 404

    async def test_delete(self, client):
        create_resp = await client.post("/api/triggers", json={
            "trigger_type": "webhook", "agent_name": "echo",
        })
        tid = (await create_resp.json())["id"]

        resp = await client.delete(f"/api/triggers/{tid}")
        assert resp.status == 200
        assert (await resp.json())["deleted"] is True

        resp = await client.get(f"/api/triggers/{tid}")
        assert resp.status == 404

    async def test_delete_not_found(self, client):
        resp = await client.delete("/api/triggers/nonexistent")
        assert resp.status == 404


# --- Manual Fire ---

class TestManualFire:
    async def test_fire(self, client):
        create_resp = await client.post("/api/triggers", json={
            "trigger_type": "webhook", "agent_name": "echo",
            "input_data": {"message": "manual"},
        })
        tid = (await create_resp.json())["id"]

        resp = await client.post(f"/api/triggers/{tid}/fire")
        assert resp.status == 201
        data = await resp.json()
        assert data["job_id"].startswith("job-")

    async def test_fire_not_found(self, client):
        resp = await client.post("/api/triggers/nonexistent/fire")
        assert resp.status == 404


# --- Webhook ---

class TestWebhook:
    async def test_webhook_no_secret(self, client):
        create_resp = await client.post("/api/triggers", json={
            "trigger_type": "webhook", "agent_name": "echo",
            "input_data": {"default": "val"},
        })
        tid = (await create_resp.json())["id"]

        resp = await client.post(f"/api/hooks/{tid}", json={"extra": "data"})
        assert resp.status == 201
        data = await resp.json()
        assert data["job_id"].startswith("job-")

    async def test_webhook_with_valid_hmac(self, client):
        secret = "test-secret"
        create_resp = await client.post("/api/triggers", json={
            "trigger_type": "webhook", "agent_name": "echo",
            "webhook_secret": secret,
        })
        tid = (await create_resp.json())["id"]

        payload = json.dumps({"message": "signed"}).encode()
        sig = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()

        resp = await client.post(
            f"/api/hooks/{tid}",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "X-Atlas-Signature": f"sha256={sig}",
            },
        )
        assert resp.status == 201

    async def test_webhook_with_bad_hmac(self, client):
        create_resp = await client.post("/api/triggers", json={
            "trigger_type": "webhook", "agent_name": "echo",
            "webhook_secret": "real-secret",
        })
        tid = (await create_resp.json())["id"]

        resp = await client.post(
            f"/api/hooks/{tid}",
            json={"message": "unsigned"},
            headers={"X-Atlas-Signature": "sha256=wrong"},
        )
        assert resp.status == 403

    async def test_webhook_not_found(self, client):
        resp = await client.post("/api/hooks/nonexistent", json={})
        assert resp.status == 404
