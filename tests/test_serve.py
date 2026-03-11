"""Tests for the HTTP serve API."""

from __future__ import annotations

import pytest
from aiohttp.test_utils import AioHTTPTestCase, TestClient, TestServer
from pathlib import Path

from atlas.contract.registry import AgentRegistry
from atlas.events import EventBus
from atlas.pool.executor import ExecutionPool
from atlas.pool.job import JobData
from atlas.pool.queue import JobQueue
from atlas.serve import create_app

AGENTS_DIR = Path(__file__).parent.parent / "agents"


@pytest.fixture
async def client():
    """Create a test client with a running pool and event bus."""
    registry = AgentRegistry(search_paths=[AGENTS_DIR])
    registry.discover()

    bus = EventBus()
    queue = JobQueue(event_bus=bus)
    pool = ExecutionPool(registry, queue, max_concurrent=2, warm_pool_size=0)

    app = create_app(registry, queue, pool, event_bus=bus)
    await pool.start()

    server = TestServer(app)
    test_client = TestClient(server)
    await test_client.start_server()

    yield test_client

    await test_client.close()
    await pool.stop()


class TestHealth:
    async def test_health(self, client):
        resp = await client.get("/api/health")
        assert resp.status == 200
        data = await resp.json()
        assert data["status"] == "ok"
        assert "pending" in data
        assert "running" in data
        assert "capacity_remaining" in data


class TestSubmitJob:
    async def test_submit(self, client):
        resp = await client.post("/api/jobs", json={
            "agent": "echo",
            "input": {"message": "hello from http"},
        })
        assert resp.status == 201
        data = await resp.json()
        assert "id" in data
        assert data["id"].startswith("job-")

    async def test_submit_missing_agent(self, client):
        resp = await client.post("/api/jobs", json={"input": {"x": 1}})
        assert resp.status == 400
        data = await resp.json()
        assert "agent" in data["error"].lower()

    async def test_submit_invalid_json(self, client):
        resp = await client.post(
            "/api/jobs",
            data=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status == 400

    async def test_submit_with_priority(self, client):
        resp = await client.post("/api/jobs", json={
            "agent": "echo",
            "input": {"message": "priority"},
            "priority": 10,
        })
        assert resp.status == 201


class TestGetJob:
    async def test_get_completed_job(self, client):
        """Submit a job, wait for completion, then GET it."""
        import asyncio

        submit_resp = await client.post("/api/jobs", json={
            "agent": "echo",
            "input": {"message": "get me"},
        })
        job_id = (await submit_resp.json())["id"]

        # Wait for pool to execute
        await asyncio.sleep(0.5)

        resp = await client.get(f"/api/jobs/{job_id}")
        assert resp.status == 200
        data = await resp.json()
        assert data["id"] == job_id
        assert data["status"] == "completed"
        assert data["output_data"]["message"] == "get me"

    async def test_get_not_found(self, client):
        resp = await client.get("/api/jobs/job-nonexistent")
        assert resp.status == 404


class TestListJobs:
    async def test_list_empty(self, client):
        resp = await client.get("/api/jobs")
        assert resp.status == 200
        data = await resp.json()
        assert data == []

    async def test_list_after_submit(self, client):
        import asyncio

        await client.post("/api/jobs", json={
            "agent": "echo",
            "input": {"message": "list me"},
        })
        await asyncio.sleep(0.5)

        resp = await client.get("/api/jobs")
        assert resp.status == 200
        data = await resp.json()
        assert len(data) >= 1

    async def test_list_filter_by_status(self, client):
        import asyncio

        await client.post("/api/jobs", json={
            "agent": "echo",
            "input": {"message": "filter me"},
        })
        await asyncio.sleep(0.5)

        resp = await client.get("/api/jobs?status=completed")
        data = await resp.json()
        assert all(j["status"] == "completed" for j in data)

    async def test_list_filter_by_agent(self, client):
        import asyncio

        await client.post("/api/jobs", json={
            "agent": "echo",
            "input": {"message": "agent filter"},
        })
        await asyncio.sleep(0.5)

        resp = await client.get("/api/jobs?agent=echo")
        data = await resp.json()
        assert all(j["agent_name"] == "echo" for j in data)


@pytest.fixture
async def idle_client():
    """Test client with pool NOT started — jobs stay pending."""
    registry = AgentRegistry(search_paths=[AGENTS_DIR])
    registry.discover()

    queue = JobQueue()
    pool = ExecutionPool(registry, queue, max_concurrent=2, warm_pool_size=0)

    app = create_app(registry, queue, pool)

    server = TestServer(app)
    test_client = TestClient(server)
    await test_client.start_server()

    yield test_client

    await test_client.close()


class TestCancelJob:
    async def test_cancel_pending(self, idle_client):
        """Submit to idle pool (not started), job stays pending, then cancel."""
        from atlas.app_keys import QUEUE
        queue: JobQueue = idle_client.app[QUEUE]
        job = JobData(agent_name="echo", input_data={"message": "cancel me"})
        await queue.submit(job)

        resp = await idle_client.delete(f"/api/jobs/{job.id}")
        assert resp.status == 200
        data = await resp.json()
        assert data["cancelled"] is True

    async def test_cancel_nonexistent(self, client):
        resp = await client.delete("/api/jobs/job-nope")
        assert resp.status == 200
        data = await resp.json()
        assert data["cancelled"] is False


# === Trace endpoints ===


class TestTraceEndpoints:
    async def test_list_traces_empty(self, client):
        resp = await client.get("/api/traces")
        assert resp.status == 200
        data = await resp.json()
        assert data == []

    async def test_trace_created_on_completion(self, client):
        import asyncio

        await client.post("/api/jobs", json={
            "agent": "echo",
            "input": {"message": "trace me"},
        })
        await asyncio.sleep(0.5)

        resp = await client.get("/api/traces")
        assert resp.status == 200
        data = await resp.json()
        assert len(data) >= 1
        assert data[0]["agent_name"] == "echo"
        assert data[0]["status"] == "completed"
        assert "trace_id" in data[0]

    async def test_get_trace_by_id(self, client):
        import asyncio

        submit = await client.post("/api/jobs", json={
            "agent": "echo",
            "input": {"message": "get trace"},
        })
        job_id = (await submit.json())["id"]
        await asyncio.sleep(0.5)

        resp = await client.get(f"/api/traces/{job_id}")
        assert resp.status == 200
        data = await resp.json()
        assert data["trace_id"] == job_id
        assert data["agent_name"] == "echo"

    async def test_get_trace_not_found(self, client):
        resp = await client.get("/api/traces/nonexistent")
        assert resp.status == 404

    async def test_list_traces_filter_by_agent(self, client):
        import asyncio

        await client.post("/api/jobs", json={
            "agent": "echo",
            "input": {"message": "filter"},
        })
        await asyncio.sleep(0.5)

        resp = await client.get("/api/traces?agent=echo")
        data = await resp.json()
        assert all(t["agent_name"] == "echo" for t in data)

        resp2 = await client.get("/api/traces?agent=nonexistent")
        data2 = await resp2.json()
        assert data2 == []


# === Orchestrator endpoints ===


class TestOrchestratorEndpoints:
    async def test_get_orchestrator_default(self, client):
        resp = await client.get("/api/orchestrator")
        assert resp.status == 200
        data = await resp.json()
        assert data["name"] == "DefaultOrchestrator"

    async def test_set_orchestrator_not_found(self, client):
        resp = await client.post("/api/orchestrator", json={"name": "nonexistent"})
        assert resp.status == 404

    async def test_reset_orchestrator(self, client):
        resp = await client.post("/api/orchestrator", json={"name": None})
        assert resp.status == 200
        data = await resp.json()
        assert data["orchestrator"] == "DefaultOrchestrator"

    async def test_set_orchestrator_real(self, client):
        resp = await client.post("/api/orchestrator", json={"name": "priority-router"})
        if resp.status == 200:
            data = await resp.json()
            assert data["orchestrator"] == "priority-router"

            # Verify it changed
            get_resp = await client.get("/api/orchestrator")
            get_data = await get_resp.json()
            assert get_data["name"] == "PriorityRouterOrchestrator"

            # Reset
            await client.post("/api/orchestrator", json={"name": None})
        else:
            # priority-router might not be discoverable in test env
            assert resp.status == 404
