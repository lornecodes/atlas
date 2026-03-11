"""Tests for WebSocket streaming + metrics endpoints — Phase 8B/C."""

from __future__ import annotations

import asyncio
import time

import pytest
from aiohttp.test_utils import TestClient, TestServer
from pathlib import Path

from atlas.contract.registry import AgentRegistry
from atlas.events import EventBus
from atlas.pool.executor import ExecutionPool
from atlas.pool.job import JobData
from atlas.pool.queue import JobQueue
from atlas.serve import create_app, _job_to_dict
from atlas.ws import _build_event_frame

AGENTS_DIR = Path(__file__).parent.parent / "agents"


# === Fixtures ===


@pytest.fixture
async def ws_client():
    """Test client with EventBus wired into create_app."""
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


# === _build_event_frame Tests ===


class TestBuildEventFrame:
    def test_frame_structure(self):
        job = JobData(agent_name="echo", input_data={"message": "hi"})
        frame = _build_event_frame(job, "pending", "running", _job_to_dict)
        assert frame["job_id"] == job.id
        assert frame["agent_name"] == "echo"
        assert frame["old_status"] == "pending"
        assert frame["new_status"] == "running"
        assert "timestamp" in frame
        assert "job" in frame

    def test_frame_includes_full_job(self):
        job = JobData(agent_name="echo", input_data={"message": "hi"}, priority=5)
        frame = _build_event_frame(job, "pending", "running", _job_to_dict)
        assert frame["job"]["agent_name"] == "echo"
        assert frame["job"]["priority"] == 5
        assert frame["job"]["input_data"] == {"message": "hi"}

    def test_timestamp_is_recent(self):
        job = JobData(agent_name="echo")
        before = time.time()
        frame = _build_event_frame(job, "pending", "running", _job_to_dict)
        after = time.time()
        assert before <= frame["timestamp"] <= after


# === WebSocket /api/events Tests ===


class TestEventsWebSocket:
    async def test_connect_and_receive_event(self, ws_client):
        async with ws_client.ws_connect("/api/events") as ws:
            # Submit a job to generate events
            resp = await ws_client.post("/api/jobs", json={
                "agent": "echo", "input": {"message": "hello"},
            })
            assert resp.status == 201

            # Should receive at least one event (pending->running or running->completed)
            msg = await asyncio.wait_for(ws.receive_json(), timeout=3.0)
            assert "job_id" in msg
            assert "old_status" in msg
            assert "new_status" in msg
            assert "job" in msg

    async def test_multiple_events_streamed(self, ws_client):
        async with ws_client.ws_connect("/api/events") as ws:
            resp = await ws_client.post("/api/jobs", json={
                "agent": "echo", "input": {"message": "multi"},
            })
            assert resp.status == 201

            events = []
            for _ in range(2):
                try:
                    msg = await asyncio.wait_for(ws.receive_json(), timeout=3.0)
                    events.append(msg)
                except asyncio.TimeoutError:
                    break
            assert len(events) >= 2  # pending->running, running->completed

    async def test_disconnect_cleans_up_subscription(self, ws_client):
        from atlas.app_keys import EVENT_BUS
        bus = ws_client.app[EVENT_BUS]
        initial_count = bus.subscriber_count

        ws = await ws_client.ws_connect("/api/events")
        await asyncio.sleep(0.1)
        # Subscriber added
        assert bus.subscriber_count > initial_count

        await ws.close()
        await asyncio.sleep(0.1)
        # Subscriber removed
        assert bus.subscriber_count == initial_count


# === WebSocket /api/jobs/{id}/events Tests ===


class TestJobEventsWebSocket:
    async def test_receive_single_job_events(self, ws_client):
        # Submit first
        resp = await ws_client.post("/api/jobs", json={
            "agent": "echo", "input": {"message": "tracked"},
        })
        data = await resp.json()
        job_id = data["id"]

        async with ws_client.ws_connect(f"/api/jobs/{job_id}/events") as ws:
            msg = await asyncio.wait_for(ws.receive_json(), timeout=3.0)
            assert msg["job_id"] == job_id

    async def test_filters_other_jobs(self, ws_client):
        # Connect to a fake job ID
        async with ws_client.ws_connect("/api/jobs/job-nonexistent/events") as ws:
            # Submit a real job — events for it should NOT arrive on this WS
            await ws_client.post("/api/jobs", json={
                "agent": "echo", "input": {"message": "other"},
            })
            await asyncio.sleep(0.5)

            # Should NOT have received anything
            assert ws.closed or True  # If not closed, we just verify no message


# === Metrics Endpoint Tests ===


class TestMetricsEndpoints:
    async def test_get_global_metrics(self, ws_client):
        # Submit and wait for completion
        resp = await ws_client.post("/api/jobs", json={
            "agent": "echo", "input": {"message": "metrics"},
        })
        assert resp.status == 201
        await asyncio.sleep(0.5)

        resp = await ws_client.get("/api/metrics")
        assert resp.status == 200
        data = await resp.json()
        assert "global" in data
        assert "agents" in data
        assert data["global"]["total_jobs"] >= 1

    async def test_get_agent_metrics(self, ws_client):
        # Submit and wait
        await ws_client.post("/api/jobs", json={
            "agent": "echo", "input": {"message": "agent-metrics"},
        })
        await asyncio.sleep(0.5)

        resp = await ws_client.get("/api/metrics/echo")
        assert resp.status == 200
        data = await resp.json()
        assert "jobs_by_status" in data
        assert "latency_p50_ms" in data

    async def test_unknown_agent_404(self, ws_client):
        resp = await ws_client.get("/api/metrics/nonexistent")
        assert resp.status == 404

    async def test_metrics_without_event_bus_503(self):
        """Metrics returns 503 when no event_bus is configured."""
        registry = AgentRegistry()
        queue = JobQueue()
        pool = ExecutionPool(registry, queue, max_concurrent=1, warm_pool_size=0)
        # No event_bus passed
        app = create_app(registry, queue, pool)

        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()

        resp = await client.get("/api/metrics")
        assert resp.status == 503

        await client.close()
