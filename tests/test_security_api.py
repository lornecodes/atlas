"""Tests for security HTTP API routes."""

from __future__ import annotations

from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer

from atlas.contract.registry import AgentRegistry
from atlas.events import EventBus
from atlas.pool.executor import ExecutionPool
from atlas.pool.queue import JobQueue
from atlas.security.policy import SecurityPolicy
from atlas.serve import create_app

AGENTS_DIR = Path(__file__).parent.parent / "agents"


@pytest.fixture
async def client_with_policy():
    registry = AgentRegistry(search_paths=[AGENTS_DIR])
    registry.discover()

    bus = EventBus()
    queue = JobQueue(event_bus=bus)
    policy = SecurityPolicy(
        container_image="test:v1",
        max_memory_mb=512,
        allowed_secrets={"SECRET_A"},
    )
    pool = ExecutionPool(
        registry, queue,
        max_concurrent=2, warm_pool_size=0,
        security_policy=policy,
    )

    app = create_app(registry, queue, pool, event_bus=bus, security_policy=policy)
    await pool.start()

    server = TestServer(app)
    test_client = TestClient(server)
    await test_client.start_server()

    yield test_client

    await test_client.close()
    await pool.stop()


@pytest.fixture
async def client_without_policy():
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


class TestSecurityPolicyEndpoint:
    @pytest.mark.asyncio
    async def test_get_policy(self, client_with_policy):
        resp = await client_with_policy.get("/api/security/policy")
        assert resp.status == 200
        data = await resp.json()
        assert data["container_image"] == "test:v1"
        assert data["max_memory_mb"] == 512
        assert "SECRET_A" in data["allowed_secrets"]

    @pytest.mark.asyncio
    async def test_no_policy_configured(self, client_without_policy):
        resp = await client_without_policy.get("/api/security/policy")
        assert resp.status == 404
        data = await resp.json()
        assert "No security policy" in data["error"]

    @pytest.mark.asyncio
    async def test_policy_has_default_permissions(self, client_with_policy):
        resp = await client_with_policy.get("/api/security/policy")
        data = await resp.json()
        assert "default_permissions" in data
        perms = data["default_permissions"]
        assert "filesystem" in perms
        assert "network" in perms

    @pytest.mark.asyncio
    async def test_policy_secret_provider(self, client_with_policy):
        resp = await client_with_policy.get("/api/security/policy")
        data = await resp.json()
        assert data["secret_provider"] == "env"

    @pytest.mark.asyncio
    async def test_health_still_works(self, client_with_policy):
        """Ensure existing endpoints still work with security wiring."""
        resp = await client_with_policy.get("/api/health")
        assert resp.status == 200
        data = await resp.json()
        assert data["status"] == "ok"
