"""Tests for skills HTTP API routes."""

from __future__ import annotations

from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer

from atlas.contract.registry import AgentRegistry
from atlas.events import EventBus
from atlas.pool.executor import ExecutionPool
from atlas.pool.queue import JobQueue
from atlas.serve import create_app
from atlas.skills.registry import SkillRegistry

AGENTS_DIR = Path(__file__).parent.parent / "agents"
SKILLS_DIR = Path(__file__).parent.parent / "skills"


@pytest.fixture
async def client_with_skills():
    registry = AgentRegistry(search_paths=[AGENTS_DIR])
    registry.discover()

    skill_registry = SkillRegistry(search_paths=[SKILLS_DIR])
    skill_registry.discover()

    bus = EventBus()
    queue = JobQueue(event_bus=bus)
    pool = ExecutionPool(registry, queue, max_concurrent=2, warm_pool_size=0)

    app = create_app(
        registry, queue, pool, event_bus=bus,
        skill_registry=skill_registry,
    )
    await pool.start()

    server = TestServer(app)
    test_client = TestClient(server)
    await test_client.start_server()

    yield test_client

    await test_client.close()
    await pool.stop()


@pytest.fixture
async def client_without_skills():
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


class TestSkillsEndpoint:
    @pytest.mark.asyncio
    async def test_get_skills(self, client_with_skills):
        resp = await client_with_skills.get("/api/skills")
        assert resp.status == 200
        data = await resp.json()
        assert len(data) >= 1
        names = [s["name"] for s in data]
        assert "reverse" in names
        # Check schema fields are present
        skill = next(s for s in data if s["name"] == "reverse")
        assert "input_schema" in skill
        assert "output_schema" in skill

    @pytest.mark.asyncio
    async def test_no_skills_configured(self, client_without_skills):
        resp = await client_without_skills.get("/api/skills")
        assert resp.status == 404
        data = await resp.json()
        assert "not configured" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_skills_have_version(self, client_with_skills):
        resp = await client_with_skills.get("/api/skills")
        data = await resp.json()
        for skill in data:
            assert "version" in skill

    @pytest.mark.asyncio
    async def test_health_still_works(self, client_with_skills):
        """Existing endpoints still work with skill wiring."""
        resp = await client_with_skills.get("/api/health")
        assert resp.status == 200
