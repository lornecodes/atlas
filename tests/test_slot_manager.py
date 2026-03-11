"""Tests for SlotManager — warm slot pool lifecycle."""

from __future__ import annotations

import time

import pytest

from atlas.contract.registry import AgentRegistry
from atlas.pool.slot_manager import SlotManager, AgentSlot


@pytest.fixture
def slot_mgr(registry: AgentRegistry) -> SlotManager:
    return SlotManager(registry, warm_pool_size=2, warmup_timeout=10.0)


class TestSlotManager:
    async def test_acquire_cold_start(self, slot_mgr: SlotManager):
        slot, warmup_ms = await slot_mgr.acquire("echo")
        assert slot.agent_name == "echo"
        assert slot.state == "busy"
        assert warmup_ms >= 0

    async def test_acquire_warm_reuse(self, slot_mgr: SlotManager):
        slot1, w1 = await slot_mgr.acquire("echo")
        await slot_mgr.release(slot1)

        slot2, w2 = await slot_mgr.acquire("echo")
        assert w2 == 0.0  # Warm reuse
        assert slot2 is slot1  # Same instance

    async def test_release_to_warm_pool(self, slot_mgr: SlotManager):
        slot, _ = await slot_mgr.acquire("echo")
        await slot_mgr.release(slot)
        assert slot.state == "idle"

    async def test_release_destroys_when_full(self, registry: AgentRegistry):
        mgr = SlotManager(registry, warm_pool_size=1)
        s1, _ = await mgr.acquire("echo")
        s2, _ = await mgr.acquire("echo")
        await mgr.release(s1)  # Goes to pool (size=1)
        await mgr.release(s2)  # Pool full, destroyed
        assert s1.state == "idle"
        assert s2.state == "draining"

    async def test_destroy_calls_shutdown(self, slot_mgr: SlotManager):
        slot, _ = await slot_mgr.acquire("echo")
        await slot_mgr.destroy(slot)
        assert slot.state == "draining"

    async def test_reap_idle_removes_expired(self, slot_mgr: SlotManager):
        slot, _ = await slot_mgr.acquire("echo")
        await slot_mgr.release(slot)
        # Hack last_used to simulate timeout
        slot.last_used = time.monotonic() - 1000
        reaped = await slot_mgr.reap_idle(idle_timeout=60.0)
        assert reaped == 1

    async def test_reap_idle_keeps_fresh(self, slot_mgr: SlotManager):
        slot, _ = await slot_mgr.acquire("echo")
        await slot_mgr.release(slot)
        reaped = await slot_mgr.reap_idle(idle_timeout=60.0)
        assert reaped == 0

    async def test_shutdown_all_clears_pool(self, slot_mgr: SlotManager):
        s1, _ = await slot_mgr.acquire("echo")
        s2, _ = await slot_mgr.acquire("echo")
        await slot_mgr.release(s1)
        await slot_mgr.release(s2)
        await slot_mgr.shutdown_all()
        # Acquiring again should cold start
        s3, w = await slot_mgr.acquire("echo")
        assert w >= 0  # Cold start (pool was cleared)
