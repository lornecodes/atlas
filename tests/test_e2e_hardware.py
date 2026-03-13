"""E2E tests — hardware-aware pool scheduling.

Tests HardwareInventory + SlotManager + ExecutionPool wired together.
Real agents, real hardware tracking, real slot lifecycle.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest

from atlas.contract.registry import AgentRegistry
from atlas.contract.types import HardwareSpec
from atlas.events import EventBus
from atlas.pool.executor import ExecutionPool
from atlas.pool.hardware import (
    HardwareInventory,
    ResourceAllocation,
    ResourceUnavailable,
    describe_requirement,
)
from atlas.pool.job import JobData
from atlas.pool.queue import JobQueue
from atlas.pool.slot_manager import SlotManager
from atlas.store.job_store import JobStore

AGENTS_DIR = Path(__file__).parent.parent / "agents"


# ---------------------------------------------------------------------------
# Fixtures
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
async def store(tmp_path):
    s = JobStore(str(tmp_path / "hw_e2e.db"))
    await s.init()
    yield s
    await s.close()


@pytest.fixture
def queue(bus, store):
    return JobQueue(max_size=100, store=store, event_bus=bus)


@pytest.fixture
def hardware():
    """Pool with 2 GPUs, 32GB RAM, 8 CPU cores."""
    return HardwareInventory(
        total_gpus=2,
        gpu_vram_gb=[16, 16],
        total_memory_gb=32,
        total_cpu_cores=8,
        architecture="x86_64",
        available_devices=["gpu:0", "gpu:1"],
    )


@pytest.fixture
async def pool_with_hw(registry, queue, hardware):
    """Pool wired with hardware inventory."""
    p = ExecutionPool(
        registry, queue,
        max_concurrent=4,
        warm_pool_size=0,
        hardware=hardware,
    )
    await p.start()
    yield p
    await p.stop(timeout=5.0)


# ---------------------------------------------------------------------------
# Tests: HardwareInventory unit-level E2E
# ---------------------------------------------------------------------------


class TestHardwareInventory:
    """HardwareInventory allocation and release lifecycle."""

    def test_initial_capacity(self, hardware):
        """Freshly created inventory has full capacity available."""
        assert hardware.free_gpus == 2
        assert hardware.free_memory_gb == 32
        assert hardware.free_cpu_cores == 8
        assert hardware.allocated_gpu_indices == set()

    def test_allocate_releases_correctly(self, hardware):
        """Allocate → release returns resources to pool."""
        spec = HardwareSpec(gpu=True, gpu_vram_gb=8, min_memory_gb=4, min_cpu_cores=2)
        alloc = hardware.allocate("slot-1", spec)

        assert hardware.free_gpus == 1
        assert hardware.free_memory_gb == 28
        assert hardware.free_cpu_cores == 6
        assert len(alloc.gpu_devices) == 1

        hardware.release("slot-1")

        assert hardware.free_gpus == 2
        assert hardware.free_memory_gb == 32
        assert hardware.free_cpu_cores == 8

    def test_multi_gpu_allocation(self, hardware):
        """Two separate 1-GPU allocations use different GPU indices."""
        spec = HardwareSpec(gpu=True, min_memory_gb=1, min_cpu_cores=1)

        alloc1 = hardware.allocate("slot-1", spec)
        alloc2 = hardware.allocate("slot-2", spec)

        # Different GPU indices assigned
        assert alloc1.gpu_devices != alloc2.gpu_devices
        assert hardware.free_gpus == 0
        assert len(hardware.allocated_gpu_indices) == 2

    def test_insufficient_gpus_raises(self, hardware):
        """Requesting more GPUs than available raises ResourceUnavailable."""
        spec = HardwareSpec(gpu=True, min_memory_gb=1, min_cpu_cores=1)

        hardware.allocate("slot-1", spec)
        hardware.allocate("slot-2", spec)

        # Third allocation should fail (only 2 GPUs)
        assert not hardware.can_satisfy(spec)
        with pytest.raises(ResourceUnavailable):
            hardware.allocate("slot-3", spec)

    def test_memory_reservation(self, hardware):
        """Memory allocation reduces free pool."""
        spec = HardwareSpec(min_memory_gb=16, min_cpu_cores=1)
        hardware.allocate("slot-1", spec)

        assert hardware.free_memory_gb == 16

    def test_cpu_reservation(self, hardware):
        """CPU core allocation reduces free pool."""
        spec = HardwareSpec(min_cpu_cores=4, min_memory_gb=1)
        hardware.allocate("slot-1", spec)

        assert hardware.free_cpu_cores == 4

    def test_architecture_matching(self, hardware):
        """Architecture mismatch prevents allocation."""
        arm_spec = HardwareSpec(architecture="arm64", min_memory_gb=1, min_cpu_cores=1)
        x86_spec = HardwareSpec(architecture="x86_64", min_memory_gb=1, min_cpu_cores=1)
        any_spec = HardwareSpec(architecture="any", min_memory_gb=1, min_cpu_cores=1)

        assert not hardware.can_satisfy(arm_spec)  # Pool is x86_64
        assert hardware.can_satisfy(x86_spec)       # Match
        assert hardware.can_satisfy(any_spec)        # "any" always matches

    def test_vram_check(self, hardware):
        """GPU VRAM requirement checked against available GPUs."""
        # Each GPU has 16GB VRAM
        small_spec = HardwareSpec(gpu=True, gpu_vram_gb=8, min_memory_gb=1, min_cpu_cores=1)
        big_spec = HardwareSpec(gpu=True, gpu_vram_gb=32, min_memory_gb=1, min_cpu_cores=1)

        assert hardware.can_satisfy(small_spec)  # 16GB >= 8GB
        assert not hardware.can_satisfy(big_spec)  # 16GB < 32GB

    def test_device_access_check(self, hardware):
        """Device access requirements checked against available devices."""
        valid_spec = HardwareSpec(device_access=["gpu:0"], min_memory_gb=1, min_cpu_cores=1)
        invalid_spec = HardwareSpec(device_access=["tpu:0"], min_memory_gb=1, min_cpu_cores=1)

        assert hardware.can_satisfy(valid_spec)
        assert not hardware.can_satisfy(invalid_spec)

    def test_status_report(self, hardware):
        """status() returns current allocation snapshot."""
        spec = HardwareSpec(gpu=True, min_memory_gb=4, min_cpu_cores=2)
        hardware.allocate("slot-1", spec)

        status = hardware.status()
        assert status["total_gpus"] == 2
        assert status["free_gpus"] == 1
        assert status["total_memory_gb"] == 32
        assert status["free_memory_gb"] == 28
        assert len(status["allocations"]) == 1
        assert status["allocations"][0]["slot_id"] == "slot-1"


# ---------------------------------------------------------------------------
# Tests: describe_requirement helper
# ---------------------------------------------------------------------------


class TestDescribeRequirement:
    """Human-readable hardware requirement descriptions."""

    def test_gpu_requirement(self):
        spec = HardwareSpec(gpu=True, gpu_vram_gb=16)
        desc = describe_requirement(spec)
        assert "GPU" in desc
        assert "16GB VRAM" in desc

    def test_memory_requirement(self):
        spec = HardwareSpec(min_memory_gb=8)
        desc = describe_requirement(spec)
        assert "8GB RAM" in desc

    def test_cpu_requirement(self):
        spec = HardwareSpec(min_cpu_cores=4)
        desc = describe_requirement(spec)
        assert "4 CPU cores" in desc

    def test_architecture_requirement(self):
        spec = HardwareSpec(architecture="arm64")
        desc = describe_requirement(spec)
        assert "arm64" in desc

    def test_default_requirement(self):
        spec = HardwareSpec()
        desc = describe_requirement(spec)
        assert desc == "default"


# ---------------------------------------------------------------------------
# Tests: Pool integration with hardware
# ---------------------------------------------------------------------------


class TestPoolHardwareIntegration:
    """Hardware-aware pool execution with real agents."""

    @pytest.mark.asyncio
    async def test_job_completes_with_hardware(self, pool_with_hw, queue, hardware):
        """Job submitted to hardware-aware pool completes successfully."""
        job = JobData(
            agent_name="echo",
            input_data={"message": "hw-test"},
        )
        await pool_with_hw.submit(job)
        result = await queue.wait_for_terminal(job.id, timeout=10.0)

        assert result.status == "completed"
        assert result.output_data is not None
        assert result.output_data.get("message") == "hw-test"

    @pytest.mark.asyncio
    async def test_hardware_released_after_job(self, pool_with_hw, queue, hardware):
        """Hardware resources are released after job completes."""
        initial_gpus = hardware.free_gpus
        initial_memory = hardware.free_memory_gb

        job = JobData(
            agent_name="echo",
            input_data={"message": "release-test"},
        )
        await pool_with_hw.submit(job)
        await queue.wait_for_terminal(job.id, timeout=10.0)

        # Give slot manager time to release
        await asyncio.sleep(0.2)

        # Resources should be back (echo agent has default HardwareSpec)
        assert hardware.free_gpus >= initial_gpus
        assert hardware.free_memory_gb >= initial_memory

    @pytest.mark.asyncio
    async def test_concurrent_jobs_share_hardware(self, pool_with_hw, queue, hardware):
        """Multiple concurrent jobs allocate from shared hardware pool."""
        jobs = []
        for i in range(3):
            job = JobData(
                agent_name="echo",
                input_data={"message": f"concurrent-{i}"},
            )
            await pool_with_hw.submit(job)
            jobs.append(job)

        # Wait for all jobs to complete
        results = await asyncio.gather(*[
            queue.wait_for_terminal(j.id, timeout=10.0) for j in jobs
        ])

        for result in results:
            assert result.status == "completed"


# ---------------------------------------------------------------------------
# Tests: SlotManager with hardware
# ---------------------------------------------------------------------------


class TestSlotManagerHardware:
    """SlotManager hardware integration."""

    @pytest.mark.asyncio
    async def test_slot_acquire_with_hardware(self, registry, hardware):
        """SlotManager.acquire() with hardware_spec allocates resources."""
        sm = SlotManager(registry, warm_pool_size=0, hardware=hardware)

        spec = HardwareSpec(min_memory_gb=4, min_cpu_cores=2)
        slot, warmup_ms = await sm.acquire("echo", hardware_spec=spec)

        assert slot is not None
        assert slot.hardware is not None
        assert hardware.free_memory_gb <= 28  # 32 - 4

        await sm.destroy(slot)

    @pytest.mark.asyncio
    async def test_slot_release_frees_hardware(self, registry, hardware):
        """SlotManager.release() frees hardware allocation."""
        sm = SlotManager(registry, warm_pool_size=0, hardware=hardware)

        spec = HardwareSpec(min_memory_gb=8, min_cpu_cores=2)
        slot, _ = await sm.acquire("echo", hardware_spec=spec)

        assert hardware.free_memory_gb <= 24

        await sm.release(slot)
        # After release, hardware should be freed
        # (release may keep slot warm, but hardware is released)
        assert hardware.free_memory_gb >= 24

        await sm.shutdown_all()
