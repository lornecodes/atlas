"""Tests for hardware scheduling — inventory, slot integration, executor, schema, health."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from atlas.contract.types import AgentContract, HardwareSpec
from atlas.pool.hardware import (
    HardwareInventory,
    ResourceAllocation,
    ResourceUnavailable,
    describe_requirement,
)


# ---------------------------------------------------------------------------
# TestHardwareInventory
# ---------------------------------------------------------------------------


class TestHardwareInventory:
    def test_defaults(self):
        inv = HardwareInventory()
        assert inv.total_gpus == 0
        assert inv.free_gpus == 0
        assert inv.total_memory_gb == 0
        assert inv.total_cpu_cores == 0
        assert inv.architecture == "any"

    def test_free_gpus(self):
        inv = HardwareInventory(total_gpus=4)
        assert inv.free_gpus == 4

    def test_free_memory(self):
        inv = HardwareInventory(total_memory_gb=64)
        assert inv.free_memory_gb == 64

    def test_free_cpu_cores(self):
        inv = HardwareInventory(total_cpu_cores=16)
        assert inv.free_cpu_cores == 16

    # -- can_satisfy --

    def test_can_satisfy_default_spec(self):
        inv = HardwareInventory()
        spec = HardwareSpec()
        assert inv.can_satisfy(spec)

    def test_can_satisfy_gpu_available(self):
        inv = HardwareInventory(total_gpus=2)
        spec = HardwareSpec(gpu=True)
        assert inv.can_satisfy(spec)

    def test_can_satisfy_gpu_unavailable(self):
        inv = HardwareInventory(total_gpus=0)
        spec = HardwareSpec(gpu=True)
        assert not inv.can_satisfy(spec)

    def test_can_satisfy_gpu_vram(self):
        inv = HardwareInventory(total_gpus=2, gpu_vram_gb=[8, 24])
        spec = HardwareSpec(gpu=True, gpu_vram_gb=16)
        assert inv.can_satisfy(spec)  # GPU 1 has 24GB

    def test_can_satisfy_gpu_vram_insufficient(self):
        inv = HardwareInventory(total_gpus=2, gpu_vram_gb=[8, 8])
        spec = HardwareSpec(gpu=True, gpu_vram_gb=16)
        assert not inv.can_satisfy(spec)

    def test_can_satisfy_memory(self):
        inv = HardwareInventory(total_memory_gb=64)
        spec = HardwareSpec(min_memory_gb=32)
        assert inv.can_satisfy(spec)

    def test_can_satisfy_memory_insufficient(self):
        inv = HardwareInventory(total_memory_gb=16)
        spec = HardwareSpec(min_memory_gb=32)
        assert not inv.can_satisfy(spec)

    def test_can_satisfy_cpu(self):
        inv = HardwareInventory(total_cpu_cores=16)
        spec = HardwareSpec(min_cpu_cores=8)
        assert inv.can_satisfy(spec)

    def test_can_satisfy_cpu_insufficient(self):
        inv = HardwareInventory(total_cpu_cores=4)
        spec = HardwareSpec(min_cpu_cores=8)
        assert not inv.can_satisfy(spec)

    def test_can_satisfy_architecture_match(self):
        inv = HardwareInventory(architecture="x86_64")
        spec = HardwareSpec(architecture="x86_64")
        assert inv.can_satisfy(spec)

    def test_can_satisfy_architecture_mismatch(self):
        inv = HardwareInventory(architecture="x86_64")
        spec = HardwareSpec(architecture="arm64")
        assert not inv.can_satisfy(spec)

    def test_can_satisfy_architecture_any_pool(self):
        inv = HardwareInventory(architecture="any")
        spec = HardwareSpec(architecture="x86_64")
        assert inv.can_satisfy(spec)

    def test_can_satisfy_architecture_any_spec(self):
        inv = HardwareInventory(architecture="x86_64")
        spec = HardwareSpec(architecture="any")
        assert inv.can_satisfy(spec)

    def test_can_satisfy_device_access(self):
        inv = HardwareInventory(available_devices=["/dev/fpga0", "/dev/tpu0"])
        spec = HardwareSpec(device_access=["/dev/fpga0"])
        assert inv.can_satisfy(spec)

    def test_can_satisfy_device_access_missing(self):
        inv = HardwareInventory(available_devices=["/dev/fpga0"])
        spec = HardwareSpec(device_access=["/dev/tpu0"])
        assert not inv.can_satisfy(spec)

    def test_can_satisfy_no_tracking_always_passes(self):
        """Zero-resource pool means no tracking — everything passes."""
        inv = HardwareInventory()
        spec = HardwareSpec(min_memory_gb=1024, min_cpu_cores=256)
        assert inv.can_satisfy(spec)

    # -- allocate / release --

    def test_allocate_gpu(self):
        inv = HardwareInventory(total_gpus=2)
        alloc = inv.allocate("slot-1", HardwareSpec(gpu=True))
        assert len(alloc.gpu_devices) == 1
        assert alloc.gpu_devices[0] in (0, 1)
        assert inv.free_gpus == 1

    def test_allocate_multiple_gpus(self):
        inv = HardwareInventory(total_gpus=3)
        a1 = inv.allocate("slot-1", HardwareSpec(gpu=True))
        a2 = inv.allocate("slot-2", HardwareSpec(gpu=True))
        assert a1.gpu_devices != a2.gpu_devices
        assert inv.free_gpus == 1

    def test_allocate_gpu_with_vram(self):
        inv = HardwareInventory(total_gpus=2, gpu_vram_gb=[8, 24])
        alloc = inv.allocate("slot-1", HardwareSpec(gpu=True, gpu_vram_gb=16))
        assert alloc.gpu_devices == [1]  # Only GPU 1 has enough VRAM

    def test_allocate_memory_and_cpu(self):
        inv = HardwareInventory(total_memory_gb=64, total_cpu_cores=16)
        alloc = inv.allocate("slot-1", HardwareSpec(min_memory_gb=16, min_cpu_cores=4))
        assert alloc.memory_gb == 16
        assert alloc.cpu_cores == 4
        assert inv.free_memory_gb == 48
        assert inv.free_cpu_cores == 12

    def test_allocate_raises_when_insufficient(self):
        inv = HardwareInventory(total_gpus=1)
        inv.allocate("slot-1", HardwareSpec(gpu=True))
        with pytest.raises(ResourceUnavailable):
            inv.allocate("slot-2", HardwareSpec(gpu=True))

    def test_release(self):
        inv = HardwareInventory(total_gpus=2, total_memory_gb=64)
        inv.allocate("slot-1", HardwareSpec(gpu=True, min_memory_gb=16))
        assert inv.free_gpus == 1
        assert inv.free_memory_gb == 48

        inv.release("slot-1")
        assert inv.free_gpus == 2
        assert inv.free_memory_gb == 64

    def test_release_nonexistent_is_noop(self):
        inv = HardwareInventory(total_gpus=2)
        inv.release("nonexistent")
        assert inv.free_gpus == 2

    def test_allocate_then_release_then_reallocate(self):
        inv = HardwareInventory(total_gpus=1)
        inv.allocate("slot-1", HardwareSpec(gpu=True))
        inv.release("slot-1")
        alloc = inv.allocate("slot-2", HardwareSpec(gpu=True))
        assert len(alloc.gpu_devices) == 1

    def test_allocated_gpu_indices(self):
        inv = HardwareInventory(total_gpus=4)
        inv.allocate("slot-1", HardwareSpec(gpu=True))
        inv.allocate("slot-2", HardwareSpec(gpu=True))
        assert len(inv.allocated_gpu_indices) == 2

    # -- status --

    def test_status_empty(self):
        inv = HardwareInventory(total_gpus=2, total_memory_gb=64, total_cpu_cores=16)
        s = inv.status()
        assert s["total_gpus"] == 2
        assert s["free_gpus"] == 2
        assert s["total_memory_gb"] == 64
        assert s["free_memory_gb"] == 64
        assert s["allocations"] == []

    def test_status_with_allocation(self):
        inv = HardwareInventory(total_gpus=2, total_memory_gb=64)
        inv.allocate("slot-1", HardwareSpec(gpu=True, min_memory_gb=16))
        s = inv.status()
        assert s["free_gpus"] == 1
        assert s["free_memory_gb"] == 48
        assert len(s["allocations"]) == 1
        assert s["allocations"][0]["slot_id"] == "slot-1"


# ---------------------------------------------------------------------------
# TestDescribeRequirement
# ---------------------------------------------------------------------------


class TestDescribeRequirement:
    def test_default_spec(self):
        assert describe_requirement(HardwareSpec()) == "default"

    def test_gpu(self):
        desc = describe_requirement(HardwareSpec(gpu=True))
        assert "GPU" in desc

    def test_gpu_with_vram(self):
        desc = describe_requirement(HardwareSpec(gpu=True, gpu_vram_gb=24))
        assert "24GB VRAM" in desc

    def test_memory(self):
        desc = describe_requirement(HardwareSpec(min_memory_gb=32))
        assert "32GB RAM" in desc

    def test_cpu(self):
        desc = describe_requirement(HardwareSpec(min_cpu_cores=8))
        assert "8 CPU" in desc

    def test_architecture(self):
        desc = describe_requirement(HardwareSpec(architecture="arm64"))
        assert "arm64" in desc

    def test_device_access(self):
        desc = describe_requirement(HardwareSpec(device_access=["/dev/fpga0"]))
        assert "fpga0" in desc


# ---------------------------------------------------------------------------
# TestResourceAllocation
# ---------------------------------------------------------------------------


class TestResourceAllocation:
    def test_defaults(self):
        alloc = ResourceAllocation(slot_id="s1")
        assert alloc.gpu_devices == []
        assert alloc.memory_gb == 0
        assert alloc.cpu_cores == 0


# ---------------------------------------------------------------------------
# TestSlotManagerHardware
# ---------------------------------------------------------------------------


class TestSlotManagerHardware:
    @pytest.fixture
    def hw_inventory(self):
        return HardwareInventory(
            total_gpus=2,
            gpu_vram_gb=[16, 24],
            total_memory_gb=64,
            total_cpu_cores=16,
        )

    @pytest.fixture
    def slot_mgr_hw(self, registry, hw_inventory):
        from atlas.pool.slot_manager import SlotManager
        return SlotManager(registry, warm_pool_size=2, hardware=hw_inventory)

    @pytest.fixture
    def slot_mgr_no_hw(self, registry):
        from atlas.pool.slot_manager import SlotManager
        return SlotManager(registry, warm_pool_size=2)

    async def test_acquire_with_hardware(self, slot_mgr_hw, hw_inventory):
        spec = HardwareSpec(gpu=True, min_memory_gb=8)
        slot, warmup = await slot_mgr_hw.acquire("echo", hardware_spec=spec)
        assert slot.hardware is not None
        assert len(slot.hardware.gpu_devices) == 1
        assert hw_inventory.free_gpus == 1

    async def test_acquire_no_hardware_configured(self, slot_mgr_no_hw):
        """No hardware inventory — behaves exactly as before."""
        spec = HardwareSpec(gpu=True)
        slot, warmup = await slot_mgr_no_hw.acquire("echo", hardware_spec=spec)
        assert slot.hardware is None  # No tracking

    async def test_acquire_no_spec(self, slot_mgr_hw, hw_inventory):
        """No hardware spec — no allocation."""
        slot, warmup = await slot_mgr_hw.acquire("echo")
        assert slot.hardware is None
        assert hw_inventory.free_gpus == 2

    async def test_acquire_insufficient_raises(self, slot_mgr_hw):
        spec = HardwareSpec(gpu=True)
        await slot_mgr_hw.acquire("echo", hardware_spec=spec)
        await slot_mgr_hw.acquire("echo", hardware_spec=spec)
        with pytest.raises(ResourceUnavailable):
            await slot_mgr_hw.acquire("echo", hardware_spec=spec)

    async def test_release_frees_hardware(self, slot_mgr_hw, hw_inventory):
        spec = HardwareSpec(gpu=True)
        slot, _ = await slot_mgr_hw.acquire("echo", hardware_spec=spec)
        assert hw_inventory.free_gpus == 1
        await slot_mgr_hw.release(slot)
        assert hw_inventory.free_gpus == 2

    async def test_destroy_frees_hardware(self, slot_mgr_hw, hw_inventory):
        spec = HardwareSpec(gpu=True)
        slot, _ = await slot_mgr_hw.acquire("echo", hardware_spec=spec)
        assert hw_inventory.free_gpus == 1
        await slot_mgr_hw.destroy(slot)
        assert hw_inventory.free_gpus == 2

    async def test_warm_reuse_allocates_hardware(self, slot_mgr_hw, hw_inventory):
        """Warm slot reuse should still allocate hardware."""
        slot1, _ = await slot_mgr_hw.acquire("echo")
        await slot_mgr_hw.release(slot1)
        # Now reuse warm slot with hardware spec
        spec = HardwareSpec(gpu=True)
        slot2, warmup = await slot_mgr_hw.acquire("echo", hardware_spec=spec)
        assert warmup == 0.0  # Warm reuse
        assert slot2.hardware is not None
        assert hw_inventory.free_gpus == 1

    async def test_reap_idle_releases_hardware(self, slot_mgr_hw, hw_inventory):
        spec = HardwareSpec(gpu=True)
        slot, _ = await slot_mgr_hw.acquire("echo", hardware_spec=spec)
        await slot_mgr_hw.release(slot)
        # Hardware should already be released on release()
        assert hw_inventory.free_gpus == 2
        # Reap should not double-free
        slot.last_used = time.monotonic() - 1000
        reaped = await slot_mgr_hw.reap_idle(60.0)
        assert reaped == 1
        assert hw_inventory.free_gpus == 2

    async def test_shutdown_all_releases_hardware(self, slot_mgr_hw, hw_inventory):
        spec = HardwareSpec(gpu=True)
        slot, _ = await slot_mgr_hw.acquire("echo", hardware_spec=spec)
        await slot_mgr_hw.release(slot)
        await slot_mgr_hw.shutdown_all()
        assert hw_inventory.free_gpus == 2

    async def test_slot_has_unique_id(self, slot_mgr_hw):
        s1, _ = await slot_mgr_hw.acquire("echo")
        s2, _ = await slot_mgr_hw.acquire("echo")
        assert s1.slot_id != s2.slot_id
        assert s1.slot_id.startswith("slot-")


# ---------------------------------------------------------------------------
# TestSchemaHardware
# ---------------------------------------------------------------------------


class TestSchemaHardware:
    def _make_yaml(self, hardware_block: str) -> str:
        return (
            "agent:\n"
            "  name: test-hw\n"
            "  version: '1.0.0'\n"
            f"  hardware:\n{hardware_block}"
        )

    def test_valid_gpu(self):
        import yaml
        content = self._make_yaml("    gpu: true\n    gpu_vram_gb: 16\n")
        data = yaml.safe_load(content)
        from atlas.contract.schema import _meta_validator
        errors = list(_meta_validator.iter_errors(data))
        assert errors == []

    def test_valid_full_spec(self):
        import yaml
        content = self._make_yaml(
            "    gpu: true\n"
            "    gpu_vram_gb: 24\n"
            "    min_memory_gb: 32\n"
            "    min_cpu_cores: 8\n"
            "    architecture: x86_64\n"
            "    node_affinity: gpu-pool\n"
            "    device_access:\n"
            "      - /dev/fpga0\n"
        )
        data = yaml.safe_load(content)
        from atlas.contract.schema import _meta_validator
        errors = list(_meta_validator.iter_errors(data))
        assert errors == []

    def test_invalid_architecture(self):
        import yaml
        content = (
            "agent:\n"
            "  name: test-hw\n"
            "  version: '1.0.0'\n"
            "  hardware:\n"
            "    architecture: mips\n"
        )
        data = yaml.safe_load(content)
        from atlas.contract.schema import _meta_validator
        errors = list(_meta_validator.iter_errors(data))
        assert len(errors) > 0

    def test_additional_properties_rejected(self):
        import yaml
        content = (
            "agent:\n"
            "  name: test-hw\n"
            "  version: '1.0.0'\n"
            "  hardware:\n"
            "    bogus_field: true\n"
        )
        data = yaml.safe_load(content)
        from atlas.contract.schema import _meta_validator
        errors = list(_meta_validator.iter_errors(data))
        assert len(errors) > 0

    def test_empty_hardware_valid(self):
        import yaml
        content = (
            "agent:\n"
            "  name: test-hw\n"
            "  version: '1.0.0'\n"
            "  hardware: {}\n"
        )
        data = yaml.safe_load(content)
        from atlas.contract.schema import _meta_validator
        errors = list(_meta_validator.iter_errors(data))
        assert errors == []


# ---------------------------------------------------------------------------
# TestHealthHardware
# ---------------------------------------------------------------------------


class TestHealthHardware:
    @pytest.fixture
    def pool_with_hw(self, registry):
        from atlas.pool.executor import ExecutionPool
        from atlas.pool.queue import JobQueue
        inv = HardwareInventory(total_gpus=2, total_memory_gb=64)
        queue = JobQueue()
        return ExecutionPool(registry, queue, hardware=inv)

    @pytest.fixture
    def pool_no_hw(self, registry):
        from atlas.pool.executor import ExecutionPool
        from atlas.pool.queue import JobQueue
        queue = JobQueue()
        return ExecutionPool(registry, queue)

    def test_pool_has_hardware(self, pool_with_hw):
        assert pool_with_hw._hardware is not None
        assert pool_with_hw._hardware.total_gpus == 2

    def test_pool_no_hardware(self, pool_no_hw):
        assert pool_no_hw._hardware is None

    def test_hardware_status_dict(self):
        inv = HardwareInventory(total_gpus=2, total_memory_gb=64, total_cpu_cores=16)
        inv.allocate("slot-1", HardwareSpec(gpu=True, min_memory_gb=16))
        s = inv.status()
        assert s["free_gpus"] == 1
        assert s["free_memory_gb"] == 48
        assert s["free_cpu_cores"] == 15  # Default spec min_cpu_cores=1 allocated
        assert len(s["allocations"]) == 1


# ---------------------------------------------------------------------------
# TestCLIHardware
# ---------------------------------------------------------------------------


class TestCLIHardware:
    def test_hardware_inventory_from_params(self):
        inv = HardwareInventory(
            total_gpus=2,
            gpu_vram_gb=[16, 24],
            total_memory_gb=128,
            total_cpu_cores=32,
            architecture="x86_64",
            available_devices=["/dev/fpga0"],
        )
        assert inv.total_gpus == 2
        assert inv.gpu_vram_gb == [16, 24]
        assert inv.total_memory_gb == 128
        assert inv.total_cpu_cores == 32
        assert inv.architecture == "x86_64"
        assert inv.available_devices == ["/dev/fpga0"]

    def test_gpu_vram_parsing(self):
        """Simulate CLI --gpu-vram '16,24' parsing."""
        gpu_vram = "16,24"
        vram_list = [int(v.strip()) for v in gpu_vram.split(",")]
        assert vram_list == [16, 24]

    def test_no_hardware_flags(self):
        """No hardware flags → no inventory."""
        inv = None
        gpu_count = 0
        pool_memory = 0
        pool_cpus = 0
        pool_arch = "any"
        pool_devices = None
        if gpu_count or pool_memory or pool_cpus or pool_arch != "any" or pool_devices:
            inv = HardwareInventory()
        assert inv is None


# ---------------------------------------------------------------------------
# TestExecutorHardware (integration-style)
# ---------------------------------------------------------------------------


class TestExecutorHardware:
    @pytest.fixture
    def hw_inventory(self):
        return HardwareInventory(total_gpus=1, total_memory_gb=32, total_cpu_cores=8)

    async def test_pool_passes_hardware_to_slots(self, registry, hw_inventory):
        from atlas.pool.executor import ExecutionPool
        from atlas.pool.queue import JobQueue
        queue = JobQueue()
        pool = ExecutionPool(registry, queue, hardware=hw_inventory)
        assert pool._slots._hardware is hw_inventory

    async def test_pool_without_hardware(self, registry):
        from atlas.pool.executor import ExecutionPool
        from atlas.pool.queue import JobQueue
        queue = JobQueue()
        pool = ExecutionPool(registry, queue)
        assert pool._slots._hardware is None
