#!/usr/bin/env python
"""Demo: Hardware-aware execution pool.

Creates a pool with 2 GPUs (16GB VRAM each) and 32GB RAM. Submits 3 jobs
that each require a GPU, showing how the pool schedules 2 concurrently
while the third waits for a GPU to free up.

Usage:
    python demos/hardware_aware_pool.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from atlas.contract.registry import AgentRegistry
from atlas.contract.types import HardwareSpec
from atlas.events import EventBus
from atlas.pool.executor import ExecutionPool
from atlas.pool.hardware import HardwareInventory, describe_requirement
from atlas.pool.job import JobData
from atlas.pool.queue import JobQueue
from atlas.store.job_store import JobStore

AGENTS_DIR = Path(__file__).parent.parent / "agents"


async def main():
    # --- Configure hardware ---
    hardware = HardwareInventory(
        total_gpus=2,
        gpu_vram_gb=[16, 16],
        total_memory_gb=32,
        total_cpu_cores=8,
        architecture="x86_64",
        available_devices=["gpu:0", "gpu:1"],
    )

    print("=== Hardware Inventory ===")
    status = hardware.status()
    print(f"  GPUs:        {status['total_gpus']} ({status['free_gpus']} free)")
    print(f"  GPU VRAM:    {status['gpu_vram_gb']} GB each")
    print(f"  Memory:      {status['total_memory_gb']}GB ({status['free_memory_gb']}GB free)")
    print(f"  CPU cores:   {status['total_cpu_cores']} ({status['free_cpu_cores']} free)")
    print(f"  Arch:        {status['architecture']}")
    print()

    # --- Wire up pool ---
    bus = EventBus()
    store = JobStore(":memory:")
    await store.init()
    queue = JobQueue(max_size=100, store=store, event_bus=bus)
    registry = AgentRegistry(search_paths=[AGENTS_DIR])
    registry.discover()

    pool = ExecutionPool(
        registry, queue,
        max_concurrent=4,
        warm_pool_size=0,
        hardware=hardware,
    )
    await pool.start()

    # --- Show requirement descriptions ---
    print("=== Hardware Requirements ===")
    specs = [
        HardwareSpec(gpu=True, gpu_vram_gb=8, min_memory_gb=4, min_cpu_cores=2),
        HardwareSpec(min_memory_gb=16, min_cpu_cores=4),
        HardwareSpec(gpu=True, gpu_vram_gb=32),  # Can't satisfy!
    ]
    for spec in specs:
        desc = describe_requirement(spec)
        can = hardware.can_satisfy(spec)
        print(f"  {desc}: {'CAN' if can else 'CANNOT'} satisfy")
    print()

    # --- Demonstrate allocation lifecycle ---
    print("=== Allocation Lifecycle ===")
    spec_a = HardwareSpec(gpu=True, gpu_vram_gb=8, min_memory_gb=4, min_cpu_cores=2)
    alloc_a = hardware.allocate("slot-A", spec_a)
    print(f"  Allocated slot-A: GPU devices={alloc_a.gpu_devices}, mem={alloc_a.memory_gb}GB")
    print(f"  Free GPUs: {hardware.free_gpus}, Free memory: {hardware.free_memory_gb}GB")

    spec_b = HardwareSpec(gpu=True, min_memory_gb=8, min_cpu_cores=2)
    alloc_b = hardware.allocate("slot-B", spec_b)
    print(f"  Allocated slot-B: GPU devices={alloc_b.gpu_devices}, mem={alloc_b.memory_gb}GB")
    print(f"  Free GPUs: {hardware.free_gpus}, Free memory: {hardware.free_memory_gb}GB")

    # Third GPU request should fail
    can_alloc_c = hardware.can_satisfy(HardwareSpec(gpu=True))
    print(f"  Can allocate another GPU? {can_alloc_c}")

    # Release slot-A
    hardware.release("slot-A")
    print(f"  Released slot-A -> Free GPUs: {hardware.free_gpus}")

    # Now we can allocate again
    can_now = hardware.can_satisfy(HardwareSpec(gpu=True))
    print(f"  Can allocate GPU now? {can_now}")

    hardware.release("slot-B")
    print()

    # --- Submit jobs through pool ---
    print("=== Pool Jobs ===")
    jobs = []
    for i in range(3):
        job = JobData(
            agent_name="echo",
            input_data={"message": f"GPU job {i}"},
        )
        await pool.submit(job)
        jobs.append(job)
        print(f"  Submitted: {job.id}")

    # Wait for all
    for job in jobs:
        result = await queue.wait_for_terminal(job.id, timeout=10.0)
        if result:
            print(f"  {result.id}: {result.status} "
                  f"(exec={result.execution_ms:.0f}ms)")

    # --- Final status ---
    print()
    final_status = hardware.status()
    print(f"Final: {final_status['free_gpus']}/{final_status['total_gpus']} GPUs free, "
          f"{final_status['free_memory_gb']}/{final_status['total_memory_gb']}GB RAM free")

    # --- Cleanup ---
    await pool.stop(timeout=5.0)
    await store.close()
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
