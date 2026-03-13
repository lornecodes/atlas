#!/usr/bin/env python
"""Demo: Trigger-driven workflow.

Creates an interval trigger that fires every 2 seconds, submitting echo jobs
to the pool. Runs for ~10 seconds to show trigger -> pool -> completion flow.

Usage:
    python demos/triggered_workflow.py
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from atlas.contract.registry import AgentRegistry
from atlas.events import EventBus
from atlas.pool.executor import ExecutionPool
from atlas.pool.job import JobData
from atlas.pool.queue import JobQueue
from atlas.store.job_store import JobStore
from atlas.store.trigger_store import TriggerStore
from atlas.triggers.models import TriggerDefinition
from atlas.triggers.scheduler import TriggerScheduler

AGENTS_DIR = Path(__file__).parent.parent / "agents"


async def main():
    # --- Wire up ---
    bus = EventBus()
    store = JobStore(":memory:")
    await store.init()
    queue = JobQueue(max_size=100, store=store, event_bus=bus)
    registry = AgentRegistry(search_paths=[AGENTS_DIR])
    registry.discover()

    pool = ExecutionPool(registry, queue, max_concurrent=4, warm_pool_size=1)
    await pool.start()

    trigger_store = TriggerStore(":memory:")
    await trigger_store.init()

    # --- Create an interval trigger ---
    trigger = TriggerDefinition(
        name="demo-heartbeat",
        trigger_type="interval",
        agent_name="echo",
        input_data={"message": "heartbeat"},
        interval_seconds=2.0,
    )
    trigger.next_fire = time.time()  # Fire immediately on first tick
    await trigger_store.save(trigger)
    print(f"Created trigger: {trigger.id} ({trigger.name})")

    # --- Track completions via EventBus ---
    completed_jobs = []

    async def on_event(job: JobData, old_status: str, new_status: str):
        if new_status == "completed":
            completed_jobs.append(job)
            print(f"  Job completed: {job.id} -> {job.output_data}")

    bus.subscribe(on_event)

    # --- Start scheduler ---
    scheduler = TriggerScheduler(
        trigger_store, pool, event_bus=bus, poll_interval=1.0
    )
    await scheduler.start()
    print("Scheduler started -- firing every 2 seconds...")
    print()

    # --- Let it run for 10 seconds ---
    await asyncio.sleep(10)

    # --- Stop ---
    await scheduler.stop()

    # Show final trigger state before closing stores
    final_trigger = await trigger_store.get(trigger.id)
    fire_count = final_trigger.fire_count if final_trigger else 0

    await pool.stop(timeout=5.0)
    await trigger_store.close()
    await store.close()

    print(f"\nCompleted {len(completed_jobs)} jobs in ~10 seconds")
    print(f"Trigger fire_count: {fire_count}")
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
