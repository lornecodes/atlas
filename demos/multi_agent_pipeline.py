#!/usr/bin/env python
"""Demo: Multi-agent pipeline with metrics and tracing.

Submits 3 jobs (echo, formatter, batch-processor) to a real execution pool,
monitors completion via EventBus, and prints a metrics summary.

Usage:
    python demos/multi_agent_pipeline.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Add parent to path so atlas is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from atlas.contract.registry import AgentRegistry
from atlas.events import EventBus
from atlas.metrics import MetricsCollector
from atlas.pool.executor import ExecutionPool
from atlas.pool.job import JobData
from atlas.pool.queue import JobQueue
from atlas.store.job_store import JobStore
from atlas.trace import TraceCollector

AGENTS_DIR = Path(__file__).parent.parent / "agents"


async def main():
    # --- Wire up the full stack ---
    bus = EventBus()
    store = JobStore(":memory:")
    await store.init()
    metrics = MetricsCollector(bus)
    traces = TraceCollector(bus)
    queue = JobQueue(max_size=100, store=store, event_bus=bus)
    registry = AgentRegistry(search_paths=[AGENTS_DIR])
    count = registry.discover()
    print(f"Discovered {count} agents")

    pool = ExecutionPool(registry, queue, max_concurrent=4, warm_pool_size=2)
    await pool.start()

    # --- Submit 3 jobs ---
    jobs = [
        JobData(agent_name="echo", input_data={"message": "Hello from Atlas!"}),
        JobData(agent_name="formatter", input_data={"content": "raw text here", "style": "uppercase"}),
        JobData(agent_name="batch-processor", input_data={
            "items": [
                {"name": "Widget A", "price": 9.99},
                {"name": "Widget B", "price": 14.50},
                {"name": "Widget C", "price": 3.75},
            ],
            "currency": "USD",
        }),
    ]

    print("\n--- Submitting jobs ---")
    for job in jobs:
        await pool.submit(job)
        print(f"  Submitted: {job.id} ({job.agent_name})")

    # --- Wait for completion ---
    print("\n--- Waiting for completion ---")
    for job in jobs:
        result = await queue.wait_for_terminal(job.id, timeout=10.0)
        if result:
            print(f"  {result.agent_name}: status={result.status}")
            if result.output_data:
                print(f"    output: {result.output_data}")
            if result.error:
                print(f"    error: {result.error}")
            print(f"    warmup={result.warmup_ms:.0f}ms, exec={result.execution_ms:.0f}ms")

    # --- Print metrics ---
    print("\n--- Global Metrics ---")
    global_metrics = metrics.get_global_metrics()
    print(f"  Total jobs:     {global_metrics.get('total_jobs', 0)}")
    print(f"  Completed:      {global_metrics.get('completed', 0)}")
    print(f"  Failed:         {global_metrics.get('failed', 0)}")
    print(f"  Avg exec time:  {global_metrics.get('avg_execution_ms', 0):.1f}ms")

    print("\n--- Per-Agent Metrics ---")
    for job in jobs:
        agent_metrics = metrics.get_agent_metrics(job.agent_name)
        if agent_metrics:
            print(f"  {job.agent_name}: {agent_metrics}")

    # --- Print traces ---
    print("\n--- Traces ---")
    all_traces = traces.list()
    for t in all_traces:
        print(f"  {t.agent_name}: status={t.status}, "
              f"warmup={t.warmup_ms:.0f}ms, exec={t.execution_ms:.0f}ms")

    # --- Cleanup ---
    await pool.stop(timeout=5.0)
    metrics.close()
    traces.close()
    await store.close()

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
