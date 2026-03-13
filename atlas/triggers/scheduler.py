"""TriggerScheduler — async background loop that fires due triggers."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from atlas.logging import get_logger
from atlas.pool.job import JobData
from atlas.triggers.models import TriggerDefinition

if TYPE_CHECKING:
    from atlas.chains.executor import ChainExecutor
    from atlas.events import EventBus
    from atlas.pool.executor import ExecutionPool
    from atlas.store.trigger_store import TriggerStore

logger = get_logger(__name__)


class TriggerScheduler:
    """Polls the trigger store for due triggers and submits jobs to the pool.

    The scheduler runs as an async background task, ticking every
    ``poll_interval`` seconds. On each tick it queries the store for
    enabled, scheduled triggers whose ``next_fire`` has passed, fires
    them (submitting jobs to the pool), and updates their state.

    Webhook triggers are not polled — they fire via ``fire_webhook()``.
    """

    def __init__(
        self,
        store: TriggerStore,
        pool: ExecutionPool,
        chain_executor: ChainExecutor | None = None,
        event_bus: EventBus | None = None,
        poll_interval: float = 10.0,
    ) -> None:
        self._store = store
        self._pool = pool
        self._chain_executor = chain_executor
        self._event_bus = event_bus
        self._poll_interval = poll_interval
        self._task: asyncio.Task | None = None
        self._tick_lock = asyncio.Lock()
        self._running = False

    @property
    def running(self) -> bool:
        return self._running

    async def start(self) -> None:
        """Start the scheduler background loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._tick_loop())
        logger.info("Trigger scheduler started (poll_interval=%.1fs)", self._poll_interval)

    async def stop(self) -> None:
        """Stop the scheduler gracefully."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        logger.info("Trigger scheduler stopped")

    async def _tick_loop(self) -> None:
        """Main loop — tick every poll_interval seconds."""
        try:
            while self._running:
                await self._tick()
                await asyncio.sleep(self._poll_interval)
        except asyncio.CancelledError:
            pass

    async def _tick(self) -> None:
        """Check for due triggers and fire them."""
        if self._tick_lock.locked():
            return  # previous tick still running
        async with self._tick_lock:
            now = time.time()
            due = await self._store.list_due(before=now)
            for trigger in due:
                try:
                    job_id = await self._fire(trigger)
                    trigger.last_fired = now
                    trigger.fire_count += 1
                    trigger.last_job_id = job_id

                    if trigger.trigger_type == "one_shot":
                        trigger.enabled = False
                        trigger.next_fire = 0.0
                    else:
                        trigger.next_fire = trigger.compute_next_fire(now=now)

                    await self._store.save(trigger)
                    logger.info(
                        "Fired trigger %s (%s) → job %s",
                        trigger.id, trigger.name or trigger.target, job_id,
                    )
                except Exception:
                    logger.exception("Failed to fire trigger %s", trigger.id)

    async def _fire(self, trigger: TriggerDefinition) -> str:
        """Create and submit a job from a trigger definition. Returns job ID."""
        metadata = dict(trigger.metadata)
        metadata["_trigger_id"] = trigger.id
        metadata["_trigger_type"] = trigger.trigger_type

        if trigger.chain_name and self._chain_executor:
            from atlas.chains.definition import ChainDefinition
            # Look up chain — for now, create a minimal chain reference
            # Chain execution returns an execution ID, not a job ID
            exec_id = self._chain_executor.submit(
                ChainDefinition(name=trigger.chain_name, steps=[]),
                dict(trigger.input_data),
            )
            return exec_id

        job = JobData(
            agent_name=trigger.agent_name,
            input_data=dict(trigger.input_data),
            priority=trigger.priority,
            metadata=metadata,
        )
        return await self._pool.submit(job)

    async def fire_webhook(
        self,
        trigger_id: str,
        payload: dict[str, Any] | None = None,
    ) -> str:
        """Fire a webhook trigger with an optional payload.

        The payload is shallow-merged into the trigger's input_data
        (payload keys override trigger defaults).

        Returns the submitted job ID.
        """
        trigger = await self._store.get(trigger_id)
        if trigger is None:
            raise ValueError(f"Trigger {trigger_id!r} not found")
        if not trigger.enabled:
            raise ValueError(f"Trigger {trigger_id!r} is disabled")
        if trigger.trigger_type != "webhook":
            raise ValueError(
                f"Trigger {trigger_id!r} is type {trigger.trigger_type!r}, not webhook"
            )

        # Merge payload into input_data
        input_data = dict(trigger.input_data)
        if payload:
            input_data.update(payload)

        metadata = dict(trigger.metadata)
        metadata["_trigger_id"] = trigger.id
        metadata["_trigger_type"] = "webhook"

        job = JobData(
            agent_name=trigger.agent_name,
            input_data=input_data,
            priority=trigger.priority,
            metadata=metadata,
        )
        job_id = await self._pool.submit(job)

        now = time.time()
        trigger.last_fired = now
        trigger.fire_count += 1
        trigger.last_job_id = job_id
        await self._store.save(trigger)

        logger.info("Webhook trigger %s fired → job %s", trigger_id, job_id)
        return job_id

    async def fire_manual(self, trigger_id: str) -> str:
        """Manually fire any trigger immediately. Returns job ID."""
        trigger = await self._store.get(trigger_id)
        if trigger is None:
            raise ValueError(f"Trigger {trigger_id!r} not found")

        job_id = await self._fire(trigger)

        now = time.time()
        trigger.last_fired = now
        trigger.fire_count += 1
        trigger.last_job_id = job_id
        if trigger.is_recurring:
            trigger.next_fire = trigger.compute_next_fire(now=now)
        await self._store.save(trigger)

        return job_id
