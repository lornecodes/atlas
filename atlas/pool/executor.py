"""ExecutionPool — lifecycle-aware agent execution with warm slots."""

from __future__ import annotations

import asyncio
import time

from atlas.contract.registry import AgentRegistry
from atlas.contract.schema import validate_input, validate_output
from atlas.logging import get_logger
from atlas.orchestrator.default import DefaultOrchestrator
from atlas.orchestrator.protocol import Orchestrator
from atlas.pool.job import JobData
from atlas.pool.queue import JobQueue
from atlas.pool.slot_manager import SlotManager
from atlas.runtime.context import SpawnResult

logger = get_logger(__name__)


class ExecutionPool:
    """Manages agent execution with warm slot reuse and concurrency control.

    Delegates slot lifecycle to SlotManager. Handles job consumption,
    validation, execution, and scheduling.
    """

    def __init__(
        self,
        registry: AgentRegistry,
        queue: JobQueue,
        *,
        max_concurrent: int = 4,
        warm_pool_size: int = 2,
        idle_timeout: float = 300.0,
        warmup_timeout: float = 30.0,
        orchestrator: Orchestrator | None = None,
    ) -> None:
        self._registry = registry
        self._queue = queue
        self._max_concurrent = max_concurrent
        self._idle_timeout = idle_timeout
        self._orchestrator: Orchestrator = orchestrator or DefaultOrchestrator()

        self._slots = SlotManager(
            registry,
            warm_pool_size=warm_pool_size,
            warmup_timeout=warmup_timeout,
        )

        self._running_tasks: set[asyncio.Task] = set()
        self._consumer_task: asyncio.Task | None = None
        self._reaper_task: asyncio.Task | None = None
        self._started = False
        self._stopping = False
        self._semaphore: asyncio.Semaphore | None = None
        self._stop_lock = asyncio.Lock()

    async def start(self) -> None:
        """Start the pool — begins consuming from the queue."""
        self._started = True
        self._stopping = False
        self._semaphore = asyncio.Semaphore(self._max_concurrent)
        self._consumer_task = asyncio.create_task(self._consume_loop())
        self._reaper_task = asyncio.create_task(self._reaper_loop())
        logger.info("Pool started (max_concurrent=%d)", self._max_concurrent)

    async def stop(self, timeout: float = 10.0) -> None:
        """Graceful shutdown — stop consuming, drain running, shutdown warm agents."""
        async with self._stop_lock:
            if not self._started:
                return
            self._stopping = True
            logger.info("Pool stopping...")

            # Cancel consumer
            if self._consumer_task:
                self._consumer_task.cancel()
                try:
                    await asyncio.wait_for(self._consumer_task, timeout=2.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass

            # Cancel reaper
            if self._reaper_task:
                self._reaper_task.cancel()
                try:
                    await asyncio.wait_for(self._reaper_task, timeout=2.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass

            # Wait for running tasks
            if self._running_tasks:
                done, pending = await asyncio.wait(
                    self._running_tasks, timeout=timeout
                )
                for task in pending:
                    task.cancel()

            # Shutdown all warm agents
            await self._slots.shutdown_all()
            self._started = False
            logger.info("Pool stopped")

    async def submit(self, job: JobData) -> str:
        """Submit a job to the queue. Returns job ID."""
        return await self._queue.submit(job)

    async def _consume_loop(self) -> None:
        """Main consumer loop — pull jobs from queue and execute."""
        try:
            while not self._stopping:
                job = await self._queue.next()
                if self._stopping:
                    break
                await self._semaphore.acquire()
                task = asyncio.create_task(self._run_job(job))
                self._running_tasks.add(task)
                task.add_done_callback(lambda t: self._on_task_done(t))
        except asyncio.CancelledError:
            pass

    def _on_task_done(self, task: asyncio.Task) -> None:
        """Clean up when a job task completes."""
        self._running_tasks.discard(task)
        if self._semaphore:
            self._semaphore.release()

    async def _run_job(self, job: JobData) -> None:
        """Full lifecycle: route, acquire slot, validate, execute, release."""
        await self._queue.update(job.id, status="running", started_at=time.time())
        logger.debug("Running job %s (agent=%s)", job.id, job.agent_name)

        try:
            # Orchestrator routing
            decision = await self._orchestrator.route(job, self._registry)

            if decision.action == "reject":
                reason = decision.metadata.get("reason", "Rejected by orchestrator")
                await self._queue.update(
                    job.id,
                    status="failed",
                    error=reason,
                    completed_at=time.time(),
                )
                await self._orchestrator.on_job_failed(job)
                return

            if decision.action == "redirect" and decision.agent_name:
                logger.debug(
                    "Orchestrator redirecting job %s: %s → %s",
                    job.id, job.agent_name, decision.agent_name,
                )
                job.agent_name = decision.agent_name

            if decision.priority is not None:
                job.priority = decision.priority

            # Acquire a slot (warm or cold)
            slot, warmup_ms = await self._slots.acquire(job.agent_name)

            # Inject spawn context into the agent's context
            entry = self._registry.get(job.agent_name)
            ctx = slot.instance.context
            ctx.job_id = job.id
            ctx.depth = job.metadata.get("_spawn_depth", 0)
            ctx.spawn_allowed = entry.contract.requires.spawn_agents if entry else False
            ctx._spawn_callback = self._make_spawn_callback()

            # Validate input
            if entry:
                errors = validate_input(entry.contract, job.input_data)
                if errors:
                    await self._queue.update(
                        job.id,
                        status="failed",
                        error=f"Input validation: {'; '.join(errors)}",
                        completed_at=time.time(),
                        warmup_ms=warmup_ms,
                    )
        
                    await self._slots.release(slot)
                    return

            # Execute with timeout from contract
            slot.state = "busy"
            exec_start = time.monotonic()
            exec_timeout = entry.contract.execution_timeout if entry else 60.0
            try:
                output = await asyncio.wait_for(
                    slot.instance.execute(job.input_data),
                    timeout=exec_timeout,
                )
            except asyncio.TimeoutError:
                logger.error("Job %s timed out after %.1fs", job.id, exec_timeout)
                await self._queue.update(
                    job.id,
                    status="failed",
                    error=f"Execution timed out after {exec_timeout}s",
                    completed_at=time.time(),
                    warmup_ms=warmup_ms,
                    execution_ms=(time.monotonic() - exec_start) * 1000,
                )
    
                await self._slots.destroy(slot)
                return
            except Exception as e:
                logger.error("Job %s failed: %s", job.id, e)
                await self._queue.update(
                    job.id,
                    status="failed",
                    error=str(e),
                    completed_at=time.time(),
                    warmup_ms=warmup_ms,
                    execution_ms=(time.monotonic() - exec_start) * 1000,
                )
    
                await self._slots.destroy(slot)
                return

            execution_ms = (time.monotonic() - exec_start) * 1000

            # Validate output
            if entry:
                errors = validate_output(entry.contract, output)
                if errors:
                    await self._queue.update(
                        job.id,
                        status="failed",
                        error=f"Output validation: {'; '.join(errors)}",
                        output_data=output,
                        completed_at=time.time(),
                        warmup_ms=warmup_ms,
                        execution_ms=execution_ms,
                    )
        
                    await self._slots.release(slot)
                    return

            # Success
            await self._queue.update(
                job.id,
                status="completed",
                output_data=output,
                completed_at=time.time(),
                warmup_ms=warmup_ms,
                execution_ms=execution_ms,
            )

            slot.jobs_completed += 1
            await self._slots.release(slot)
            await self._orchestrator.on_job_complete(job)
            logger.debug("Job %s completed (exec=%.1fms, warmup=%.1fms)", job.id, execution_ms, warmup_ms)

        except Exception as e:
            logger.error("Pool error for job %s: %s", job.id, e)
            await self._queue.update(
                job.id,
                status="failed",
                error=f"Pool error: {e}",
                completed_at=time.time(),
            )
            await self._orchestrator.on_job_failed(job)


    def _make_spawn_callback(self):
        """Create a spawn callback bound to this pool."""
        async def _spawn(
            agent_name: str,
            input_data: dict,
            priority: int,
            parent_depth: int,
        ) -> SpawnResult:
            return await self._spawn_agent(agent_name, input_data, priority, parent_depth)
        return _spawn

    async def _spawn_agent(
        self,
        agent_name: str,
        input_data: dict,
        priority: int,
        parent_depth: int,
    ) -> SpawnResult:
        """Execute a spawned child agent through the full pool lifecycle."""
        child_job = JobData(
            agent_name=agent_name,
            input_data=input_data,
            priority=priority,
            metadata={"_spawn_depth": parent_depth + 1},
        )

        await self._queue.submit(child_job)

        # Wait for child to reach terminal state
        result = await self._queue.wait_for_terminal(child_job.id, timeout=60.0)

        if result and result.status == "completed":
            return SpawnResult(success=True, data=result.output_data or {})
        error = (result.error if result else "Child job timed out") or "Child job failed"
        return SpawnResult(success=False, error=error)

    async def _reaper_loop(self) -> None:
        """Periodically check for idle slots that have timed out."""
        try:
            while not self._stopping:
                await asyncio.sleep(min(self._idle_timeout / 2, 30.0))
                reaped = await self._slots.reap_idle(self._idle_timeout)
                if reaped:
                    logger.debug("Reaped %d idle slots", reaped)
        except asyncio.CancelledError:
            pass
