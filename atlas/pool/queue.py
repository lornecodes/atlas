"""Async priority queue for jobs — in-memory, no external dependencies."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from atlas.events import EventBus
    from atlas.store.job_store import JobStore

from atlas.pool.job import JobData


class QueueFullError(Exception):
    """Raised when the job queue is at capacity."""


class JobQueue:
    """In-memory async priority queue with status tracking.

    Jobs are dequeued in priority order (higher priority first).
    Ties are broken by creation time (FIFO).
    """

    def __init__(
        self,
        max_size: int = 1000,
        store: JobStore | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self._queue: asyncio.PriorityQueue[tuple[float, str]] = asyncio.PriorityQueue()
        self._jobs: dict[str, JobData] = {}
        self._waiters: dict[str, asyncio.Event] = {}  # job_id → terminal event
        self._max_size = max_size
        # O(1) counters — safe without locks because asyncio is single-threaded
        # (cooperative scheduling means no preemption between await points).
        self._active_count = 0  # Non-terminal jobs
        self._pending_count = 0
        self._running_count = 0
        self._store = store  # Optional JobStore for persistence
        self._event_bus = event_bus  # Optional EventBus for lifecycle events

    async def submit(self, job: JobData) -> str:
        """Add a job to the queue. Returns the job ID.

        Raises QueueFullError if the queue is at capacity.
        """
        if self._active_count >= self._max_size:
            raise QueueFullError(
                f"Queue is full ({self._max_size} jobs). "
                "Wait for jobs to complete or increase max_size."
            )
        if job.id in self._jobs:
            raise ValueError(f"Duplicate job ID: {job.id}")
        self._jobs[job.id] = job
        self._waiters[job.id] = asyncio.Event()
        self._active_count += 1
        self._pending_count += 1
        # Persist on submit
        if self._store:
            await self._store.save(job)
        # Priority queue is min-heap, so negate priority for max-priority-first
        score = (-job.priority, job.created_at)
        await self._queue.put((score, job.id))
        return job.id

    async def load_pending(self) -> int:
        """Reload pending jobs from store on restart. Returns count loaded."""
        if not self._store:
            return 0
        jobs = await self._store.list(status="pending")
        count = 0
        for job in jobs:
            if job.id not in self._jobs:
                self._jobs[job.id] = job
                self._waiters[job.id] = asyncio.Event()
                self._active_count += 1
                self._pending_count += 1
                score = (-job.priority, job.created_at)
                await self._queue.put((score, job.id))
                count += 1
        return count

    async def next(self) -> JobData:
        """Get the next job to execute (blocks until available)."""
        while True:
            _, job_id = await self._queue.get()
            job = self._jobs.get(job_id)
            if job and job.status == "pending":
                return job
            # Job was cancelled or removed while queued — skip it

    def get(self, job_id: str) -> JobData | None:
        """Get a job by ID."""
        return self._jobs.get(job_id)

    async def update(self, job_id: str, **fields: Any) -> None:
        """Update fields on a job."""
        job = self._jobs.get(job_id)
        if not job:
            return
        old_status = job.status
        was_terminal = job.is_terminal
        for key, value in fields.items():
            if hasattr(job, key):
                setattr(job, key, value)
        new_status = job.status

        # Maintain O(1) counters on status transitions
        if old_status != new_status:
            if old_status == "pending":
                self._pending_count = max(0, self._pending_count - 1)
            elif old_status == "running":
                self._running_count = max(0, self._running_count - 1)
            if new_status == "pending":
                self._pending_count += 1
            elif new_status == "running":
                self._running_count += 1

        # Decrement active count when job becomes terminal
        if not was_terminal and job.is_terminal:
            self._active_count = max(0, self._active_count - 1)

        # Persist updated state to store
        if self._store and old_status != new_status:
            await self._store.save(job)

        # Emit event on status change — subscribers process before waiters wake
        if old_status != new_status and self._event_bus:
            await self._event_bus.emit(job, old_status, new_status)

        # Signal waiters LAST so all subscribers have processed the event
        if job.is_terminal and job_id in self._waiters:
            self._waiters[job_id].set()
            del self._waiters[job_id]

    async def wait_for_terminal(self, job_id: str, timeout: float | None = None) -> JobData | None:
        """Block until a job reaches terminal state.

        Waits for the waiter event to fire, which guarantees that all
        side effects (store persistence, EventBus emission) have completed
        before returning.
        """
        event = self._waiters.get(job_id)
        if event:
            # Always wait on the event — it fires AFTER store save + bus emit
            try:
                await asyncio.wait_for(event.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                pass
            return self._jobs.get(job_id)
        # No waiter means job already reached terminal (waiter was cleaned up)
        return self._jobs.get(job_id)

    def list_all(self) -> list[JobData]:
        """List all jobs in the queue."""
        return list(self._jobs.values())

    def list_by_status(self, status: str) -> list[JobData]:
        """List all jobs with a given status."""
        return [j for j in self._jobs.values() if j.status == status]

    async def cancel(self, job_id: str) -> bool:
        """Cancel a pending job. Returns True if cancelled."""
        job = self._jobs.get(job_id)
        if job and job.status == "pending":
            old_status = job.status
            job.status = "cancelled"
            self._active_count = max(0, self._active_count - 1)
            self._pending_count = max(0, self._pending_count - 1)
            if job_id in self._waiters:
                self._waiters[job_id].set()
                del self._waiters[job_id]
            if self._event_bus:
                await self._event_bus.emit(job, old_status, "cancelled")
            return True
        return False

    @property
    def pending_count(self) -> int:
        """O(1) — maintained by submit/update/cancel."""
        return self._pending_count

    @property
    def running_count(self) -> int:
        """O(1) — maintained by update."""
        return self._running_count

    @property
    def capacity_remaining(self) -> int:
        """How many more jobs can be submitted."""
        return max(0, self._max_size - self._active_count)
