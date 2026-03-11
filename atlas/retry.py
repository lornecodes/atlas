"""RetrySubscriber — automatic retry with exponential backoff via EventBus."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from atlas.logging import get_logger
from atlas.pool.job import JobData

if TYPE_CHECKING:
    from atlas.contract.registry import AgentRegistry
    from atlas.pool.queue import JobQueue

logger = get_logger(__name__)


class RetrySubscriber:
    """EventBus subscriber that retries failed jobs with exponential backoff.

    Reads the RetrySpec from the agent's contract to decide whether and
    how to retry. Creates a new job linked to the original via
    original_job_id.
    """

    def __init__(self, queue: JobQueue, registry: AgentRegistry) -> None:
        self._queue = queue
        self._registry = registry

    async def __call__(
        self, job: JobData, old_status: str, new_status: str
    ) -> None:
        if new_status != "failed":
            return

        entry = self._registry.get(job.agent_name)
        if not entry:
            return

        spec = entry.contract.retry
        if spec.max_retries <= 0:
            return

        if job.retry_count >= spec.max_retries:
            logger.info(
                "Job %s exhausted retries (%d/%d)",
                job.id,
                job.retry_count,
                spec.max_retries,
            )
            return

        delay = spec.backoff_base * (2 ** job.retry_count)
        retry_count = job.retry_count + 1
        original_id = job.original_job_id or job.id

        logger.info(
            "Scheduling retry %d/%d for job %s in %.1fs",
            retry_count,
            spec.max_retries,
            job.id,
            delay,
        )

        asyncio.get_event_loop().call_later(
            delay,
            lambda: asyncio.ensure_future(
                self._submit_retry(job, retry_count, original_id)
            ),
        )

    async def _submit_retry(
        self, failed_job: JobData, retry_count: int, original_id: str
    ) -> None:
        """Submit a retry job to the queue."""
        retry_job = JobData(
            agent_name=failed_job.agent_name,
            input_data=dict(failed_job.input_data),
            priority=failed_job.priority,
            retry_count=retry_count,
            original_job_id=original_id,
        )
        job_id = await self._queue.submit(retry_job)
        logger.info(
            "Retry job %s submitted (attempt %d, original=%s)",
            job_id,
            retry_count,
            original_id,
        )
